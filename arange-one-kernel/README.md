# tinygrad bounty #1

> Remove realize from \_\_setitem\_\_ and get TestSetitemLoop.test_arange to be one kernel -- $200

## current behavior
Currently, in tinygrad, the setitem function in `tensor.py` realizes the tensor as soon as you write to it. This is not optimal because it runs one kernel for each write; you lose throughput and the ability to combine writes together into one bigger kernel. See an example:
```py
# DEBUG=4 to see generated kernels
N = 2 
cmp = Tensor.empty(N)
for i in range(N): cmp[i] = i
```
Scheduled 2 kernels in 0.71 ms:
```cpp
#include <metal_stdlib>
using namespace metal;
kernel void E(device float* data0_1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  *(data0_1+0) = 0.0f;
}
// *** METAL      1 E                                            arg  1 mem  0.00 GB tm      7.08us/     0.01ms (     0.00 GFLOPS    0.0|0.0     GB/s) ['__setitem__']
```
```cpp
#include <metal_stdlib>
using namespace metal;
kernel void En1(device float* data0_2, device float* data1_1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  float val0 = (*(data1_1+0));
  *(data0_2+0) = val0;
}
// *** METAL      2 En1                                          arg  2 mem  0.00 GB tm      4.00us/     0.01ms (     0.00 GFLOPS    0.0|0.0     GB/s) ['__setitem__']
```
```cpp
#include <metal_stdlib>
using namespace metal;
kernel void En2(device float* data0_1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  *(data0_1+0) = 1.0f;
}
// *** METAL      3 En2                                          arg  1 mem  0.00 GB tm      3.00us/     0.01ms (     0.00 GFLOPS    0.0|0.0     GB/s) ['__setitem__']
```

```cpp
#include <metal_stdlib>
using namespace metal;
kernel void En3(device float* data0_2, device float* data1_1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  float val0 = (*(data1_1+0));
  *(data0_2+1) = val0;
}
// *** METAL      4 En3                                          arg  2 mem  0.00 GB tm      3.62us/     0.02ms (     0.00 GFLOPS    0.0|0.0     GB/s) ['__setitem__']
```

You can see that 4 kernels were generated. Each `cmp[i] = i` turns into two kernels: 
1. Realize the rhs scalar i on the device. `E` writes `0.0f` into its own 1-element buffer. `En2` writes 1.0f into another 1-element buffer.
2. Copy that 1-element buffer into the destination slice.

The goal is to add to the Tensor's Uop graph whenever a write happens (lazily) instead of immediately writing the values. By the end, this should fuse into one kernel. 

### why?
```py
# Tensor.py (abridged)
def __setitem__(self, indices, v:Tensor|ConstType) -> None:
  # this line is responsible for realizing the constant into a Tensor on the device
  if isinstance(v, get_args(ConstType)): v = Tensor(v, device=self.device, dtype=self.dtype)

  # the realize!
  res = self.realize()._getitem(indices, v)

  if res.shape == self.shape and res.uop is not self.uop:
    self.assign(res).realize()
  else: # no copy, basic setitem
    v = v.cast(res.dtype)._broadcast_to(_broadcast_shape(res.shape, v.shape)).contiguous()
    # the other realize 
    res.assign(v).realize()
```

> `res = self.realize()._getitem(indices, v)` <br>

In order to know *where* the new Tensor should be written to, we force `self` to have actual device storage now. `getitem` produces a view of `self` (for basic indexing) or a rebuilt full tensor (for advanced). The fusion is essentially killed here. 

> `res.assign(v).realize()` <br>

This realize does the write now instead of lazily queueing it. 

## .assign() is broken on master 
See the following test: 
```py
A = Tensor.empty(4,4) 
B = Tensor.arange(16).reshape(4,4)  # [[0..3], [4..7], [8..11], [12..15]]
ret = A.permute(1,0).assign(B)
lst = ret.tolist() # intended return value is B
lst2 = A.tolist() # should return B.T
```
```py
# lst (ret)
[[0.0, 4.0, 8.0, 12.0], [1.0, 5.0, 9.0, 13.0], [2.0, 6.0, 10.0, 14.0], [3.0, 7.0, 11.0, 15.0]]
# lst2
[[0.0, 4.0, 8.0, 12.0], [1.0, 5.0, 9.0, 13.0], [2.0, 6.0, 10.0, 14.0], [3.0, 7.0, 11.0, 15.0]]
```
The behavior here is correct up until we call `.tolist()` to evaluate the Tensor. If we print the ShapeTracker for `B` and `ret`: 
```py
ShapeTracker(views=(View(shape=(4, 4), strides=(4, 1), offset=0, mask=None, contiguous=True),))
ShapeTracker(views=(View(shape=(4, 4), strides=(1, 4), offset=0, mask=None, contiguous=False),))
```
we can see the stride flip after the permute operation. The current output of `.tolist()` also confirms that the `.assign()` is writing *through* the permute. However, when we realize the Tensor, we get `B.T`, which tells us that the stride (1,4) was dropped somewhere along the way. The correct output for `lst` is just `B`, since you're setting `ret = B.T` and then reading the rows as columns. Those should cancel each other out and evaluate as `B`. 

To trace this issue and fix the `assign` behavior, we need to find how `assign` nodes are scheduled and lowered into kernels, since everything before that point is working fine. See [docs/ramp.py](https://github.com/boopdotpng/tinygrad/blob/master/docs/ramp.py) for how tinygrad lowers Tensor operations into gpu kernels. Most of the optimizations / rewrites / lowering in tinygrad is done using PatternMatchers, which rewrite the Uop graph underlying each Tensor using a set list of rules. We can find out which pattern matchers and rules were matched using the `TRACK_MATCH_STATS=2` environment variable. 

## possible solution 
i thought i could do this 
```py
def lazy_setitem(a, i, val):
  # [0,1,2,3,5]
  idx = Tensor.arange(a.shape[0] * a.shape[1]).reshape(a.shape)
  # if i=0, [t, f, f, f, f]
  mask = (idx == i)
  # just whatever val is 
  rhs = Tensor(val)
  return mask.where(rhs, a)
``` 

where you overwrite the current tensor with a bunch of mask updates 

so that way, if you write a loop: 
```py
a0 = Tensor.empty(6)
  for i in range(6):
    a0 = lazy_setitem(a0, i, i)
```
its lazy and it just builds a giant graph of CMPNEs and realizes into one kernel at the end, like this: (with DEBUG=5)
```bash
(tinygrad) ➜  real-tinygrad git:(master) ✗ DEBUG=5 python3 test_setitem.py
Using LLVM at '/opt/homebrew/opt/llvm@20/lib/libLLVM.dylib'
METAL: using MetalCompiler
opened device METAL from pid:49764
c0 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(6), arg=0, src=())
c1 = c0.view(ShapeTracker(views=(View(shape=(6,), strides=(1,), offset=0, mask=None, contiguous=True),)))
c2 = UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(6,), strides=(0,), offset=0, mask=None, contiguous=False),)), src=())
c3 = UOp.const(dtypes.int, 1, src=c2)
c4 = c3.view(ShapeTracker(views=(View(shape=(7, 11), strides=(0, 0), offset=0, mask=((0, 7), (5, 11)), contiguous=False), View(shape=(6, 6), strides=(1, 12), offset=0, mask=None, contiguous=False))))
c5 = c4.f(Ops.REDUCE_AXIS, arg=(Ops.ADD, (1,)))
c6 = c5.view(ShapeTracker(views=(View(shape=(6,), strides=(1,), offset=0, mask=None, contiguous=True),)))
c7 = (c6+UOp.const(dtypes.int, -1, src=c2))
c8 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(6), arg=1, src=())
c9 = c8.load()
c10 = c7.alu(Ops.CMPNE, UOp.const(dtypes.int, 5, src=c2)).alu(Ops.CMPNE, UOp.const(dtypes.bool, True, src=c2)).where(UOp.const(dtypes.float, 5.0, src=c2), c7.alu(Ops.CMPNE, UOp.const(dtypes.int, 4, src=c2)).alu(Ops.CMPNE, UOp.const(dtypes.bool, True, src=c2)).where(UOp.const(dtypes.float, 4.0, src=c2), c7.alu(Ops.CMPNE, UOp.const(dtypes.int, 3, src=c2)).alu(Ops.CMPNE, UOp.const(dtypes.bool, True, src=c2)).where(UOp.const(dtypes.float, 3.0, src=c2), c7.alu(Ops.CMPNE, UOp.const(dtypes.int, 2, src=c2)).alu(Ops.CMPNE, UOp.const(dtypes.bool, True, src=c2)).where(UOp.const(dtypes.float, 2.0, src=c2), c7.alu(Ops.CMPNE, c3).alu(Ops.CMPNE, UOp.const(dtypes.bool, True, src=c2)).where(UOp.const(dtypes.float, 1.0, src=c2), c7.alu(Ops.CMPNE, UOp.const(dtypes.int, 0, src=c2)).alu(Ops.CMPNE, UOp.const(dtypes.bool, True, src=c2)).where(UOp.const(dtypes.float, 0.0, src=c2), c9))))))
c11 = c1.store(c10)
ast = c11.sink()
upcasting masked axis : 0
#include <metal_stdlib>
using namespace metal;
kernel void E_6(device float* data0_6, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  *((device float4*)((data0_6+0))) = float4(0.0f,1.0f,2.0f,3.0f);
  *((device float2*)((data0_6+4))) = float2(4.0f,5.0f);
}
*** METAL      1 E_6                                          arg  1 mem  0.00 GB tm      7.38us/     0.01ms (     0.00 GFLOPS    0.0|0.0     GB/s) ['tolist', 'where', '__eq__', 'arange']
[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
```

you can see only one kernel is generated at the end 


but there are two issues with this: 
1. the graph is going to become huge if you have `for i in range(1000)` or something very large, we have to optimize those out somehow. you can see all the extra bloat we added to the graph at the end just for that loop
2. views don't work anymore! in tinygrad right now, views don't own data, its just a reference to a base expression and a shapetracker. setitem today is eager, meaning it writes and mutates the base buffer as soon as something changes. but if you make setitem lazy in this way, any existing view you create is bound to the old version of the item who you called setitem on. so it'll have the "wrong" value



