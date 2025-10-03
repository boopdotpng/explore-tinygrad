# Remove realize from \_\_setitem\_\_ and get TestSetitemLoop.test_arange to be one kernel

In current tinygrad (080b26e7d7392e5e4ba844b7d2c26a99eef98e21), the setitem function in `tensor.py` realizes the tensor as soon as you write to it. This is not optimal because it runs one kernel for each write; you lose throughput and the ability to combine writes together into one bigger kernel. See an example:
```py
# DEBUG=4 to see generated kernels
N = 2 
cmp = Tensor.empty(N)
for i in range(N): cmp[i] = i
```
```bash
scheduled 2 kernels in 0.71 ms
#include <metal_stdlib>
using namespace metal;
kernel void E(device float* data0_1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  *(data0_1+0) = 0.0f;
}
*** METAL      1 E                                            arg  1 mem  0.00 GB tm      7.08us/     0.01ms (     0.00 GFLOPS    0.0|0.0     GB/s) ['__setitem__']
#include <metal_stdlib>
using namespace metal;
kernel void En1(device float* data0_2, device float* data1_1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  float val0 = (*(data1_1+0));
  *(data0_2+0) = val0;
}
*** METAL      2 En1                                          arg  2 mem  0.00 GB tm      4.00us/     0.01ms (     0.00 GFLOPS    0.0|0.0     GB/s) ['__setitem__']
scheduled 2 kernels in 0.65 ms
#include <metal_stdlib>
using namespace metal;
kernel void En2(device float* data0_1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  *(data0_1+0) = 1.0f;
}
*** METAL      3 En2                                          arg  1 mem  0.00 GB tm      3.00us/     0.01ms (     0.00 GFLOPS    0.0|0.0     GB/s) ['__setitem__']
#include <metal_stdlib>
using namespace metal;
kernel void En3(device float* data0_2, device float* data1_1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  float val0 = (*(data1_1+0));
  *(data0_2+1) = val0;
}
*** METAL      4 En3                                          arg  2 mem  0.00 GB tm      3.62us/     0.02ms (     0.00 GFLOPS    0.0|0.0     GB/s) ['__setitem__']

```


## current output
> $ DEBUG=3 python3 test/test_setitem.py TestSetitemLoop.test_arange
*** METAL      1 E                                            arg  1 mem  0.00 GB tm      7.50us/     0.01ms (     0.00 GFLOPS    0.0|0.0     GB/s) ['__setitem__']











## why?
```py
def __setitem__(self, indices, v:Tensor|ConstType) -> None:
  if isinstance(self.device, str) and self.device.startswith("DISK"):
    self.realize()._getitem(indices).assign(v)
    return
  # NOTE: check that setitem target is valid first
  if not unwrap(self.uop.st).contiguous: raise RuntimeError("setitem target needs to be contiguous")
  if isinstance(v, get_args(ConstType)): v = Tensor(v, device=self.device, dtype=self.dtype)
  if not isinstance(v, Tensor): raise TypeError(f"can't set a {type(v).__name__} to a Tensor")
  if self.requires_grad or v.requires_grad: raise NotImplementedError("setitem with requires_grad is not supported")

  res = self.realize()._getitem(indices, v)
  # if shapes match and data is not shared it's a copy and we assign to self
  if res.shape == self.shape and res.uop is not self.uop:
    self.assign(res).realize()
  else: # no copy, basic setitem
    v = v.cast(res.dtype)._broadcast_to(_broadcast_shape(res.shape, v.shape)).contiguous()
    res.assign(v).realize()
```

### rant 

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



