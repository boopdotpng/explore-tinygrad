# tinygrad is not that complicated
> Bounty:  Remove realize from \_\_setitem\_\_ and get TestSetitemLoop.test_arange to be one kernel

## stepping through test_arange
We will trace how the above operation gets realized into code:

A tensor in tinygrad is just a thin wrapper over a Directed Arcylic Graph of Uops. Every single tensor op defined in `Tensor.py` just generates Uops that are added to that Tensor's graph. `.sum()` is essentially just a reduce operation of type add on an axis with a reshape at the end (so you get a scalar) value. `Tensor.uop` stores this graph. 

The entry point is the sum method on Tensor. It defines the accumulation dtype and defines sum as a reduce op with type ADD; it also passes on the axis and keepdim arguments. 
```py
# tinygrad/Tensor.py
def sum(self, axis:int|Sequence[int]|None=None, keepdim=False, dtype:DTypeLike|None=None) -> Tensor:
  # axis = None, keepdim=False in our example
  ret = self.cast(sum_acc_dtype(self.dtype) if dtype is None else dtype)._reduce(Ops.ADD, axis, keepdim)
  return ret.cast(self.dtype) if dtype is None and self.dtype in (dtypes.float16, dtypes.bfloat16, *dtypes.fp8s) else ret
```

This is the generic helper function that handles all reduce ops (product, min, and max for example). A reduce op is any operation that collapses dimensions, producing a tensor of lower rank. In this case, sum is an addition of all the elements on an axis and the resulting shape is () unless keepdim is specified. Since we specified `axis=None`, it assumes we want to sum across every single axis.  
```py
# tinygrad/Tensor.py
def _reduce(self, op:Ops, axis:int|Sequence[int]|None=None, keepdim=False) -> Tensor:
  # op = Ops.Add, axis=None, keepdim=False
  # self.ndim = 2
  axis = tuple(self._resolve_dim(x) for x in (range(self.ndim) if axis is None else make_tuple(axis, 1)))
  # axis = (0,1) -- whole tensor
  if self.ndim == 0: axis = ()
  ret = self._apply_uop(UOp.r, op=op, axis=axis)
  return ret if keepdim else ret.reshape(tuple(s for i,s in enumerate(self.shape) if i not in axis))
```

`fxn` here is the Uop that you're applying (in this case Uop.r, which is a generic builder for all reduce ops). It creates `new_op` by calling UOp.r on the graphs of all tensors involved in the computation.
```py
# tinygrad/Tensor.py
def _apply_uop(self, fxn:Callable, *x:Tensor, extra_args=(), **kwargs) -> Tensor:
  # fxn = Uop.r, x = None (there is only one tensor in this case), kwargs includes op=Ops.Add, passed
  new_uop: UOp = fxn(*[t.uop for t in (self,)+x], *extra_args, **kwargs)
  if (metadata:=_METADATA.get()) is not None: all_metadata[new_uop] = (metadata,)
  needs_input_grad = [t.requires_grad for t in (self,)+x]
  return Tensor(new_uop, device=new_uop.device, requires_grad=True if any(needs_input_grad) else None if None in needs_input_grad else False)
```

```py
# tinygrad/uop/ops.py
def r(self, op:Ops, axis:tuple[int, ...]):
  axis = tuple(sorted([x for x in axis if resolve(self.shape[x] != 1)]))
  if len(axis) == 0: return self
  # move any non reduce axis before the first reduce axis
  move_early, rest = partition(range(axis[0], len(self.shape)), lambda i: i not in axis and resolve(self.shape[i] != 1))
  permaxis = tuple(range(axis[0])) + tuple(move_early) + tuple(rest)
  ret = self.permute(permaxis)
  new_axis = tuple([x for x in range(axis[0]+len(move_early), len(self.shape)) if resolve(ret.shape[x] != 1)])
  assert len(axis) == len(new_axis)
  # wraps all Uops passed in with a new reduce op. you can see op (the old op) is the child here 
  ret = UOp(Ops.REDUCE_AXIS, self.dtype, (ret,), (op, new_axis))
  return ret.reshape(tuple([x if i not in axis else 1 for i,x in enumerate(self.shape)]))
```

If you run with `DEBUG=4` you can see the full Uop graph produced by the sum operation.
```py
a = Tensor.rand(2048, 2048).realize()
# Since we don't really care about the `.rand()` part of the Uop graph, we realize the tensor so it just appears as a buffer to the next operation.
# .realize() runs through the entire pipeline and actually runs the kernel, leaving behind Uop.BUFFER in its place.
a = a.sum()
print(a.Uop)
```
```bash
UOp(Ops.RESHAPE, dtypes.float, arg=(), src=(
  UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (0, 1)), src=(
    UOp(Ops.RESHAPE, dtypes.float, arg=(2048, 2048), src=(
      UOp(Ops.BUFFER, dtypes.float, arg=4194304, src=(
        UOp(Ops.UNIQUE, dtypes.void, arg=8, src=()),
        UOp(Ops.DEVICE, dtypes.void, arg='METAL', src=()),)),)),)),))
```
Here, `arg` contains op-specific parameters (shape, constants, etc), `src` contains the parents of the node (i.e. what the op reads from). There is also a `children` property, which is an internal, back-ref set of nodes that *consume* this node. The topmost Op is reshape, because our `2048x2048` array turns into shape `()` after the sum (since we want a scalar value). Reduce axis is a reduction op of type Add, which was created in the `r` function. The args here are the axes (0,1) since we're summing every element in the Tensor of shape (n,n). 

You might wonder: "why is there a 'duplicate' reshape?" After you call `.realize()` on a Tensor, its shapetracker is absent or effectively 1-D. In the new Uop graph, you have to re-build that shape so that the rest of the graph knows what shape they're supposed to nest into. 
```py
# tinygrad/uop/ops.py
if self.op is Ops.RESHAPE and self.src[0].st is None:
  return ShapeTracker.from_shape(self.arg)
```
In this case, we created a `2048x2048` Tensor of random values before the sum and realized it. That's the `Ops.Buffer` you see in the graph above.

Ops.UNIQUE is a tiny leaf node that makes a buffer distinct. The `arg` value computed is just a monotonically increasing counter (0,1,2, ...). Since the Uop metaclass interns non-unique Uops and all buffers have to be unique, we need to identify it somehow. Ops.DEVICE just tells you where the buffer lives. Both of these are fed into Ops.BUFFER (you can think of these as parameters, kind of). This is the first step in the pipeline. 

The next step in the pipeline is grouping the Uop graph into kernels that can be lowered into code:
```py
# tinygrad/Tensor.py
def kernelize(self, *lst:Tensor) -> Tensor:
  big_sink = UOp.sink(*[x.uop for x in (self,)+lst])

  # verify Tensors match the spec
  if __debug__: type_verify(list(big_sink.toposort()), tensor_uop_spec)

  becomes_map = get_rangeify_map(big_sink) if RANGEIFY else get_kernelize_map(big_sink)
  _apply_map_to_tensors(becomes_map, name="Apply Kernelize Map")
  return self
```
> `RANGEIFY` is a new abstraction [being worked on](https://x.com/__tinygrad__/status/1964037572503752910). For now, we'll just go over the old codegen path and revisit `RANGEIFY` later.

First, we wrap all tensor.uop graphs in the operation with a sink Uop (which just marks the root nodes of the graph). Every rewrite can target one root instead of N separate tensors. 

All matching, rewriting, and transforming of Uop graphs in tinygrad is done using the `PatternMatcher` class and `UPat`s. These are written as "rewrite rules", which replace certain patterns in the graph with new patterns. Rewrite rules (UPats) are placed into a `PatternMatcher` and then applied to a graph. There are a *lot* of these scattered throughout tinygrad. This is where most of the optimization and speed comes from (the other part comes from BEAM, but we'll discuss that later.). 

Roughly, there are three optimization layers: kernelize optimizations, post-kernelize operations, and renderer-specific optimizations. 

**Kernelize PMs**:
- Multi (for multi-gpu) 
- Merge_views (view simplification) 
- do_fuse and pm_fuse
- kernelize_sym 
- replace_contiguous, add_contiguous, finalize_contiguous
- create_kernels
- replace_buffers, create_ast

**Optimization PMs** (most of these are in `symbolic.py`)
- propagate_invalid (check invalid index bounds)
- symbolic_simple (handles constant folding and a lot of common optimizations) 
- kernalize_sym (previous pattern matcher + more)
- commutative (flipping)
All of the above PatternMatchers are responsible for transforming the initial Uop graph into kernel ASTs.

**Post kernel PMs** 
- merge_views 
- view_left, view_right
- block_create
- pm_blockend_merge
- pm_lowerer, pm_reduce_simplify, 
These run on the kernel AST.

**Renderer specific PMs** 
There are also PMs that run based on which Renderer (hardware) that you're using to lower the kernel AST. You can find an example of this in `cstyle.py`. 

`TRACK_MATCH_STATS=1` prints a summary of which rules your Uop graph matched against and which PMs your graph went through before your program ends.

**Rangeify specific PMs** (will go over this later (todo!)) 

Consider: 

```py
from tinygrad import Tensor
a = Tensor.ones(10) * 15
b = Tensor.ones(10) * 30 
c = a+b
print(c.uop)
with Context(DEBUG=4):
  print(c.kernelize().uop)  
```
```bash
UOp(Ops.CONST, dtypes.float, arg=45.0, src=(
  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(10,), strides=(0,), offset=0, mask=None, contiguous=False),)), src=(
    UOp(Ops.DEVICE, dtypes.void, arg='METAL', src=()),)),))
```
No kernels were run for this computation. 
