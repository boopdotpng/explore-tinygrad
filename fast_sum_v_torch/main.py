from tinygrad import Tensor
from tinygrad.codegen import full_rewrite_to_sink
from tinygrad import TinyJit
from tinygrad import Context

def multiple_tensor():
  a = Tensor.rand(4).realize()
  b = Tensor.ones(4).realize()
  c = (a+b)
  print(c.uop)
  print("\n"*5, "kernelized:", c.kernelize().uop)
  with Context(DEBUG=4):
    c.realize()

def kern_example():
  rand_buffer = (Tensor.rand(2048, 2048)).realize() # realize to avoid polluting the graph
  b = rand_buffer.sum() 
  print("\n\nuop graph:")
  print(b.uop)
  print("\n\nkernelized:")
  b_kernelized = b.kernelize()
  print(b_kernelized.uop)
  # rewrite / optimization loop
  # optimized = full_rewrite_to_sink(b_kernelized)

def constant_folding():
  a = Tensor.ones(10) * 15
  b = Tensor.ones(10) * 30 
  c = a+b
  print(c.uop)
  # for some reason track match stats doesn't work in the Context helper. 
  # will investigate later 
  with Context(DEBUG=4, TRACK_MATCH_STATS=2):
    print(c.kernelize().uop)

if __name__  == "__main__":
  constant_folding()
