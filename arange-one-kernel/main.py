from tinygrad import Tensor

def zeros_vs_empty():
  z = Tensor.zeros(4)
  print(z.uop)
  print(z.uop.st)
  print("e" + "\n"*3)
  e = Tensor.empty(4)
  print(e.uop)
  print(e.uop.st)

if __name__ == "__main__":
  # zeros_vs_empty()
  N = 2 
  cmp = Tensor.zeros(N).contiguous()
  for i in range(N): cmp[i] = i
  exit()
  a = Tensor.arange(4)
  print(a.realize())

  b = Tensor.zeros(4)
  print("b uop graph: ", b.uop)
  print("b cont st: ", b.contiguous().uop.st)
  print("b cont uop graph: ", b.contiguous().uop)
  b = b.contiguous()
  for i in range(4): b[i] = i
