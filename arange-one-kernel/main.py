from tinygrad import Tensor
from tinygrad.helpers import Context

def test_broken_assign():
  A = Tensor.empty(4,4) 
  B = Tensor.arange(16).reshape(4,4) 
  print(B.uop.st)
  ret = A.permute(1,0).assign(B) # should return B 
  lst = ret.tolist()
  lst2 = A.tolist()
  print(lst)
  print(lst2)

def track_matches_min():
  A = Tensor.empty(4,4) 
  B = Tensor.arange(16).reshape(4,4) 
  ret = A.permute(1,0).assign(B) # should return B 
  lst = ret.tolist()
  print(lst)

def test_kernel_fusion():
  N = 4
  N = 2 
  cmp = Tensor.empty(N)
  for i in range(N): cmp[i] = i

if __name__ == "__main__":
  # DEBUG=4 to see generated kernels
  # with Context(DEBUG=4):
  #   N = 2 
  #   cmp = Tensor.empty(N)
  #   for i in range(N): cmp[i] = i
  # track_matches_min()
  test_broken_assign()
  # test_kernel_fusion()
