import torch

a = torch.tensor([1, 10, 50, 100])
ret = torch.gather(a, 0, torch.tensor([3]).long())
print(ret)

t = torch.tensor([[1, 2], [3, 4]])
ret = torch.gather(t, 0, torch.tensor([[0, 1],[1,0]]))
print(ret)
