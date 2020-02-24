# Pytorch packages:
#  - torch: the top level packages
#  - torch.nn: contains layers, weights...
#  - torch.autograd: handles the derivative calculations
#  - torch.nn.functional: loss functions/activation functions/convolution operations
#  - torch.optim: optimization algorithms: SGD/Adam
#  - torch.utils: datasets/dataloaders
###########################################################################
#  CUDA:
#   - GPU are better than CPU if the computation can be done in paralel (CPUs can have 4/8/16 cores, comparing to GPUs which can have thousands of cores(higher GPUs have ≈3000 cores))
#   - NN are embarassingly parallel (could be easly broken into smaller computations: e.g.: Convolution operation)
#   - CUDA is a SW platform that pairs with the GPU platform
#   - CUDA is the SW layer that provides an API to the developers
#   - Don't use CUDA for simple tasks
###########################################################################
import torch

t = torch.tensor([[1, 2, 3],[4, 5, 6]])
# tensor([1, 2, 3]) is on CPU, by default
t = t.cuda()
# tensor([1, 2, 3], device='cuda:0') is on GPU (the first GPU)
###########################################################################
# Tensors:

# - number = scalar
# - array = vector
# - 2d-array = matrix
# - nd-tensor = nd-array

# Rank of a tensor: the number of dimensions (matrix = rank2)
# Shape of a tensor: 
t.shape # torch.Size([2, 3])
# Reshaping a tensor: 
t.reshape(1, 6).shape # torch.Size([1, 6])
###########################################################################
# Methods
import torch
import numpy as np

t = torch.Tensor()
print(t.dtype) # torch.float
print(t.device) # cpu
print(t.layout) # torch.strided : how our tensors data is laid out in memory
device = torch.device('cuda:0') # device(type='cuda', index=0)
###########################################################################
# ERRORS

t1 = torch.tensor([1, 2, 3])
t2 = torch.tensor([1., 2., 3.])
t1 + t2 # ERROR

t1 = torch.tensor([1, 2, 3])
t2= t1.cuda()
t1 + t2 # ERROR
###########################################################################
# Creation of tensors

data = np.array([1, 2, 3])
torch.Tensor(data) # tensor([1., 2., 3.])
torch.tensor(data) # tensor([1, 2, 3], dtype=torch,int32)
torch.as_tensor(data) # tensor([1, 2, 3], dtype=torch,int32)
torch.from_numpy(data) # tensor([1, 2, 3], dtype=torch,int32)

torch.eye() # identity tensor
# 1 0
# 0 1
torch.zeros(2, 2)
# 0 0
# 0 0
torch.ones(2, 2)
# 1 1
# 1 1
torch.rand(2, 2)
# 0.312 0.652
# 0.452 0.912







