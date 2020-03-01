###########################################################################
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
###########################################################################
# Differences
data = np.array([1, 2, 3])
torch.Tensor(data) # tensor([1., 2., 3.]) # Class constructor
torch.tensor(data) # tensor([1, 2, 3], dtype=torch,int32) # Factory function ( also as_tensor, from_numpy) -> Prefered 
torch.as_tensor(data) # tensor([1, 2, 3], dtype=torch,int32)
torch.from_numpy(data) # tensor([1, 2, 3], dtype=torch,int32)
# Set the data type
torch.tensor(np.array([1, 2, 3]), dtype=torch.float64) # tensor([1., 2., 3.], dtype=torch.float64)
# Change the array
data = np.array([0, 0, 0])
torch.Tensor(data) # tensor([1., 2., 3.]) -> Unchanged/Create an additional copy of the data in memory
torch.tensor(data) # tensor([1, 2, 3], dtype=torch,int32) -> Unchanged/Create an additional copy of the data in memory --> Most used
torch.as_tensor(data) # tensor([0, 0, 0], dtype=torch,int32) -> Changed/Share data --> Accepts any array
torch.from_numpy(data) # tensor([0, 0, 0], dtype=torch,int32) -> Changed/Share data --> Accepts only numpy arrays

# as_tensor doesn't work with built-in data structures like lists.
###########################################################################
import torch

t = torch.tensor([
  [1, 1, 1, 1],
  [2, 2, 2, 2],
  [3, 3, 3, 3]
], dtype=torch.float32)
# To find the shape:
t.size() # torch.Size([3, 4])
t.shape # torch.Size([3, 4])
# To see thwe number of elements
t.numel() # 12
# Squeezing and unsqueezing a tensor
t.reshape(1, 12).squeeze() # tensor([1., 1., 1., 2., 2... 3.])
t.reshape(1, 12).squeeze().unsqueeze(dim=0) # tensor([[1., 1., 1., 2., 2... 3.]])
# Flattening function:

def flatten(my_tensor):
	my_tensor = my_tensor.reshape(1, -1)
	my_tensor = my_tensor.squeeze()
	return my_tensor

t = torch.tensor([
  [1, 1, 1, 1],
  [2, 2, 2, 2]
], dtype=torch.float32)

flatten(t) # tensor([1, 1, 1, 1, 2, 2, 2, 2])
###########################################################################
# Image of an eight: 8x8 pixels
# 0 0 0 0 0 0 0 0 
# 0 0 0 x x 0 0 0 
# 0 0 x 0 0 x 0 0 
# 0 0 0 x x 0 0 0 
# 0 0 x 0 0 x 0 0 
# 0 0 x 0 0 x 0 0 
# 0 0 0 x x 0 0 0 
# 0 0 0 0 0 0 0 0 
###########################################################################
import torch

t1 = torch.tensor([
	[1, 1, 1, 1],
	[1, 1, 1, 1],
	[1, 1, 1, 1],
	[1, 1, 1, 1]
])

t2 = torch.tensor([
	[2, 2, 2, 2],
	[2, 2, 2, 2],
	[2, 2, 2, 2],
	[2, 2, 2, 2]
])

t3 = torch.tensor([
	[3, 3, 3, 3],
	[3, 3, 3, 3],
	[3, 3, 3, 3],
	[3, 3, 3, 3]
])

# Concatenate

t = torch.stack((t1, t2, t3))
t.shape # torch.Size([3, 4, 4]) # batch of 3 tensors with the height and weight of 4
# In order for a CNN to understand the imput (it expects also a color channel, we need to reshape the tensor:
t = t.reshape(3, 1, 4 ,4)
3 - image; 1 - color channel; 4 - rows of pixels; 4 - pixels per row
# When working with CNNs, flattening is required. Flattening examples:
t = torch.tensor([[
	[1, 1],
	[2, 2]],
	[[3, 3],
	 [4, 4]]]

t.reshape(1, -1)[0]
t.reshape(-1)
t.view(t.numel())

t.flatten() # Flatten all the 3 images ( we don't want that)
# tensor([1, 1, 2, 2, 3, 3, 4, 4])
t.flatten(start_dim=1) # Flatten all the 3 images ( we don't want that)
# tensor([[1, 1, 2, 2],
	[3, 3, 4, 4]])

###########################################################################
# Element wise operations:
# The 2 tensors needs to have the same shape to perform an element wise operation:
t1 = torch.tensor([
	[1, 2],
	[3, 4]])
t2 = torch.tensor([
	[9, 8],
	[7, 6]])
t1 + t2 # or t1.add(t2)
# 10.0 10.0
# 10.0 10.0

# Broadcasting:
# t1 + 2 means that the 2 is broadcasted:
np.broadcast_to(2, t1.shape)
# array([2, 2],
	[2, 2]])

a = array([10, 5, -1])
print(a>0)
# array([True, True, False], dtype=bool)

t1 = torch.tensor([
	[1, 2],
	[3, 4]])
t2 = torch.tensor([9, 8])
np.broadcast_to(t2, t1.shape)
# array([9, 8],
	[9, 8]])
###########################################################################
# Reduction operations: Is an operations which reduces the number of elements

import torch
import numpy as np

t = torch.tensor([
	[0, 1, 0],
	[2, 0, 2],
	[0, 3, 0]
], dtype=torch.float32)
# Sum
t.sum() # tensor(8.)
t.numel() # 9
t.sum().numel # 1

# Product, mean, std
t.prod() # tensor(0.)
t.mean() # tensor(0.8889)
t.std() # tensor(1.1667)

# Reduce a specific axis

t = torch.tensor([
	[1, 1, 1, 1],
	[2, 2, 2, 2],
	[3, 3, 3, 3]
], dtype=torch.float32)
# Sum
t.sum(dim=0) # tensor([6., 6., 6., 6.])
t.sum(dim=1) # tensor([4., 8., 12.]) 
# Argmax function: Tells the index location of the maximum value inside a tensor
t = torch.tensor([
	[1, 1, 1, 2],
	[3, 2, 2, 2],
	[4, 3, 1, 5]
], dtype=torch.float32)
t.max() # tensor(5.)
t.argmax() # tensor(11)
t.flatten() #  t = torch.tensor([1, 1, 1, 2, 3, 2, 2, 2, 4, 3, 1, 5])
t.max(dim=0) # the max values followed by the indexes: (tensor([4., 3., 2., 5.]), tensor([2., 2., 1., 2.]))
t.argmax(dim=0) # Only the indexes: tensor([2., 2., 1., 2.])
# What we do when we need the value:
t.max() # tensor(5.)
t.max().item() # 5.0
# What we do when we need the values:
t.mean(dim=0) # tensor([2.6, 2., 1.3, 3.])
t.mean(dim=0).tolist() # [2.6, 2., 1.3, 3.]
t.mean(dim=0).numpy() # array([2.6, 2., 1.3, 3.], dtype=float32) 
###########################################################################
# MNIST data set transformations:
# PNG
# trimming
# resizing
# sharpening
# extending
# negating
# grayscaling
###########################################################################
# 4 steps of AI implementation
# Prepare the data : ETL process (Extract; Transform ; Load)
# Build the model
# Train the model
# Analyze the model's results
###########################################################################
class OHLC(Dataset):
	def __init__(self, csv_file):
		self.data = pd.read_csv(csv_file)

	def __getitem__(self, index):
		r = self.data.iloc[index]:
		label = torch.tensor(r.is_up_day, dtype=torch.long)
		sample = self.normalize(torch.tensor([r.open, r.high, r.low, r.close]))
		return sample, label

	def __len__(self):
		return len(self.data)
###########################################################################
# Loading the data
import torch # top level pytorch package
import torchvision # acces to popular datasets and image transformation
import torchvision.transforms as transforms # gives acces to common transformation to image processing

train_set = torchvision.datasets.FashionMNIST(
	root='./data/FashionMNIST', # the location on disk where the data is located
	train=True, # the train set
	download=True, # tells the class to download the data if it's not present at the location we precised
	transforms=transforms.Compose([ # Compose class allows us to do more than 1 transformations
		transforms.ToTensor() # we need tensors
	])
)

train_loader = torch.utils.data.Dataloader(train_set, batch_size=10) # we can shuffle, have batch size ( quarrying )
###########################################################################
# Visualize the data
import numpy as np
import matplotlib.pyplot as plt

torch.set_printoptions(linewidth=120)
train_set.train_labels # tensor([9, 0, 0 ... 5])
train_set.train_labels.bincount() # tensor([6000, 6000, ...., 6000]) the frequency of the data

# One sample
sample = next(iter(train_set))
len(sample) # 2
type(sample) # tuple
image, label = sample
image.shape # torch.Size([1, 28, 28])
label.shape # torch.Size([]) ; scalar
plt.imshow(image.squeeze(), cmap='gray')

# One batch
batch = next(iter(train_loader))
len(batch) # 2
type(batch) # list
images, labels = batch
images.shape # torch.Size([10, 1, 28, 28])
label.shape # torch.Size([10]) # rank one tensor
grid = torchvision.utils.make_grid(images, nrows=10)

plt.figure(figsize=(15, 15))
plt.imshow(np.transpose(grid, (1, 2, 0)))
###########################################################################
# Build the model: with torch.nn
# In CNNs: kernel = filter
import torch.nn as nn

class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()

		self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
		self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

		self.fc1 = nn.Linear(in_features=12*4*4, out_features=120) # fully connected / Dense / Linear layers
		self.fc2 = nn.Linear(in_features=120, out_features=60)
		self.out = nn.Linear(in_features=60, out_features=10)


	def forward(self, t):
		t = self.layer(t)
		return t

	def __repr__(self):
		return "overriten above nn.Module"
###########################################################################
# Learnable parameters in nn
network = Network()
print(network)
# Network(
# 	(conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1)) # the stride is the sliding of the filter after each computation
# 	(conv2): Conv2d(6, 12, kernel_size=(5, 5), stride=(1, 1))
# 	(fc1): Linear(in_features=192, out_features=120, bias=True) # the bias
# 	(fc2): Linear(in_features=120, out_features=60, bias=True)
# 	(out): Linear(in_features=60, out_features=10, bias=True)
# )
or
print(network.conv1)
# 	(conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
# To check the weights
























