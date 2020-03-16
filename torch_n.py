import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


class Network(torch.nn.Module):
    def __init__(self):
	    super().__init__()
	    self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
	    self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
	    
	    self.fc1 = torch.nn.Linear(in_features=12 * 4 * 4, out_features=120)
	    self.fc2 = torch.nn.Linear(in_features=120, out_features=60)
	    
	    self.out = torch.nn.Linear(in_features=60, out_features=10)
    
    def forward(self, t):
        t = self.conv1(t)
        t = torch.nn.functional.relu(t)
        t = torch.nn.functional.max_pool2d(t, kernel_size=2, stride=2)
        
        t = self.conv2(t)
        t = torch.nn.functional.relu(t)
        t = torch.nn.functional.max_pool2d(t, kernel_size=2, stride=2)
        
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = self.torch.nn.functional.relu(t)
        
        t = self.fc2(t)
        t = self.torch.nn.functional.relu(t)
        
        t = self.out(t)
#       t = self.torch.nn.functional.softmax(t)

train_set = torchvision.datasets.FashionMNIST(
root = './data',
train=True,
download=True,
transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
)

network = Network()

train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
optimizer = torch.optim.Adam(network.parameters(), lr=0.01)

for epoch in range(5):

    total_loss = 0
    total_correct = 0

    for batch in train_loader:
        images, labels = batch

        preds = network(images)
        loss = torch.nn.functional.cross_entropy(preds, labels)

        optimizer.zero_grad() # to reset the gradient (it adds it up by default)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)

    print("epoch:", epoch, "total_correct:", total_correct, "loss:", total_loss)


print(total_correct / len(train_set)) # 0.7798375


