import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np

from fetcher import *

#X_train, Y_train, X_test, Y_test = read_file_both_sides_old('AramDataSet38W.txt', 'ChampionList624.txt', 134)
X_train, Y_train, X_test, Y_test = read_file_both_sides('AramDataSet624.txt', 'ChampionList624.txt', 134)

# Hyper Parameters 
input_size = 134*2
hidden_size = 1500
num_classes = 2
num_epochs = 100
batch_size = 256
learning_rate = 0.001

# Neural Network Model (1 hidden layer)
class Net(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc1.weight.data.normal_(0, 0.1)
		self.sigmoid = nn.Sigmoid()
		self.fc2 = nn.Linear(hidden_size, num_classes)  
		self.fc2.weight.data.normal_(0, 0.1)
		self.softmax = nn.Softmax()
	
	def forward(self, x):
		out = self.fc1(x)
		out = self.sigmoid(out)
		out = self.fc2(out)
		out = self.softmax(out)
		return out
		
net = Net(input_size, hidden_size, num_classes)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate) 

# 开始训练
for epoch in range(num_epochs):
	for i in range(0, len(X_train), batch_size):
		x_train = Variable(torch.FloatTensor(X_train[i:i+batch_size]))
		y_train = Variable(torch.LongTensor(Y_train[i:i+batch_size].astype(np.int64)))
		
		# forward
		optimizer.zero_grad()  # zero the gradient buffer
		outputs = net(x_train)
		loss = criterion(outputs, y_train)
		loss.backward()
		optimizer.step()
	
	correct = 0
	total = 0
	for i in range(0, len(X_test), batch_size):
		x_test = Variable(torch.FloatTensor(X_test[i:i+batch_size]))
		y_test = torch.LongTensor(Y_test[i:i+batch_size].astype(np.int64))
		outputs = net(x_test)
		predicted = torch.max(outputs.data, 1)[1]
		total += len(y_test)
		correct += (predicted == y_test).sum()

	print('Epoch[%d] : Accuracy of the network on the %d test cases: %f %%' % (epoch+1, total, 100 * correct / total))
