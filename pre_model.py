""" This file includes a preliminary model """
import torch
import torch.nn as nn

# import data
force_tensor = torch.load("data/force_tensor.pt")
aev_tensor = torch.load("data/aev_tensor.pt")
print(force_tensor.size(), aev_tensor.size())

# hyper parameters
input_size = aev_tensor.size()[1]
output_size = force_tensor.size()[1]
hidden_size_1 = 512
hidden_size_2 = 128
hidden_size_3 = 64

epochs = 32
batch_size = 100
learning_rate = 0.0001

print(input_size)


class Network(nn.Module):
    """ construct a simple nn """
    def __init__(self):
        super(Network, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size_1)
        self.activate_1 = nn.ReLU()
        self.drop_1 = nn.Dropout(p=0.2)
        self.l3 = nn.Linear(hidden_size_1, hidden_size_2)
        self.activate_2 = nn.ReLU()
        self.l5 = nn.Linear(hidden_size_2, hidden_size_3)
        self.activate_3 = nn.ReLU()
        self.l7 = nn.Linear(hidden_size_3, output_size)

    def forward(self, x):
        x = self.l1(x)
        x = self.activate_1(x)
        x = self.drop_1(x)
        x = self.l3(x)
        x = self.activate_2(x)
        x = self.l5(x)
        x = self.activate_3(x)
        x = self.l7(x)
        return x


net = Network()
net.cuda()
print(net)

loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

loss_log = []

for e in range(epochs):
    for i in range(0, aev_tensor.size()[0], batch_size):
        x_mini = aev_tensor[i: i + batch_size].float().item()
        y_mini = force_tensor[i: i + batch_size].float().item()

        optimizer.zero_grad()
        net_out = net(x_mini)

        loss = loss_fn(net_out, y_mini)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            loss_log.append(loss.data[0])

    print('Epoch: {} - Loss: {:.6f}'.format(e, loss.data[0]))
