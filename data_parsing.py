#%%


import matplotlib.pyplot as plt
from scipy.io import wavfile
from pydub import AudioSegment
import numpy as np
import struct

import torch
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
# from model import Net


#%%
v1 = AudioSegment.from_file('audio/v1.m4a')
v2 = AudioSegment.from_file('audio/v2.m4a')
t1 = AudioSegment.from_file('audio/t1.m4a')
t2 = AudioSegment.from_file('audio/t2.m4a')

#%%

def __convertData(row):
    result = []
    width = 2
    for idx in range(0, len(row), width):
        raw = row[idx:idx+width]
        # if r == True:
            # data = float(struct.unpack('>h', raw)[0])
        # else:
        data = float(struct.unpack('<H', raw)[0])
            
        result.append(data)
    return np.array(result)

#%%

def draw(file, answer):
    frame_rate = file.frame_rate
    raw_data = file.get_array_of_samples()
    # plt.figure(figsize=(20, 3))
    # plt.plot(r)
    # plt.show()
    result = []
    t = (frame_rate / 1000 * 250)
    for i in range(int(len(raw_data) / t) - 1):
        r = file.get_array_of_samples()[int(t * i):int(t * (i + 1))]
        freq_sepctrum = np.fft.fft(r)
        freq = np.fft.fftfreq(len(freq_sepctrum), d= 1 / frame_rate)
        len_data = int(len(freq) / 2)
        x = freq[:len_data][:32*32]
        y = np.abs(freq_sepctrum[:len_data][:32*32])
        # plt.figure(figsize=(10,3)); plt.plot(x, y); plt.show()
        result.append({"data": torch.from_numpy(np.array(y).astype('float32')), "answer": torch.from_numpy(np.array(answer))})
        
    return np.array(result)
    
#%%

vv1 = draw(v1, 0)
vv2 = draw(v2, 1)
tt1 = draw(t1, 0)
tt2 = draw(t2, 1)

vv3 = np.append(vv1, vv2)
tt3 = np.append(tt1, tt2)


# print(len(vv1))
# print(len(r[0]))

#%% 



#%%

device = torch.device("cuda")

train_kwargs = {'batch_size': 1}
test_kwargs = {'batch_size': 1}
cuda_kwargs = {'num_workers': 1,
                'pin_memory': True,
                'shuffle': True}

train_kwargs.update(cuda_kwargs)
test_kwargs.update(cuda_kwargs)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(196, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        output = F.log_softmax(x, dim=1)
        return output



#%%

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for idx, elem in enumerate(train_loader):
        data, target = elem["data"], elem["answer"]
        
        data, target = data.reshape((1,32,32)).to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx * len(data), len(train_loader.dataset),
                100. * idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

#%%


model = Net().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=1.0)

scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
for epoch in range(10):
    train(model, device, vv3, optimizer, epoch)
    # test(model, device, tt3)
    scheduler.step()

torch.save(model.state_dict(), "mnist_cnn.pt")





#%%

# frame_width = v1.frame_width
# max_possible_amp = v1.max_possible_amplitude
# channels = v1.channels
# duration = v1.duration_seconds
# max_dBFS = v1.max_dBFS
# rms = v1.rms
# sample_width = v1.sample_width

# print(frame_rate, frame_width, max_possible_amp, channels, duration, max_dBFS, rms, sample_width)

# %%
