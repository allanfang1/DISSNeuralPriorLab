import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from PIL import Image 
import torchvision.transforms as transforms 
from matplotlib import pyplot as plt


class MyUNet(nn.Module):
    def __init__(self):
        super(MyUNet, self).__init__()
        self.conv1= nn.Conv2d(16, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)

        self.skip128 = nn.Conv2d(128, 128, 5, 1, 2)
        self.skip64 = nn.Conv2d(64, 64, 3, 1, 1)

        self.up1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(128+128, 64, 4, 2, 1)
        self.up3 = nn.ConvTranspose2d(64+64, 32, 4, 2, 1)
        self.up4 = nn.ConvTranspose2d(32, 3, 4, 2, 1)

        self.leakyRelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        d1 = self.leakyRelu(self.conv1(x))
        d2 = self.leakyRelu(self.conv2(d1))
        d3 = self.leakyRelu(self.conv3(d2))
        d4 = self.leakyRelu(self.conv4(d3))

        u1 = self.leakyRelu(self.up1(d4))
        s1 = self.leakyRelu(self.skip128(d3))
        c1 = torch.cat((u1, s1), dim=1)
        u2 = self.leakyRelu(self.up2(c1))
        s2 = self.leakyRelu(self.skip64(d2))
        c2 = torch.cat((u2, s2), dim=1)
        u3 = self.leakyRelu(self.up3(c2))
        u4 = self.up4(u3)
        x = self.sigmoid(u4)

        return x
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

myunet = MyUNet()
myunet.to(device)


input_img = Image.open('testcrop.jpg')
transform = transforms.Compose([transforms.PILToTensor()])
target = transform(input_img)/255
target = target.unsqueeze(0).float().to(device)

h = target.size()[2]
w = target.size()[3]
z = torch.rand(1,16, h, w).to(device)


#Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(myunet.parameters(), lr=1e-3)
losslog = []

#test
for i in range(0,2000):
    optimizer.zero_grad()
    output = myunet(z)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    losslog.append(loss.item())
    if i % 100 == 0:
        print("Iteration: {}, Loss: {:.4f}".format(i, losslog[-1] if losslog else 0))

output = myunet(z)

plt.imsave('final.jpg', output[0].cpu().detach().permute(1,2,0).numpy())

# display the loss
#plt.figure(figsize=(6,4))
#plt.yscale('log')
#plt.plot(losslog, label = 'loss ({:.4f})'.format(losslog[-1]))
#plt.xlabel("Epochs")
#plt.legend()
#plt.show()
#plt.close()
