import torchvision
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



class resnet18_reg(nn.Module):
    def __init__(self):
        super(resnet18_reg, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_features,512)
    def forward(self,x):
        out = self.model(x)
        #print(out.shape)
        out = torch.reshape(out, (256,2))
        return out
    
class resnet34_reg(nn.Module):
    def __init__(self):
        super(resnet34_reg, self).__init__()
        self.model = torchvision.models.resnet34(pretrained=True)
        self.num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_features,512)
    def forward(self,x):
        out = self.model(x)
        out = torch.reshape(out, (256,2))
        return out


# model = resnet18()
# model.to('cuda:0')
# model.train()
# optimizer = torch.optim.Adam(model.parameters())

# image = np.load('imagecbw.npy')
# position = np.load('labelcbw.npy')
# valimage = np.load('vimagecbw.npy')
# valposition = np.load('vlabelcbw.npy')

# criterion = torch.nn.MSELoss()

# bestloss = 100000
# vbestloss = 100000
# score = []
# valscore = []

# for epoch in range(0,150):
#     result = 0
#     optimizer.zero_grad()
#     progress_bar = tqdm(image)
#     model.train()
#     for i,k in enumerate(progress_bar):
#         if k is None:
#             break
#         img = np.stack((k,)*3, axis=0)
#         img = torch.tensor(img,dtype=torch.float,requires_grad=False).cuda().to('cuda:0')
#         img=img.unsqueeze(0)
#         #print(img.shape)
#         out = model(img)
#             #scoremap.retain_grad()
#             # out, N_matches = tracker.update(img, kpts, desc)
#         status = f"Epoch:{epoch:.1f}"
#         progress_bar.set_description(status)
#         label = torch.tensor(position[i],dtype=torch.float,requires_grad=False).cuda().to('cuda:0')
#         loss = criterion(out, label)
#         result = result + loss.item()
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#     if  result<bestloss:
#         bestloss = result
#         torch.save(model.state_dict(), 'train_resnet18.pth') 
#     print(result/len(image))
#     score.append(result/len(image))
    
    
    
#     vprogress_bar = tqdm(valimage)
#     result = 0
#     model.eval()
#     for i,k in enumerate(vprogress_bar):
#         if k is None:
#             break
        
#         img = np.stack((k,)*3, axis=0)
#         img = torch.tensor(img,dtype=torch.float,requires_grad=False).cuda().to('cuda:0')
#         img=img.unsqueeze(0)
#         #print(img.shape)
#         out = model(img)
#             #scoremap.retain_grad()
#             # out, N_matches = tracker.update(img, kpts, desc)
#         status = f"Epoch:{epoch:.1f}"
#         vprogress_bar.set_description(status)
#         label = torch.tensor(valposition[i],dtype=torch.float,requires_grad=False).cuda().to('cuda:0')
#         loss = criterion(out, label)
#         result = result + loss.item()
#     if  result<vbestloss:
#         vbestloss = result
#         torch.save(model.state_dict(), 'val_resnet18.pth') 
             
             
             
#     print(result/len(valimage))
#     valscore.append(result/len(valimage))
                
        
# print(score)
# plt.clf()
# plt.plot(score,label='train')
# plt.plot(valscore,label='val')
# plt.legend()
# plt.savefig('resnet18-508.png')
# plt.show()
