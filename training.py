import linecache
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm as tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt
import argparse
from numpy import pi
from torch import cos,sin

parser = argparse.ArgumentParser()
parser.add_argument("--b", help="batch size",default=6400,type=int)
parser.add_argument("--wd",help="weight deccay",default=0,type=float)
parser.add_argument('--lr',help="learning rate",default=1e-3,type=float)
parser.add_argument('--n',help="instance number",default=0,type=int)
parser.add_argument('--s',help="network size",default=10,type=int)
parser.add_argument('--nl',help='Non Linear layer',default='tanh',type=str)
args = parser.parse_args()

sys.path.append('')
#emptying previous loss file
path = 'C:\\Users\\Quentin Delignon\\Documents\\Python\\PJE\\inverse_kin\\loss_%d.txt'%(args.n)
loss_file = open(path,'w')
loss_file.write('')
loss_file.close()

class MyDataset(torch.utils.data.Dataset):
    def __init__(self,mode='train'):
        self.mode = mode
        if self.mode =='train':
            self.path = 'C:\\Users\\Quentin Delignon\\Documents\\Python\\PJE\\inverse_kin\\data_train.txt'
        else:
            self.path = 'C:\\Users\\Quentin Delignon\\Documents\\Python\\PJE\\inverse_kin\\data_val.txt'
        self.f = open(self.path,'r')
        self.length = len(self.f.readlines())
        self.f.close()

    def __len__(self):
        return self.length-1

    def __getitem__(self,n):
        line = linecache.getline(self.path,n+1)
        #line = line.split(';')
        pos = line.split(',')[:-1]#;q=line[1].split(',')[:-1]
        pos = torch.tensor([float(i) for i in pos])
        #q = torch.tensor([float(i) for i in q])
        return pos

"""
class InvKinNet(nn.Module):
    def __init__(self,nb_layer):
        #on génère e nombre de couche
        #on prend les 5 angles en entrée
        #on veut obtenir la position cartésienne en sortie
        super(InvKinNet,self).__init__()
        nb_layer = int(nb_layer//2*2)
        sizes = np.linspace(6,200,(nb_layer)//2)
        sizes2 = np.linspace(200,10,(nb_layer)//2)
        self.layers = []
        for i in range(len(sizes)-1):
            #self.layers.append(nn.BatchNorm1d(1))
            self.layers.append(nn.Linear(int(sizes[i]),int(sizes[i+1])))
            self.layers.append(nn.Tanh())
        for i in range(len(sizes2)-1):
            #self.layers.append(nn.BatchNorm1d(1))
            self.layers.append(nn.Linear(int(sizes2[i]),int(sizes2[i+1])))
            if i != len(sizes2)-2:
              self.layers.append(nn.Tanh())
        #self.layers = tuple(self.layers)
        self.sequence = nn.Sequential(*self.layers)
    def forward(self,x):
        x = self.sequence(x)
        #x = torch.fmod(x+2*np.pi,2*np.pi)
        return x
"""
class InvKinNet(nn.Module):
    def __init__(self,nb_layers,nl='tanh'):
        super(InvKinNet,self).__init__()
        if nl == 'tanh':
          self.nl = nn.Tanh()
        if nl == 'hardtanh':
          self.nl = nn.Hardtanh(min_val = -1,max_val=1)
        if nl == 'relu':
          self.nl = nn.ReLU()
        self.fc1 = nn.Sequential(*(nn.Linear(3,30),self.nl))
        self.bloc1 = self.bloc(30,nb_layers)
        self.fc2 = nn.Sequential(*(nn.Linear(30,50),self.nl))
        self.bloc2 = self.bloc(50,nb_layers)
        self.fc3 = nn.Sequential(*(nn.Linear(50,60),self.nl))
        self.bloc3 = self.bloc(60,nb_layers)
        self.fc4 = nn.Sequential(*(nn.Linear(60,50),self.nl))
        self.bloc4 = self.bloc(50,nb_layers)

        self.fc5 = nn.Sequential(*(nn.Linear(50,30),self.nl))
        self.bloc5 = self.bloc(30,nb_layers)
        self.fc6 = nn.Sequential(*(nn.Linear(30,20),self.nl))
        self.bloc6 = self.bloc(20,nb_layers)
        self.fc7 = nn.Sequential(*(nn.Linear(20,5),self.nl))
        self.bloc7 = self.bloc(5,nb_layers,lin=True)

        self.fc5_b = nn.Sequential(*(nn.Linear(50,30),self.nl))
        self.bloc5_b = self.bloc(30,nb_layers)
        self.fc6_b = nn.Sequential(*(nn.Linear(30,20),self.nl))
        self.bloc6_b = self.bloc(20,nb_layers)
        self.fc7_b = nn.Sequential(*(nn.Linear(20,5),self.nl))
        self.bloc7_b = self.bloc(5,nb_layers,lin=True)

    def bloc(self,n_out,nbl,lin=False):
        out = []
        n_out = int(n_out)
        for k in range(nbl):
          out.append(nn.Linear(n_out,n_out))
        if not (lin==True and k == nbl-1):
            out.append(self.nl)
        out = nn.Sequential(*out)
        return out
    def forward(self,x):
        x = self.fc1(x)
        x_prev = x
        x = self.bloc1(x)
        x = x+x_prev
        x = self.fc2(x)
        x_prev = x
        x = self.bloc2(x)
        x = x+x_prev
        x = self.fc3(x)
        x_prev = x
        x = self.bloc3(x)
        x = x+x_prev
        x = self.fc4(x)
        x_prev = x
        x = self.bloc4(x)

        x_mid = x+x_prev

        x = self.fc5(x_mid)
        x_prev = x
        x = self.bloc5(x)
        x = x+x_prev
        x = self.fc6(x)
        x_prev = x
        x = self.bloc6(x)
        x = x+x_prev
        x = self.fc7(x)
        x_prev = x
        x = self.bloc7(x)
        x1 = x+x_prev

        x = self.fc5_b(x_mid)
        x_prev = x
        x = self.bloc5_b(x)
        x = x+x_prev
        x = self.fc6_b(x)
        x_prev = x
        x = self.bloc6_b(x)
        x = x+x_prev
        x = self.fc7_b(x)
        x_prev = x
        x = self.bloc7_b(x)
        x2 = x+x_prev
        x = torch.cat((x1,x2),dim=2)
        if self.nl == 'relu':
            x = x - torch.ones_like(x)
        return x


def decode(x):
  c = x[:,:,:5]; s = x[:,:,5:]
  res = torch.atan2(s,c)
  #res = torch.fmod(res +2*np.pi,2*np.pi)
  return res

def encode(q):
  c = torch.cos(q)
  s = torch.sin(q)
  return torch.cat((c,s),dim=2)


def encode2(q):
  c = torch.cos(q)
  s = torch.sin(q)
  res = torch.cat((c,s),dim=1).unsqueeze(1)
  res = F.unfold(res,(2,1)).permute(0,2,1)
  print(res.shape)
  res = res.reshape(-1,10)
  print(res.shape)
  return res




def plot():
  loss_file = open(path,'r')
  data = loss_file.readlines()
  loss_file.close()
  for k in range(len(data)):
      data[k] = data[k].split(';')
      for i in range(len(data[k])):
          if i ==0:
              data[k][i] = int(data[k][i])
          else:
              data[k][i] = float(data[k][i])
  data = np.array(data)
  graph.scatter(data[:,0],data[:,1],c='b',label='Train')
  graph.scatter(data[:,0],data[:,2],c='orange',label='Val')
  if noLegend == True:
      graph.legend()
  #graph_angle.scatter(data[:,0],data[:,2],c='b',label ='Train')
  #graph_angle.scatter(data[:,0],data[:,4],c='orange',label='Val')
  #if noLegend == True:
      #graph_angle.legend()
  plt.savefig('C:\\Users\\Quentin Delignon\\Documents\\Python\\PJE\\inverse_kin\\loss_%d.png'%(args.n))

def myMSE(out,target):
  res = torch.abs(out - target)
  res2 = -res+2*np.pi
  res[res>res2] = res2[res>res2]
  res = res*180/np.pi
  res = res*res
  res = torch.mean(res)
  return res

def myMAE(out,target):
  res = torch.abs(out - target)
  res[res>np.pi] = -res[res>np.pi]+2*np.pi
  res = torch.mean(res)
  return res

def huber_mod(out,target,thresh):
  beta = thresh**2-thresh
  diff = out - target
  test = torch.abs(diff)
  res = test + beta
  res[test<thresh] = diff[test<thresh]*diff[test<thresh]
  return torch.mean(res)
def myMSE(out,target):
  res = torch.abs(out - target)
  #res = 100*(1-torch.cos(res))**2
  res2 = -res+2*np.pi
  res[res>res2] = res2[res>res2]
  res = res * 180/np.pi
  res = res*res
  res = torch.mean(res)
  return res.detach()

def myMAE(out,target):
  res = torch.abs(out - target)
  res2 = -res+2*np.pi
  res[res>res2] = res2[res>res2]
  res = torch.mean(res)
  return res

def huber_mod(out,target,thresh):
  beta = thresh**2-thresh
  diff = out - target
  test = torch.abs(diff)
  res = test + beta
  res[test<thresh] = diff[test<thresh]*diff[test<thresh]
  return torch.mean(res)

def getRot(q):
  c1 = cos(q[:,:,0]) ; c234 = cos(q[:,:,1]+q[:,:,2]+q[:,:,3]) ; c5 = cos(q[:,:,4])
  s1 = sin(q[:,:,0]) ; s234 = sin(q[:,:,1]+q[:,:,2]+q[:,:,3]) ; s5 = sin(q[:,:,4])
  l1 = torch.cat((c1*c234*c5+s1*s5, -c1*c234*s5+s1*c5, -c1*s234),dim=1).unsqueeze(2)
  l2 = torch.cat((c1*c234*c5-s1*s5, -s1*c234*s5-c1*c5, -s1*s234),dim=1).unsqueeze(2)
  l3 = torch.cat((-s234*c5,        s234*s5,            -c234),dim=1).unsqueeze(2)
  mat = torch.cat((l1,l2,l3),dim=2)
  angle = 2*torch.acos((mat[:,0,0]+mat[:,1,1]+mat[:,2,2])/2)
  angle = (angle+2*np.pi).fmod(2*np.pi).unsqueeze(1).unsqueeze(2)
  norm = torch.sqrt((mat[:,2,1]-mat[:,1,2])**2+(mat[:,0,2]-mat[:,2,0])**2+(mat[:,1,0]-mat[:,0,1])**2)
  ux = ((mat[:,2,1]-mat[:,1,2])/norm).unsqueeze(1).unsqueeze(2)
  uy = ((mat[:,0,2]-mat[:,2,0])/norm).unsqueeze(1).unsqueeze(2)
  uz = ((mat[:,1,0]-mat[:,0,1])/norm).unsqueeze(1).unsqueeze(2)
  res = torch.cat((ux,uy,uz,angle),dim=2)
  return res

def getPosition(q):
    """
    Retourne la position de la pointe avec les angles actuels
    """
    b = q.shape[0]
    a = torch.tensor([0,0.5,0.5,0,0]).unsqueeze(0)
    a = torch.cat((a,)*b,dim=0).unsqueeze(1).cuda().detach()
    d = torch.tensor([0.2,0,0,0,0.2]).unsqueeze(0)
    d = torch.cat((d,)*b,dim=0).unsqueeze(1).cuda().detach()
    al = torch.tensor([-pi/2,0,0,pi/2,0]).unsqueeze(0)
    al = torch.cat((al,)*b,dim=0).unsqueeze(1).cuda().detach()


    l1 = (cos(q[:,:,0])*(-d[:,:,4]*sin(q[:,:,1]+q[:,:,2]+q[:,:,3])+a[:,:,2]*cos(q[:,:,1]+q[:,:,2])+a[:,:,1]*cos(q[:,:,1]))).unsqueeze(0)
    l2 = (sin(q[:,:,0])*(-d[:,:,4]*sin(q[:,:,1]+q[:,:,2]+q[:,:,3])+a[:,:,2]*cos(q[:,:,1]+q[:,:,2])+a[:,:,1]*cos(q[:,:,1]))).unsqueeze(0)
    l3 = (d[:,:,0]-a[:,:,1]*sin(q[:,:,1])-a[:,:,2]*cos(q[:,:,1]+q[:,:,2])-d[:,:,4]*cos(q[:,:,1]+q[:,:,2]+q[:,:,3])).unsqueeze(0)
    pos = torch.cat((l1,l2,l3),dim=2).squeeze().unsqueeze(1)
    """
    pos = torch.tensor([cos(q[:,:,0])*(-d[:,:,4]*sin(q[:,:,1]+q[:,:,2]+q[:,:,3])+a[:,:,2]*cos(q[:,:,1]+q[:,:,2])+a[:,:,1]*cos(q[:,:,1])),
                    sin(q[:,:,0])*(-d[:,:,4]*sin(q[:,:,1]+q[:,:,2]+q[:,:,3])+a[:,:,2]*cos(q[:,:,1]+q[:,:,2])+a[:,:,1]*cos(q[:,:,1])),
                    d[:,:,0]-a[:,:,1]*sin(q[:,:,1])-a[:,:,2]*cos(q[:,:,1]+q[:,:,2])-d[:,:,4]*cos(q[:,:,1]+q[:,:,2]+q[:,:,3])])
    """
    return pos

BATCH_SIZE = args.b
LR = args.lr
WEIGHT_DECAY = args.wd
MAX_EPOCH = 1000000
NET_SIZE = args.s

hyperparameter ='BATCH SIZE = %d\nLEARNING RATE = %F\nWEIGHT DECAY = %f\nMAX EPOCH= %d\nNet Size: %d'%(BATCH_SIZE,LR,WEIGHT_DECAY,MAX_EPOCH,NET_SIZE)

fig_perf = plt.figure(figsize=(8,10))
graph = fig_perf.add_subplot(211)
graph.set_title('Evolution of Error during Training %d\n%s'%(args.n,hyperparameter),pad=20,wrap=True)
#graph_angle = fig_perf.add_subplot(212)
graph.set_ylabel('MSE Loss [m]')
graph.set_xlabel('Epochs')
graph.grid('on')
graph.legend()
#graph_angle.set_xlabel('Epoch')
#graph_angle.set_ylabel('Angular MAE')
#graph_angle.grid('on')

fig_perf.tight_layout()

"""
Définition du réseau & Initialisation
"""
net = InvKinNet(NET_SIZE,nl = args.nl)
net = nn.DataParallel(net)
try:
    net.cuda()
except:
    pass

"""
dataset
"""
dtset_train = MyDataset('train')
dataloader_train = torch.utils.data.DataLoader(dtset_train, batch_size=BATCH_SIZE,shuffle=True, num_workers=0)

dtset_eval = MyDataset('eval')
dataloader_eval = torch.utils.data.DataLoader(dtset_eval, batch_size=BATCH_SIZE,shuffle=True, num_workers=0)
"""
Optimizer
"""
optimizer = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=WEIGHT_DECAY, amsgrad=False)
#optimizer = torch.optim.SGD(net.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
#optimizer = torch.optim.Adadelta(net.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=WEIGHT_DECAY)

print('beginning...')
print(hyperparameter)
epoch_loss_train = []
epoch_loss_eval = []
epoch_loss_train_angle = []
epoch_loss_eval_angle = []
noLegend= True
for epoch in range(MAX_EPOCH) :
    if epoch == 1:
        noLegend = False
    running_loss = 0
    iter = 0
    loss_file = open(path,'a+')
    loss_file.write('%d;'%(epoch))
    loss_file.close()
    print('---------------- EPOCH N° %d ----------------'%(epoch-1))
    print('////////////////  TRAIN  ////////////////')
    data = tqdm(dataloader_train)
    running_loss_angle = 0
    #data = dataloader_train
    for values in data:
        p = values
        net.train()
        iter += 1
        p = p.unsqueeze(1)
        try:
            p=p.cuda()
        except:
            pass
        optimizer.zero_grad()
        out = net(p)
        out_decoded = decode(out)
        pos = getPosition(out_decoded)
        rot = getRot(out_decoded)
        #state = torch.cat((pos,rot),dim=2)

        loss = 0.1*nn.MSELoss()(pos,p)#myMSE(out_decoded,q)
        loss.backward()
        optimizer.step()

        running_loss += (10*loss.tolist())**0.5
        #running_loss_angle += myMAE(out_decoded,q).tolist()*180/np.pi
        data.set_description('Loss:%.5f, rot: %.2f, pos:%.2f'%(running_loss/iter,torch.mean(rot).tolist(),torch.mean(pos).tolist()))
        #clear_output()
    loss_file = open(path,'a+')
    loss_file.write('%f;'%(running_loss/len(data)))
    loss_file.close()
    print('/////////////  EVAL  ////////////////')
    running_loss = 0
    iter = 0
    data = tqdm(dataloader_eval)
    running_loss_angle = 0
    #data=dataloader_eval
    for values in data:
        net.eval()
        p = values
        iter += 1
        p = p.unsqueeze(1)#;q = q.unsqueeze(1)
        try:
            p=p.cuda()#;q=q.cuda()
        except:
            pass
        out = net(p)
        out_decoded = decode(out)
        pos = getPosition(out_decoded)
        #state = torch.cat((pos,getRot(out_decoded)),dim=2)
        loss = nn.MSELoss()(pos,p)
        running_loss += loss.tolist()**0.5
        #running_loss_angle += myMAE(out_decoded,q).tolist()*180/np.pi
        data.set_description('Loss:%.5f, rot: %.2f, pos:%.2f'%(running_loss/iter,torch.mean(rot).tolist(),torch.mean(pos).tolist()))
    loss_file = open(path,'a+')
    loss_file.write('%f\n'%(running_loss/len(data)))
    loss_file.close()
    if epoch % 20 == 0:
        torch.save(net.state_dict(),'InvKin_%d__Epoch_%d'%(args.n,epoch))

    plot()
