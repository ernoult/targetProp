import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import pickle
import datetime
import numpy as np
import torch.optim as optim
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='Target Prop')
parser.add_argument(
    '--batch-size',
    type=int,
    default=128,
    help='input batch size for training (default: 20)')
parser.add_argument(
    '--test-batch-size',
    type=int,
    default=1000,
    metavar='N',
    help='input batch size for testing (default: 1000)')   
parser.add_argument(
    '--epochs',
    type=int,
    default=1,
help='number of epochs to train (default: 1)')    
parser.add_argument(
    '--lr_tab',
    nargs = '+',
    type=float,
    default=[0.05, 0.01],
    help='learning rate (default: [0.05, 0.1])')
parser.add_argument(
    '--size_tab',
    nargs = '+',
    type=int,
    default=[784, 512, 10],
    help='tab of layer sizes (default: [10])')
parser.add_argument(
    '--action',
    type=str,
    default='test',
    help='action to execute (default: test)')
parser.add_argument(
    '--sigma',
    type=float,
    default=0.01,
    help='standard deviation of the noise used to train feedback weights (default: 0.01)')
parser.add_argument(
    '--lr_target',
    type=float,
    default=0.01,
    help='learning rate used to compute the target of the last layer (default: 0.01)')
parser.add_argument(
    '--device',
    type=int,
    default=0,
    help='selects cuda device (default 0, -1 to select )')
parser.add_argument(
    '--SDTP', 
    default = False, 
    action = 'store_true', 
    help='simplified target prop (default: False)')

args = parser.parse_args()

batch_size = args.batch_size
batch_size_test = args.test_batch_size

class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)
        
'''        
class ReshapeTransformTarget:
    def __init__(self, number_classes):
        self.number_classes = number_classes
    
    def __call__(self, target):
        target=torch.tensor(target).unsqueeze(0).unsqueeze(1)
        target_onehot=torch.zeros((1,self.number_classes))    
        return target_onehot.scatter_(1, target, 1).squeeze(0)
'''
        

mnist_transforms=[torchvision.transforms.ToTensor(),ReshapeTransform((-1,))]

train_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST(root='./data', train=True, download=True,
                     transform=torchvision.transforms.Compose(mnist_transforms)),
                     #target_transform=ReshapeTransformTarget(10)),
batch_size = args.batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST(root='./data', train=False, download=True,
                     transform=torchvision.transforms.Compose(mnist_transforms)),
                     #target_transform=ReshapeTransformTarget(10)),
batch_size = args.test_batch_size, shuffle=True)


class Net(nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()

        self.size_tab = args.size_tab

        wf = nn.ModuleList([])
        wb = nn.ModuleList([])

        for i in range(len(self.size_tab) - 1):
            wf.append(nn.Linear(self.size_tab[i], self.size_tab[i + 1]))

        for i in range(len(self.size_tab) - 2):
            wb.append(nn.Linear(self.size_tab[-1 - i], self.size_tab[-2 - i]))

        self.logsoft = nn.LogSoftmax(dim=1)
        self.wf = wf
        self.wb = wb
        self.lr_target = args.lr_target
        self.SDTP = args.SDTP
        self.sigma = args.sigma

    def forward(self, x):
        s = [x]

        for i in range(len(self.size_tab) - 2):
            s.append(torch.tanh(self.wf[i](s[i])))

        s.append(torch.exp(self.logsoft(self.wf[-1](s[-1]))))

        return s

    def computeTargets(self, s, y, criterion):
        t = []

        #s.reverse()
        
        if self.SDTP:
            #compute first target for simplified DTP
            t.append(F.one_hot(y, num_classes=10).float())
        else:
            #compute first target for standard DTP
            loss = criterion(s[0].float(), y)
            init_grad = torch.tensor([1 for i in range(data.size(0))], dtype=torch.float, device=device, requires_grad=True)
            grad = torch.autograd.grad(loss, s[0], grad_outputs=init_grad)
            t.append(s[0] - self.lr_target*grad[0])

        #compute targets for lower layers       
        for i in range(len(s) - 2):
            t.append(s[i + 1] - torch.tanh(self.wb[i](s[i])) + torch.tanh(self.wb[i](t[i])))

        #s.reverse()
        #t.reverse()

        return t

    
    def reconstruct(self, s, i):
        r = torch.tanh(self.wf[- i](s))
        r = torch.tanh(self.wb[i - 1](r))
        return r
        
                     

if __name__ == '__main__':

    if args.device >= 0:
        device = torch.device("cuda:"+str(args.device)+")")

    nlll = nn.NLLLoss(reduction='none')
    mse = torch.nn.MSELoss(reduction='sum')
    net = Net(args)
    net.to(device)


    if args.action == 'train':

        net.train()

        #build optimizers for forward weights
        optim_wf_param = []
        for i in range(len(net.wf) - 1):
            optim_wf_param.append({'params':net.wf[i].parameters(), 'lr': args.lr_tab[0]})
        optimizer_wf = torch.optim.SGD(optim_wf_param)

        #build optimizers for the last forward weight
        optimizer_wf_top = torch.optim.SGD([{'params':net.wf[-1].parameters(), 'lr': args.lr_tab[0]}])

        #build optimizers for backward weights
        optim_wb_param = []
        for i in range(len(net.wb)):
            optim_wb_param.append({'params':net.wb[i].parameters(), 'lr': args.lr_tab[0]})
        optimizer_wb = torch.optim.SGD(optim_wb_param)          

        #start training
        for epoch in range(1, args.epochs + 1):
            correct = 0
            print('Epoch {}'.format(epoch))
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(device), targets.to(device)

                #forward pass
                s = net(data)

                #compute targets
                s.reverse()
                t = net.computeTargets(s, targets, nlll)

                #train backward weights
                for i in range(len(s) - 2):
                    #generate corrupted data
                    s_corrupt = s[i + 1] + net.sigma*torch.randn_like(s[i + 1])

                    #reconstruct the data
                    r = net.reconstruct(s_corrupt, i + 1)

                    #update the backward weights
                    loss_wb = (1/(2*data.size(0)))*mse(s[i + 1], r)
                    optimizer_wb.zero_grad()
                    loss_wb.backward(retain_graph = True)
                    optimizer_wb.step()
                
                #train forward weights
                s.reverse()
                t.reverse()

                for i in range(len(s) - 2):
                    loss_wf = (1/(2*data.size(0)))*mse(s[i + 1], t[i])
                    optimizer_wf.zero_grad()
                    loss_wf.backward(retain_graph = True)
                    optimizer_wf.step()
                                    
                #train the top forward weights
                loss_wf_top = nlll(s[-1].float(), targets).mean()  
                optimizer_wf_top.zero_grad()
                loss_wf_top.backward(retain_graph = True)
                optimizer_wf_top.step()

                #compute prediction error
                pred = s[-1].data.max(1, keepdim=True)[1]
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum()
                

            print('\nAverage Training Error Rate: {:.2f}% ({}/{})\n'.format(
                    100*(len(train_loader.dataset)- correct.item() )/ len(train_loader.dataset), 
                    len(train_loader.dataset)-correct.item(), len(train_loader.dataset)))


    if args.action == 'test':


        #print(net)
        _, (data, target) = next(enumerate(train_loader))
        data, target = data.to(device), target.to(device)

        s = net(data)

        #check layer size
        '''
        for i in s:
            print(i.size())
        '''          

        #check gradient computation at the last layer
        '''
        loss = criterion(s[-1].float(), target)
        init_grad = torch.tensor([1 for i in range(data.size(0))], dtype=torch.float, device=device, requires_grad=True)
        grad = torch.autograd.grad(loss, s[-1], grad_outputs=init_grad)
        print(grad[0].size())
        '''
        
        #check target computation
        '''
        s.reverse()
        t = net.computeTargets(s, target, criterion)                
        #print(t)
        for i in t:
            print(i.size())
        '''
        
        #check reconstruction
        '''
        s.reverse()
        for i in range(len(s) - 2):
            r = net.reconstruct(s[i + 1], i + 1)
            print(r.size())
        '''



















