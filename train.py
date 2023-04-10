from lib import*


from GPUtil import showUtilization as gpu_usage
from numba import cuda
import torch.optim as optim
torch.manual_seed(1234)
import time 
import argparse
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop
import torchvision
from ViT import ViT

def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()                             

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ap = argparse.ArgumentParser()
ap.add_argument('-patch_size','--patch_size',type=int,default=4)
ap.add_argument('-pic_size','--pic_size',type=int,default=32)
ap.add_argument('-d_model','--d_model',type=int, default=512)
ap.add_argument('-patch','--no_patches',type=int,default=64)
ap.add_argument('-head','--num_head',type=int,default=8)
ap.add_argument('-en_block','--encoder_block',type=int,default=3)
ap.add_argument('-hidden','--hidden_layer',type=int,default=2048)
ap.add_argument('-drop','--dropout_prob',type=float,default=0.1)
ap.add_argument('-epoch','--epoch',type=float,default=30)
ap.add_argument('-lr','--learning_rate',type=float,default=10e-3)
ap.add_argument('-decay','--decay',type=float,default=0.003)
args = vars(ap.parse_args())

free_gpu_cache()
print('device: ', device)



transform_training_data = Compose(
    [RandomCrop(32, padding=4),Resize(args['pic_size']), RandomHorizontalFlip(), ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

train_data = torchvision.datasets.CIFAR10(root="data",train=True, download=False, transform=transform_training_data)

trainloader = DataLoader(train_data, batch_size=4,
                                          shuffle=True)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


model = ViT(args['d_model'],(args['patch_size'],args['patch_size'],3),args['no_patches'],10,device,args['num_head'],args['encoder_block'],args['hidden_layer'])


optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['decay'])
criterion = nn.CrossEntropyLoss()

def train_model(model,criterion,optimizer,num_epochs,dataloader):
    model.to(device)
    
    for epoch in range(num_epochs):
        iteration=0
        epoch_train_loss=0.0
        iter_loss   =0.0
        t_epoch_start=time.time()
        t_iter_start=time.time()
        print('---'*20)
        print('Epoch: {}/{}'.format(epoch+1,num_epochs))
        print('---'*20)
        
        for inputs,labels in dataloader:
            inputs=inputs.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()
            
            outputs=model(inputs)
            
            loss=criterion(outputs,labels)
            epoch_train_loss+=loss.item()
            iter_loss+=loss.item()
            loss.backward()
            optimizer.step()
            iteration+=1
            
            if(iteration%20==0):
                print("Batch{} : LOSS:{:.4f}  DURATION:{:.4f} sec".format(iteration,iter_loss/20,time.time()-t_iter_start))
                t_iter_start=time.time()
                iter_loss=0.0
                
        print("###"*20)
        print("Epoch{}/{} LOSS:".format(epoch+1,num_epochs))
        print('Duration : {:.4f} sec'.format(time.time()-t_epoch_start))
        print("###"*20)
        if((epoch+1)%10==0):
            torch.save(model.save_dict(),'./weight/epoch_{}.pth'.format(epoch+1))
            
            

train_model(model,criterion,optimizer,args['epoch'],trainloader).to(device)
