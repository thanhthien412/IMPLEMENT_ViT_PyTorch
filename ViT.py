from lib import*
from Encoder import Encoder
from Positional_encoding import PositionalEncoding_1D
from Split_patches import inputembeeding
import numpy as np
class ViT(nn.Module):
    def __init__(self,d_input,patchs_size,patchs,classes,device,num_head=8,stack_block=3,hidden=2048,drop_out=0.1):
        super(ViT,self).__init__()
        #split patches
        self.split      =inputembeeding(patchs_size)
        # flatten
        self.flatten    =nn.Flatten(-3,-1)
        # linear
        self.embeeding  =nn.Linear(patchs_size[0]*patchs_size[1]*patchs_size[2],d_input)
        #learnable class 
        self.learnableclass =nn.Parameter(torch.zeros(1,d_input))
        #positional_encoder
        self.postional_encoding=PositionalEncoding_1D(d_input,patchs+1)().to(device)
        
        self.encoder    =Encoder(d_input,num_head,stack_block,hidden,drop_out)
        
        
        #output
        self.linear1    =nn.Linear(d_input*(patchs+1),hidden)
        self.dropout    =nn.Dropout(drop_out)
        self.relu       =nn.ReLU()
        self.linear2    =nn.Linear(hidden,classes)
        self.softmax    =nn.Softmax(dim=-1)
    
        
    def forward(self,x):
        x   =self.split(x)
        x   =self.flatten(x)
        x   =self.embeeding(x)
        x   =torch.concat([self.learnableclass.unsqueeze(0).repeat(x.size()[0],1,1),x],1)
        x   += self.postional_encoding
        x   =self.encoder(x)
        x   =nn.Flatten(-2,-1)(x)
        x   =self.linear1(x)
        x   =self.dropout(x)
        x   =self.relu(x)
        x   =self.linear2(x)
        x   =self.softmax(x)
        
        return x
    
# model=ViT(512,(16,16,3),196,20)
# A=torch.randn(30,224,224,3)
# out=model(A)
# print(out.size())
# print(out)
