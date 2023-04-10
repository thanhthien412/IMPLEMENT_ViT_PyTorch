from lib import*
from layernormalization import Layernormalization
from MLP_block import MLP_block
from multiheadattention import MultiHeadAttention

class Encoder_block(nn.Module):
    def __init__(self,d_model,num_head,hidden,drop_out):
        super(Encoder_block,self).__init__()
        self.norm1              =Layernormalization(d_model)
        self.norm2              =Layernormalization(d_model)
        self.multiheadattention =MultiHeadAttention(d_model,num_head)
        self.mlpblock           =MLP_block(d_model,hidden,drop_out)
    
    def forward(self,x):
        residual1   =x
        attention   =self.norm1(x)
        attention   =self.multiheadattention(attention)
        attention   += residual1
        residual2   =attention
        attention   =self.norm2(attention)
        attention   =self.mlpblock(attention)
        
        attention   += residual2
        
        return attention
    
class Encoder(nn.Module):
    def __init__(self,d_input,num_head,stack_block,hidden,drop_out):
        super(Encoder,self).__init__()
        self.model= nn.Sequential(*[Encoder_block(d_input,num_head,hidden,drop_out) for _ in range(stack_block)])
        
    def forward(self,x):
        x =self.model(x)
        
        return x
    
# d_model=512
# num=8
# model=Encoder(d_model,num,3,2048,0.1)
# #model=Layernormalization(512)
# x=torch.rand((30,5,d_model))
# out=model(x)
# print(out.size())
# print(out)
        