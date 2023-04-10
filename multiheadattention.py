from lib import*

def scaled_dot_product(q,k,v):
    d_k         =q.size()[-1]
    attention   =torch.matmul(q,k.transpose(-1,-2))/math.sqrt(d_k)
    
    attention   =torch.softmax(attention,-1)
    value       =torch.matmul(attention,v)
    
    return value


class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_head):
        super(MultiHeadAttention,self).__init__()
        self.head_dim       =d_model//num_head
        self.num_head       =num_head
        self.qkv_layer      =nn.Linear(d_model,d_model*3)
        self.output         =nn.Linear(d_model,d_model)
    
    
    def forward(self,x):
        batch_size,patchs,_=x.size()
        qkv=self.qkv_layer(x) #(batch,patchs,dim*3)
        qkv=qkv.reshape(batch_size,patchs,self.num_head,self.head_dim*3)
        qkv=qkv.permute(0,2,1,3) #(batch,num_head,patchs,self.head_dim*3)
        q,k,v=qkv.chunk(3,dim=-1)
        value=scaled_dot_product(q,k,v)#(bath,num_head,patchs,self.head_dim)
        value=value.reshape(batch_size,patchs,self.num_head*self.head_dim)
        output=self.output(value)
        return output
    