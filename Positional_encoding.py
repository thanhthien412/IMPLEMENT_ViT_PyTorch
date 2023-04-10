from lib import*


class PositionalEncoding_1D(nn.Module):
    def __init__(self,d_model,patchs):
        super(PositionalEncoding_1D,self).__init__()
        self.patchs    =patchs
        self.d_model   =d_model
        
    def forward(self):
        even_i_model    =torch.arange(0,self.d_model,2)
        denominator     =torch.pow(10000,even_i_model/self.d_model)
        position        =torch.arange(self.patchs).reshape(-1,1)
        PE_even         =torch.sin(position/denominator)
        PE_odd          =torch.cos(position/denominator)
        PE_total        =torch.stack([PE_even,PE_odd],dim=2).reshape(self.patchs,-1)
        
        return PE_total
    
    
    
    
class PositionalEncoding_2D(nn.Module):
    def __init__(self,d_model,height,width): #actually height*width = number of image patchs
        super(PositionalEncoding_2D,self).__init__()
        if(d_model %4 !=0):
            raise ValueError('The value of d_model must be even number and larger than 4')
        
        self.d_model=d_model
        self.height=height
        self.width=width
        
    def forward(self):
        d_model= self.d_model /2
        
        even_i_half_model               =torch.arange(0,d_model,2)
        even_i_half_remain_model        =torch.arange(d_model,self.d_model,2) 
         
        denominator_half_model          =torch.pow(1000,even_i_half_model/d_model)
        denominator_half_remain_model   =torch.pow(1000,even_i_half_remain_model/d_model)
        
        position_w                      =torch.arange(0,self.width).reshape(-1,1)
        position_h                      =torch.arange(0,self.height).reshape(-1,1)
        
        PE_even_half_model              =torch.sin(position_w/denominator_half_model)   
        PE_odd_half_model               =torch.cos(position_w/denominator_half_model)
        
        PE_even_half_remain_model       =torch.sin(position_h/denominator_half_remain_model)
        PE_odd_half_remain_model        =torch.cos(position_h/denominator_half_remain_model)
        
        PE_even_half_model              =PE_even_half_model.unsqueeze(0).repeat(self.height,1,1)
        PE_odd_half_model               =PE_odd_half_model.unsqueeze(0).repeat(self.height,1,1)
        
        PE_even_half_remain_model       =PE_even_half_remain_model.unsqueeze(1).repeat(1,self.width,1)
        PE_odd_half_remain_model        =PE_odd_half_remain_model.unsqueeze(1).repeat(1,self.width,1)
        
        # we need to reshape because after embeeding we have shape (batch,patch,d_model)
        PE_total                        =torch.stack([PE_even_half_model,PE_odd_half_model,PE_even_half_remain_model,PE_odd_half_remain_model],3).reshape(-1,self.d_model)
        
        return PE_total
    

# D_2    =PositionalEncoding_2D(512,3,3)
# D_1    =PositionalEncoding_1D(512,9)

# pos1    =D_1()

# pos2    =D_2()

# print(pos1.size())
# print(pos2.size())        