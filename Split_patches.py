from lib import*
import einops

def split_image_patches(image,no_patchs):
    image=torch.from_numpy(image)
    time=int(math.sqrt(no_patchs))
    size_patch=int(image.size()[0]/time)
    patches=[]
    print(size_patch)
    for i in range(no_patchs):
        patches.append(image[(i/time)*size_patch:((i/time)+1)*size_patch,
                             (i%time)*size_patch:((i%time)+1)*size_patch])
        
    return torch.stack(patches,0)


# another way


class inputembeeding(nn.Module):
    def __init__(self,patch_size):
        super().__init__()
        self.patch_size=patch_size
        
    def forward(self,x):
        patches = einops.rearrange(
            x, 'b c (h h1) (w w1)-> b (h w) h1 w1 c', h1=self.patch_size[0], w1=self.patch_size[1])
        
        return patches