from lib import*
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self,images,labels) -> None:
        super().__init__()
        self.images =images
        self.labels =labels
        
        # instead of list or array image we can pass h5py to avoid running out of memory

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index],self.labels[index]
    
    
def my_collate_fn(batch):
    batch_patches=[]
    batch_label=[]
    for sample in batch:
        batch_patches.append(sample[0])
        batch_label.append(sample[1])
        
    batch_patches=torch.stack(batch_patches,0)
    batch_label=torch.stack(batch_label,0)
    
    return batch_patches,batch_label