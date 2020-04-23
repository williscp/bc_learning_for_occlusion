import torch 
import numpy as np

def apply_mixture(mixing_method, img_tensor1, img_tensor2, label1, label2, num_classes=8):
     
    label_tensor = torch.zeros(num_classes, dtype=torch.float)
    
    if mixing_method == 'linear':
        
        mixture_ratio = np.random.random()

        label_tensor[label1] += mixture_ratio
        label_tensor[label2] += (1 - mixture_ratio)
        
        mix1 = mixture_ratio * (img_tensor1 - img_tensor1.mean())
        mix2 = (1 - mixture_ratio) * (img_tensor2 - img_tensor2.mean())
        mixed_tensor = mix1 + mix2

    elif mixing_method == 'prop':
       
        mixture_ratio = np.random.random()

        label_tensor[label1] += mixture_ratio
        label_tensor[label2] += (1 - mixture_ratio)
        
        p = 1 / ( 1 + (img_tensor1.std() / img_tensor2.std()) * (( 1 - mixture_ratio) / mixture_ratio))
        mix1 = p * (img_tensor1 - img_tensor1.mean()) 
        mix2 = (1 - p) * (img_tensor2 - img_tensor2.mean())

        denom = torch.sqrt(p ** 2 + (1 - p) ** 2)

        mixed_tensor = (mix1 + mix2) / denom

    elif mixing_method == 'VH':
        
        h_ratio = np.random.random() # height ratio
        w_ratio = np.random.random() # width ratio
        mixture_ratio = np.random.random()
        
        label_tensor[label1] += h_ratio * (w_ratio + (1 - w_ratio) * mixture_ratio) + (1 - h_ratio) * w_ratio * (1 - mixture_ratio)
        label_tensor[label2] += (1 - h_ratio) * ((1 - w_ratio) + w_ratio * mixture_ratio) + h_ratio * (1 - w_ratio) * (1 - mixture_ratio)
                
        p = 1 / (1 + (img_tensor1.std() / img_tensor2.std()) * (( 1 - mixture_ratio) / mixture_ratio))
        denom = torch.sqrt(p ** 2 + (1 - p) ** 2)
        
        mix1 = (p * (img_tensor1 - img_tensor1.mean()) + (1 - p) * (img_tensor2 - img_tensor2.mean())) / denom
        mix1 = img_tensor1
        mix2 = ((1 - p) * (img_tensor1 - img_tensor1.mean()) + p * (img_tensor2 - img_tensor2.mean())) / denom
        
        _, H, W = img_tensor1.shape
        
        h_boundary = int(max(h_ratio * H, 1)) 
        w_boundary = int(max(w_ratio * W, 1))
        
        top_half = torch.cat((img_tensor1[:,0:h_boundary,0:w_boundary], mix1[:,0:h_boundary,w_boundary-1:-1]), dim=2)
        bottom_half = torch.cat((mix2[:,h_boundary-1:-1,0:w_boundary], img_tensor2[:,h_boundary-1:-1,w_boundary-1:-1]), dim=2)
        
        mixed_tensor = torch.cat((top_half, bottom_half), dim=1)
        
        """
        print("Shapes:")
        print(img_tensor1.shape)
        print(img_tensor2.shape)
        print(mixed_tensor.shape)
        """
    return mixed_tensor, label_tensor
        
                                               