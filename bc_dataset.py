import torch
import numpy as np
import os
from dataset import KOTrainDataset
from mixture.mixture import apply_mixture

class BCTrainDataset(KOTrainDataset):

    def __init__(self, configs):
        super(BCTrainDataset, self).__init__(configs)
        self.num_classes = configs.num_classes
        self.bc_mixing_method = configs.bc_mixing_method

    def __getitem__(self, idx):

        # generate random other test sample to compare
        # currently this could be of the same class
        second_idx = np.random.randint(0, len(self.labels))
        
        label1 = self.labels[idx]
        label2 = self.labels[second_idx]

        if not self.load_into_memory:
            img_tensor1, mask_tensor1 = self._load_data(self.data_paths[idx], self.mask_paths[idx])
            img_tensor2, mask_tensor2 = self._load_data(self.data_paths[second_idx], self.mask_paths[second_idx])
        else:
            mask_tensor1 = self.masks[idx]
            img_tensor1 = self.data[idx]
            mask_tensor2 = self.masks[second_idx]
            img_tensor2 = self.data[second_idx]
            
        if self.randomized_background:
            # Combine segmentation masks by taking element-wise max
            mask_tensor = torch.max(mask_tensor1, mask_tensor2)
            img_tensor1 = img_tensor1 * mask_tensor 
            img_tensor2 = img_tensor2 * mask_tensor
        
        #print(img_tensor1.shape)
        #print(img_tensor2.shape)
        
        img_tensor, label_tensor = apply_mixture(
            self.bc_mixing_method,
            img_tensor1,
            img_tensor2,
            label1,
            label2,
            self.num_classes
        )

        if self.randomized_background:

            if not self.load_into_memory:

                bg_path = self.background_paths[np.random.randint(0, len(self.backgrounds))]
                bg_img = Image.open(bg_path)
                bg_img = bg_img.resize((self.image_width, self.image_height))
                bg_tensor = self.totensor(bg_img)

            else:
                bg_tensor = self.backgrounds[np.random.randint(0, len(self.backgrounds))]
                
            bg_tensor -= bg_tensor.mean()

            inverse_mask = torch.ones(mask_tensor.shape, dtype=torch.float) - mask_tensor
            masked_background = bg_tensor * inverse_mask

            img_tensor = img_tensor + masked_background

        if self.visualize_data:
            img_vis = img_tensor - torch.min(img_tensor)
            img_vis = img_vis / torch.max(img_vis)
            img = self.toPIL(img_vis)
            img.save(os.path.join(self.output_dir, 'example_train_image_{}.jpg'.format(idx)))

        return img_tensor, label_tensor
