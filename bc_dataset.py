import torch
import numpy as np
from dataset import KOTrainDataset

class BCTrainDataset(KOTrainDataset):

    def __init__(self, configs):
        super(BCTrainDataset, self).__init__(configs)
        self.num_classes = configs.num_classes
        self.bc_mixing_method = configs.bc_mixing_method

    def __getitem__(self, idx):

        # generate random other test sample to compare
        # currently this could be of the same class
        second_idx = np.random.randint(0, len(self.labels))
        mixture_ratio = np.random.random()

        # create the label tensor with mixture ratios as labels
        label_tensor = torch.zeros(self.num_classes, dtype=torch.float)
        label_tensor[self.labels[idx]] += mixture_ratio
        label_tensor[self.labels[second_idx]] += (1 - mixture_ratio)


        if not self.load_into_memory:
            img_tensor, mask_tensor = self._load_data(self.data_paths[idx], self.mask_paths[idx])
            img_tensor2, mask_tensor2 = self._load_data(self.data_paths[second_idx], self.mask_paths[second_idx])
        else:
            mask_tensor = self.masks[idx]
            img_tensor = self.data[idx]
            mask_tensor2 = self.masks[second_idx]
            img_tensor2 = self.masks[second_idx]

        if self.randomized_background:

            # Combine segmentation masks by taking element-wise max
            mask_tensor = torch.max(mask_tensor, mask_tensor2)

            if not self.load_into_memory:

                bg_path = self.background_paths[np.random.randint(0, len(self.backgrounds))]
                bg_img = Image.open(bg_path)
                bg_img = bg_img.resize((self.image_width, self.image_height))
                bg_tensor = self.totensor(bg_img)

            else:
                bg_tensor = self.backgrounds[np.random.randint(0, len(self.backgrounds))]

            # apply mixing
            if self.bc_mixing_method == 'linear':
                mix1 = mixture_ratio * img_tensor * mask_tensor
                mix2 = (1 - mixture_ratio) * img_tensor2 * mask_tensor
                masked_object = mix1 + mix2
                
            elif self.bc_mixing_method == 'prop':
                img_tensor = img_tensor * mask_tensor 
                img_tensor2 = img_tensor2 * mask_tensor 
                
                p = 1 / ( 1 + (img_tensor.std() / img_tensor2.std()) * (( 1 - mixture_ratio) / mixture_ratio))
                mix1 = p * (img_tensor - img_tensor.mean()) 
                mix2 = (1 - p) * (img_tensor2 - img_tensor2.mean())
                
                denom = torch.sqrt(p ** 2 + (1 - p) ** 2)
                
                masked_object = (mix1 + mix2) / denom

            inverse_mask = torch.ones(mask_tensor.shape, dtype=torch.float) - mask_tensor
            masked_background = bg_tensor * inverse_mask

            img_tensor = masked_object + masked_background

        if self.visualize_data:
            img = self.toPIL(img_tensor)
            img.save(os.path.join(self.output_dir, 'example_train_image.jpg'))

        return img_tensor, label_tensor
