import torch
import os
import imghdr
import numpy as np

from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

class KOTrainDataset():

    def __init__(self, configs):

        self.data_dir = configs.data_dir
        self.output_dir = configs.output_dir
        self.randomized_background = configs.randomized_background
        self.visualize_data = configs.visualize_data
        self.load_into_memory = configs.load_into_memory
        self.device = configs.device
           
        self.data_paths = [] # store paths to images
        self.data = [] # store image tensors
        self.mask_paths = [] # store paths to masks
        self.masks = [] # store mask tensors
        self.background_paths = [] # store paths to backgrounds
        self.backgrounds = [] # store background tensor
        
        self.labels = [] # store labels


        self.totensor = ToTensor() # for converting PIL Image to pytorch Tensors
        self.toPIL = ToPILImage()

        train_dir = os.path.join(self.data_dir,'multiple', 'train')

        labels = os.listdir(train_dir)

        # mapping from string label to class int
        # perhaps this should be explicitly defined idk
        self.label_to_class = {label: idx for idx, label in enumerate(labels)}

        print(self.label_to_class)

        assert len(self.label_to_class) == configs.num_classes

        for dir in os.listdir(train_dir):
            object_dir = os.path.join(train_dir, dir)

            assert os.path.isdir(object_dir)

            for file in os.listdir(object_dir):
                file_path = os.path.join(object_dir, file)
                if imghdr.what(file_path) == 'jpeg':
                    
                    self.labels.append(self.label_to_class[dir])
                    
                    self.data_paths.append(file_path)
                    
                    mask_path = os.path.join(object_dir, file.split('.')[0] + '_mask.png')
                    self.mask_paths.append(mask_path)
                    
                    if self.load_into_memory:
                    
                        img = Image.open(file_path)
                        img_tensor = self.totensor(img)
                                            
                        self.data.append(img_tensor)  
                        
                        mask_img = Image.open(mask_path)
                        mask_tensor = self.totensor(mask_img)
                        
                        self.masks.append(mask_tensor)
        
        # parse backgrounds
        background_dir = os.path.join(self.data_dir, 'negative')
        if self.randomized_background:
            for file in os.listdir(background_dir):
                bg_path = os.path.join(background_dir, file)
                self.background_paths.append(bg_path)
                
                if self.load_into_memory:
                    bg_img = Image.open(bg_path)
                    bg_tensor = self.totensor(bg_img)
                    self.backgrounds.append(bg_tensor)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        if not self.load_into_memory:
            
            img = Image.open(self.data_paths[idx])
            img_tensor = self.totensor(img)
        else:
            img_tensor = self.data[idx]

        if self.randomized_background:
            
            if not self.load_into_memory:
                
                mask_path = self.masks_paths[idx]
                mask_img = Image.open(mask_path)
                mask_tensor = self.totensor(mask_img)

                bg_path = self.background_paths[np.random.randint(0, len(self.backgrounds))]
                bg_img = Image.open(bg_path)
                bg_tensor = self.totensor(bg_img)
                
            else:
                mask_tensor = self.masks[idx]
                bg_tensor = self.backgrounds[np.random.randint(0, len(self.backgrounds))]
            
            masked_object = img_tensor * mask_tensor
            inverse_mask = torch.ones(mask_tensor.shape, dtype=torch.float) - mask_tensor
            masked_background = bg_tensor * inverse_mask

            img_tensor = masked_object + masked_background

            if self.visualize_data:
                img = self.toPIL(img_tensor)
                img.save(os.path.join(self.output_dir, 'example_train_image.jpg'))

        return img_tensor, self.labels[idx]

class KOTestDataset():

    def __init__(self, configs):

        self.data_dir = os.path.join(configs.data_dir)
        test_dir = os.path.join(self.data_dir,'multiple', 'test')

        self.data = [] # store paths to images
        self.bbox = [] # store bboxes // currently unused
        self.labels = [] # store labels

        self.totensor = ToTensor() # for converting PIL Image to pytorch Tensors

        labels = os.listdir(test_dir)

        # mapping from string label to class int
        # perhaps this should be explicitly defined idk
        self.label_to_class = {label: idx for idx, label in enumerate(labels)}

        print(self.label_to_class)

        assert len(self.label_to_class) == configs.num_classes

        for dir in os.listdir(test_dir):
            object_dir = os.path.join(test_dir, dir)

            assert os.path.isdir(object_dir)

            for file in os.listdir(object_dir):
                file_path = os.path.join(object_dir, file)
                if imghdr.what(file_path) == 'jpeg':

                    self.data.append(file_path)

                    bbox_path = os.path.join(object_dir, file.split('.')[0] + '.bbox')

                    with open(bbox_path, 'r') as stream:

                        bbox_str = stream.readlines()[0]

                    x, y, w, h, _ = bbox_str.split(' ')

                    bbox = [float(x), float(y), float(x) + float(w), float(y) + float(h)]

                    self.bbox.append(bbox)

                    self.labels.append(self.label_to_class[dir])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx])
        img_tensor = self.totensor(img)
        return img_tensor, self.labels[idx]
