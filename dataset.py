import torch
import os
import imghdr
import numpy as np

from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage, RandomCrop

def mask_to_bbox(mask):
    rows =  np.where(np.max(mask, axis=1) == 1)[0]
    cols = np.where(np.max(mask, axis=0) == 1)[0]

    top = rows[0]
    bottom = rows[-1]

    left = cols[0]
    right = cols[-1]

    #print((top, left, bottom, right))

    return left, top, right, bottom

def squarify_bbox(image, bbox):
    W, H  = image.size
    left, top, right, bottom = bbox
    x_center = (right + left) / 2
    y_center = (bottom + top) / 2

    width = right - left
    height = bottom - top

    if width > height:
        top = max(0, y_center - width / 2)
        bottom = min(H, y_center + width / 2)
    else:
        left = max(0, x_center - height / 2)
        right = min(W, x_center + height / 2)

    return left, top, right, bottom

class KOTrainDataset():

    def __init__(self, configs):

        self.data_dir = configs.data_dir
        self.output_dir = configs.output_dir

        self.image_height = configs.image_height
        self.image_width = configs.image_width
        self.randomized_background = configs.randomized_background
        self.visualize_data = configs.visualize_data
        self.load_into_memory = configs.load_into_memory
        self.apply_cropping = configs.apply_cropping
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

                        img_tensor, mask_tensor = self._load_data_(file_path, mask_path)

                        self.data.append(img_tensor)
                        self.masks.append(mask_tensor)

        # parse backgrounds
        background_dir = os.path.join(self.data_dir, 'negative')
        if self.randomized_background:
            for file in os.listdir(background_dir):
                bg_path = os.path.join(background_dir, file)
                self.background_paths.append(bg_path)

                if self.load_into_memory:
                    bg_img = Image.open(bg_path)
                    bg_img = bg_img.resize((self.image_width, self.image_height))
                    bg_tensor = self.totensor(bg_img)
                    self.backgrounds.append(bg_tensor)

    def _load_data_(self, img_path, mask_path):
        img = Image.open(img_path)
        mask_img = Image.open(mask_path)

        if self.apply_cropping:

            left, top, right, bottom = mask_to_bbox(np.array(mask_img))
            left, top, right, bottom = squarify_bbox(img, (left, top, right, bottom))
            img = img.crop((left, top, right, bottom))
            mask_img = mask_img.crop((left, top, right, bottom))

            img = img.resize((self.image_width, self.image_height))
            mask_img = mask_img.resize((self.image_width, self.image_height))

        img_tensor = self.totensor(img)
        mask_tensor = self.totensor(mask_img)

        return img_tensor, mask_tensor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        if not self.load_into_memory:

            img_tensor, mask_tensor = self._load_data(self.data_paths[idx], self.mask_paths[idx])
        else:
            mask_tensor = self.masks[idx]
            img_tensor = self.data[idx]

        if self.randomized_background:

            if not self.load_into_memory:

                bg_path = self.background_paths[np.random.randint(0, len(self.backgrounds))]
                bg_img = Image.open(bg_path)
                bg_img = bg_img.resize((self.image_width, self.image_height))
                bg_tensor = self.totensor(bg_img)

            else:
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
        self.output_dir = configs.output_dir
        self.apply_cropping = configs.apply_cropping
        self.visualize_data = configs.visualize_data

        self.image_height = configs.image_height
        self.image_width = configs.image_width

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
        if self.apply_cropping:
            #print(self.bbox[idx])
            left, top, right, bottom = self.bbox[idx]
            left, top, right, bottom = squarify_bbox(img, (left, top, right, bottom))

            img = img.crop((left, top, right, bottom))
        img = img.resize((self.image_height, self.image_width))

        if self.visualize_data:
            img.save(os.path.join(self.output_dir, 'example_val_image_{}.jpg'.format(idx)))
        img_tensor = self.totensor(img)
        return img_tensor, self.labels[idx]
