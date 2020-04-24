import torch
import numpy as np

def apply_mixture(mixing_method, img_tensor1, img_tensor2, label1, label2, num_classes=8):

    label_tensor = torch.zeros(num_classes, dtype=torch.float)

    if mixing_method == 'double':

        mixture_ratio = np.random.random()

        label_tensor[label1] += mixture_ratio
        label_tensor[label2] += (1 - mixture_ratio)

        mix1 = mixture_ratio * 2 * (img_tensor1 - img_tensor1.mean())
        mix2 = (1 - mixture_ratio) * 2 * (img_tensor2 - img_tensor2.mean())
        mixed_tensor = mix1 + mix2

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
    elif mixing_method == 'Gauss':

        _, H, W = img_tensor1.shape

        mu_x = np.random.random() * 0.5 + 0.25 # between 0.25 and 0.75
        mu_y = np.random.random() * 0.5 + 0.25

        sigma = np.random.random() * 0.1 + 0.1 # between 0.1 and 0.2

        c = 1

        x_linspace = torch.linspace(0,1,W).repeat(3, 1)
        y_linspace = torch.linspace(0,1,H).repeat(3, 1)

        x_gauss = c * torch.exp( - (x_linspace - mu_x) ** 2 / sigma)
        y_gauss = c * torch.exp( - (y_linspace - mu_y) ** 2 / sigma)

        x_gauss = x_gauss.reshape(3, -1, W)
        y_gauss = y_gauss.reshape(3, H, -1)

        mask = x_gauss * y_gauss

        inverse_mask = torch.ones(3, H, W) - mask

        label_tensor[label1] = torch.sum(inverse_mask) / (3 * H * W)
        label_tensor[label2] = torch.sum(mask) / (3 * H * W)

        img_tensor1 = inverse_mask * img_tensor1
        img_tensor2 = mask * img_tensor2

        mixed_tensor = img_tensor1 + img_tensor2

    return mixed_tensor, label_tensor
