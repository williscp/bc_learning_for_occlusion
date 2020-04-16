import torch

class LinearModel(torch.nn.Module):

    def __init__(self, configs):
        super(SpectrogramModel, self).__init__()

        self.linear1 = torch.nn.Linear(3 * configs.image_height * configs.image_width, 1024)
        self.batch1 = torch.nn.BatchNorm1d(1024)
        self.linear2 = torch.nn.Linear(1024, 512)
        self.batch2 = torch.nn.BatchNorm1d(512)
        self.linear3 = torch.nn.Lienar(512, configs.num_classes)

    def forward(self, data):

        B, C, H, W = data.shape

        print(data.shape)

        data = data.view(B, -1)

        layer_1 = torch.nn.functional.relu(self.batch1(self.linear1(data)))
        layer_2 = torch.nn.functional.relu(self.batch2(self.linear2(layer_1)))
        layer_3 = self.linear3(layer_2)

        return layer_3
