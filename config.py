class Config():

    def __init__(self):

        # environment

        self.device = 'cpu' # 'cpu' or 'cuda' depending on environment

        # paths:

        self.data_dir = 'data/CMU_KO8'
        self.output_dir = 'output'

        # data

        self.num_classes = 8
        self.image_height = 480
        self.image_width = 640
        self.randomized_background = True
        self.visualize_data = True

        # training

        self.epochs = 3
        self.lr = 1e-3
