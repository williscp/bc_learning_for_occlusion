class Config():

    def __init__(self):

        # environment

        self.device = 'cuda' # 'cpu' or 'cuda' depending on environment

        # paths:

        self.data_dir = 'data/CMU_KO8'
        self.output_dir = 'output'
        self.model_save_path = 'saves'

        # data

        self.num_classes = 8
        self.image_height = 480
        self.image_width = 640
        self.visualize_data = False

        self.randomized_background = True # apply random bg during training
        self.load_into_memory = True # may take a lot of memory
        # crop to localized objects for easier classification task
        self.apply_cropping = True

        # training

        self.batch_size = 16
        self.epochs = 100
        self.lr = 1e-4
