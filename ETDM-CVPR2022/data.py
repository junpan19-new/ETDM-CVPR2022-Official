#from load_duf import DataloadFromFolder
from load_train import DataloadFromFolder
from load_test import DataloadFromFolderTest
from torchvision.transforms import Compose, ToTensor

def transform():
    return Compose([
             ToTensor(),
            ])

def get_training_set(data_dir, upscale_factor, data_augmentation, file_list, data_type):
    return DataloadFromFolder(data_dir, upscale_factor, data_augmentation, file_list, data_type, transform=transform())

def get_test_set(data_dir, file_list, upscale_factor,  test_name):
    return DataloadFromFolderTest(data_dir, file_list, upscale_factor, test_name,transform=transform())
