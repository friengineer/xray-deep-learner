import os, torch
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import pil_loader

# dataset studies
mura_studies = ['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST']
lera_studies = ['XR_ANKLE', 'XR_FOOT', 'XR_HIP', 'XR_KNEE']
data_cat = ['train', 'valid']

def get_study_level_data(study_type):
    """
    Returns a dictionary, with keys 'train' and 'valid' and respective values as study level dataframes,
    these dataframes contain three columns 'Path', 'Count', 'Label'.

    Args:
        study_type (string): one of the seven study type folder names in 'train/valid/test' dataset
    """
    study_data = {}
    study_label = {'positive': 1, 'negative': 0}

    if study_type in mura_studies:
        dataset = 'MURA-v1.1'
    else:
        dataset = 'LERA'

    # retrieve a list of patient folders and extract the label for each study in each patient folder
    for phase in data_cat:
        BASE_DIR = '%s/%s/%s/' % (dataset, phase, study_type)
        patients = list(os.walk(BASE_DIR))[0][1]
        study_data[phase] = pd.DataFrame(columns=['Path', 'Count', 'Label'])
        i = 0

        for patient in tqdm(patients):
            for study in os.listdir(BASE_DIR + patient):
                label = study_label[study.split('_')[1]]
                path = BASE_DIR + patient + '/' + study + '/'
                study_data[phase].loc[i] = [path, len(os.listdir(path)), label]
                i+=1

    return study_data


def get_all_data(dataset):
    """
    Returns a dictionary, with keys 'train' and 'valid' and respective values as dataframes,
    these dataframes contain three columns 'Path', 'Count', 'Label'.
    """
    study_data = {}
    study_label = {'positive': 1, 'negative': 0}

    if dataset == 'MURA-v1.1':
        study_types = mura_studies
    else:
        study_types = lera_studies

    # retrieve a list of patient folders and extract the label for each study in each patient folder
    for phase in data_cat:
        study_data[phase] = pd.DataFrame(columns=['Path', 'Count', 'Label'])
        i = 0

        for body_part in study_types:
            BASE_DIR = '%s/%s/%s/' % (dataset, phase, body_part)
            patients = list(os.walk(BASE_DIR))[0][1]

            for patient in tqdm(patients):
                for study in os.listdir(BASE_DIR + patient):
                    label = study_label[study.split('_')[1]]
                    path = BASE_DIR + patient + '/' + study + '/'
                    study_data[phase].loc[i] = [path, len(os.listdir(path)), label]
                    i+=1

    return study_data


class ImageDataset(Dataset):
    """Training dataset."""

    def __init__(self, df, transform=None):
        """
        Args:
            df (pd.DataFrame): a pandas DataFrame with image path and labels
            transform (callable, optional): Optional transform to be applied
                on a sample
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        study_path = self.df.iloc[idx, 0]
        count = self.df.iloc[idx, 1]
        images = []

        for i in range(count):
            image = pil_loader(study_path + 'image%s.png' % (i+1))
            images.append(self.transform(image))

        images = torch.stack(images)
        label = self.df.iloc[idx, 2]
        sample = {'images': images, 'label': label}

        return sample


def get_dataloaders(data, batch_size=8, study_level=False):
    '''
    Returns dataloader pipeline with data augmentation.
    '''
    data_transforms = {
        'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: ImageDataset(data[x], transform=data_transforms[x]) for x in data_cat}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in data_cat}

    return dataloaders


if __name__=='main':
    pass
