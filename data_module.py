import lightning as L
import os
import requests
from zipfile import ZipFile
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch


class UCIDataModule(L.LightningDataModule):
    def __init__(self, return_index_as_label=False):
        self._return_index_as_label = return_index_as_label
        self.setup()

    def setup(self):
        # Check if the data folder exists
        if not os.path.exists('data'):
            # Check if the zip file exists
            if not os.path.exists('data.zip'):
                print('Downloading the data...')
                request = requests.get('https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip')
                open('data.zip', 'wb').write(request.content)
            # Unzip the file
            print('Unzipping...')
            with ZipFile('data.zip', 'r') as zip_ref:
                zip_ref.extractall('data')
            with ZipFile('data/UCI HAR Dataset.zip', 'r') as zip_ref:
                zip_ref.extractall('data/extracted')
        
        train_acc_x = pd.read_csv('data/extracted/UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt', header=None, sep='\s+')
        train_acc_y = pd.read_csv('data/extracted/UCI HAR Dataset/train/Inertial Signals/body_acc_y_train.txt', header=None, sep='\s+')
        train_acc_z = pd.read_csv('data/extracted/UCI HAR Dataset/train/Inertial Signals/body_acc_z_train.txt', header=None, sep='\s+')
        train_gyro_x = pd.read_csv('data/extracted/UCI HAR Dataset/train/Inertial Signals/body_gyro_x_train.txt', header=None, sep='\s+')
        train_gyro_y = pd.read_csv('data/extracted/UCI HAR Dataset/train/Inertial Signals/body_gyro_y_train.txt', header=None, sep='\s+')
        train_gyro_z = pd.read_csv('data/extracted/UCI HAR Dataset/train/Inertial Signals/body_gyro_z_train.txt', header=None, sep='\s+')
        train_total_acc_x = pd.read_csv('data/extracted/UCI HAR Dataset/train/Inertial Signals/total_acc_x_train.txt', header=None, sep='\s+')
        train_total_acc_y = pd.read_csv('data/extracted/UCI HAR Dataset/train/Inertial Signals/total_acc_y_train.txt', header=None, sep='\s+')
        train_total_acc_z = pd.read_csv('data/extracted/UCI HAR Dataset/train/Inertial Signals/total_acc_z_train.txt', header=None, sep='\s+')

        self._train_x = pd.concat([train_acc_x, train_acc_y, train_acc_z, train_gyro_x, train_gyro_y, train_gyro_z, train_total_acc_x, train_total_acc_y, train_total_acc_z], axis=1)
        self._train_x = torch.tensor(self._train_x.values, dtype=torch.float32)
        if self._return_index_as_label:
            self._train_y = torch.arange(len(self._train_x))
        else:
            self._train_y = pd.read_csv('data/extracted/UCI HAR Dataset/train/y_train.txt', header=None, sep='\s+') - 1
            self._train_y = torch.tensor(self._train_y.values, dtype=torch.long)

        test_acc_x = pd.read_csv('data/extracted/UCI HAR Dataset/test/Inertial Signals/body_acc_x_test.txt', header=None, sep='\s+')
        test_acc_y = pd.read_csv('data/extracted/UCI HAR Dataset/test/Inertial Signals/body_acc_y_test.txt', header=None, sep='\s+')
        test_acc_z = pd.read_csv('data/extracted/UCI HAR Dataset/test/Inertial Signals/body_acc_z_test.txt', header=None, sep='\s+')
        test_gyro_x = pd.read_csv('data/extracted/UCI HAR Dataset/test/Inertial Signals/body_gyro_x_test.txt', header=None, sep='\s+')
        test_gyro_y = pd.read_csv('data/extracted/UCI HAR Dataset/test/Inertial Signals/body_gyro_y_test.txt', header=None, sep='\s+')
        test_gyro_z = pd.read_csv('data/extracted/UCI HAR Dataset/test/Inertial Signals/body_gyro_z_test.txt', header=None, sep='\s+')
        test_total_acc_x = pd.read_csv('data/extracted/UCI HAR Dataset/test/Inertial Signals/total_acc_x_test.txt', header=None, sep='\s+')
        test_total_acc_y = pd.read_csv('data/extracted/UCI HAR Dataset/test/Inertial Signals/total_acc_y_test.txt', header=None, sep='\s+')
        test_total_acc_z = pd.read_csv('data/extracted/UCI HAR Dataset/test/Inertial Signals/total_acc_z_test.txt', header=None, sep='\s+')

        self._test_x = pd.concat([test_acc_x, test_acc_y, test_acc_z, test_gyro_x, test_gyro_y, test_gyro_z, test_total_acc_x, test_total_acc_y, test_total_acc_z], axis=1)
        self._test_x = torch.tensor(self._test_x.values, dtype=torch.float32)
        
        self._test_y = pd.read_csv('data/extracted/UCI HAR Dataset/test/y_test.txt', header=None, sep='\s+') - 1
        self._test_y = torch.tensor(self._test_y.values, dtype=torch.long)
        return
    
    def train_dataloader(self):
        return DataLoader(SimpleDataset(self._train_x, self._train_y), batch_size=128, shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(SimpleDataset(self._test_x, self._test_y), batch_size=128, shuffle=False)


class SimpleDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        # if self._return_index_as_label:
        #     label = idx
        # else:
        #     label = 
        return self.x_data[idx], self.y_data[idx]