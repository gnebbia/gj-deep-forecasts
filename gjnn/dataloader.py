from torch.utils import data
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class Dataset(data.Dataset):
  def __init__(self, df):
        user_1_features = [0,1,2,3,4,5,6,7,8,9,10,11,15]
        user_2_features = [0,1,2,3,4,5,16,17,18,19,20,21,22]
        self.user_1 = df.iloc[:, user_1_features]
        self.user_2 = df.iloc[:, user_2_features]
        self.df = df

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        user_1_data = np.array(self.user_1.iloc[index].tolist(), dtype=np.float32)
        user_2_data = np.array(self.user_2.iloc[index].tolist(), dtype=np.float32)
        user_1_dist = np.array(self.user_1.iloc[index]["distance_1"].tolist(), dtype=np.float32)
        user_2_dist = np.array(self.user_2.iloc[index]["distance_2"].tolist(), dtype=np.float32)

        # Load data and get label
        return user_1_data, user_2_data, user_1_dist, user_2_dist
