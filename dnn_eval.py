import torch
import torch.optim as optim
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import classification_report
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
import os
from collections import Counter

use_cuda = torch.cuda.is_available()
torch.cuda.empty_cache()
torch.cuda.get_device_name(0)
print(use_cuda)

def load_data(fileListPath, folder):
    unfoundedFiles = 0
    df = pd.read_csv(fileListPath, sep=',')
    # numpy array, 2D,
    name_label = df.values
    mixed_name_label = name_label

    dirTargetHaar2D = os.getcwd() + folder

    data_with_padding = list()
    y_label_number = list()
    index = 0

    for entryIndex in tqdm(range(len(tqdm(mixed_name_label)))):
        fetched_name_label = mixed_name_label[entryIndex]
        name_with_extension = fetched_name_label[0]
        pathTargetHaar2D = os.path.join(dirTargetHaar2D, name_with_extension)
        try:
            df_haar = pd.read_csv(pathTargetHaar2D, sep=',', header=None)
            data_non_pad = df_haar.values.reshape(-1).tolist()

            data_with_padding.append(data_non_pad)
            y_label = mixed_name_label[entryIndex][1]
            y_label_number.append(y_label)
            index += 1

        except FileNotFoundError:
            print("File does not exist: " + name_with_extension)
            unfoundedFiles += 1

    y_label_category = y_label_number

    return mixed_name_label, data_with_padding, y_label_category

class MLPRegressor(nn.Module):
    def __init__(self):
        # 항상 torch.nn.Module을 상속받고 시작
        super(MLPRegressor, self).__init__()

        fc = nn.Linear(in_features=27, out_features=22)
        fc2 = nn.Linear(in_features=22, out_features=13)
        fc3 = nn.Linear(in_features=13, out_features=7)
        #fc4 = nn.Linear(in_features=30, out_features=7)
        self.fc_module = nn.Sequential(
            fc,
            nn.ReLU(),
            fc2,
            fc3,
         #   fc4,
        )

        # gpu로 할당
        if use_cuda:
            self.fc_module = self.fc_module.cuda()

    def forward(self, x):
        out = self.fc_module(x)
        return F.softmax(out, dim=1)
#
# class MLPRegressor(nn.Module):
#     def __init__(self):
#         # 항상 torch.nn.Module을 상속받고 시작
#         super(MLPRegressor, self).__init__()
#
#         fc = nn.Linear(in_features=19, out_features=30)
#         fc2 = nn.Linear(in_features=30, out_features=10)
#         fc3 = nn.Linear(in_features=10, out_features=7)
#
#         self.fc_module = nn.Sequential(
#             fc,
#             nn.ReLU(),
#             fc2,
#             fc3,
#         )
#
#         # gpu로 할당
#         if use_cuda:
#             self.fc_module = self.fc_module.cuda()
#
#     def forward(self, x):
#         out = self.fc_module(x)
#         return F.softmax(out, dim=1)

def labelEncoder(y):
    m = {'bcf':0 ,'sub':1,'fla':2,'bcf_fla':3,'bcf_sub':4,'sub_fla':5,'original':6}

    y = list(map(lambda x:m.get(x),y))
    return y
if __name__ == '__main__':

    MLP = MLPRegressor()
    MLP.load_state_dict(torch.load("./model/gcc_top_15_nomov.pth"))
    MLP.eval()

    l, X_test, y_test = load_data('./filename/label_tigress_rename.csv', '/FilePreprocessing/tigress_top_15_nomov')

    y_test = labelEncoder(y_test)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.int64)

    testset = TensorDataset(X_test, y_test)
    val_loader = DataLoader(testset, batch_size=64, shuffle=False)

    e = []
    y_pred = np.empty(0)
    y_true = np.empty(0)

    with torch.no_grad():
        corr_num = 0
        total_num = 0
        for j, val in enumerate(val_loader):
            val_x, val_label = val
            if use_cuda:
                val_x = val_x.cuda()
                val_label = val_label.cuda()
            val_output = MLP(val_x)
            model_label = val_output.argmax(dim=1)

            m = model_label.cpu().numpy()
            v = val_label.cpu().numpy()
            y_pred = np.concatenate((y_pred, m), axis=None)
            y_true = np.concatenate((y_true, v), axis=None)

            corr = val_label[val_label == model_label].size(0)
            corr_num += corr
            total_num += val_label.size(0)

    print("acc: {:.2f}".format(corr_num / total_num * 100))
    print(classification_report(y_true, y_pred))
