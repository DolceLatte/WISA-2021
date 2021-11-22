from tqdm import tqdm
import pandas as pd
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

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


def labelEncoder(y):
    m = {'bcf':0 ,'sub':1,'fla':2,'bcf_fla':3,'bcf_sub':4,'sub_fla':5,'original':6}

    y = list(map(lambda x:m.get(x),y))
    return y

def t_sne(X,y):
    y = np.array(y)
    X = np.array(X)

    print(len(X))

    tsne = TSNE(random_state=0,metric='cosine')
    digits_tsne = tsne.fit_transform(X)


    colors = ['#476A2A', '#7851B8', '#BD3430', '#4A2D4E', '#875525',
              '#A83683', '#4E655E', '#853541', '#3A3120', '#535D8E']

    for i in range(len(X)):  # 0부터  digits.data까지 정수
        plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(y[i]), color=colors[y[i]],
                 fontdict={'weight': 'bold', 'size': 9}
                 )
    plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max())  # 최소, 최대
    plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max())  # 최소, 최대
    plt.xlabel('t-SNE 0')  # x축 이름
    plt.ylabel('t-SNE 1')  # y축 이름
    plt.show()  # 그래프 출력

if __name__ == '__main__':
    #l, X, y = load_data('./filename/label_gcc_removed_2.csv', '/dataset/gcc_best')
    #l, X, y = load_data('./FilePreprocessing/gcc_fileProcess_label.csv', '/dataset/gcc_filePreprocessing')
    l, X, y = load_data('./FilePreprocessing/filelist/label_gcc_removed_smalldata.csv', './FilePreprocessing/gcc_top_15')
    y = labelEncoder(y)
    t_sne(X,y)