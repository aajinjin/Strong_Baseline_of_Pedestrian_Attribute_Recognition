import os
import numpy as np
import random
import pickle

from easydict import EasyDict
from scipy.io import loadmat

np.random.seed(0)
random.seed(0)

# note: ref by annotation.md

group_order = [10, 18, 19, 30, 15, 7, 9, 11, 14, 21, 26, 29, 32, 33, 34, 6, 8, 12, 25, 27, 31, 13, 23, 24, 28, 4, 5,
               17, 20, 22, 0, 1, 2, 3, 16]


def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


def generate_data_description(save_dir, reorder):
    """
    create a dataset description file, which consists of images, labels
    """
    peta_data = loadmat(os.path.join(save_dir, 'PETA.mat'))
    dataset = EasyDict()
    dataset.description = 'peta'
    dataset.reorder = 'group_order'
    dataset.root = os.path.join(save_dir, 'images')
    dataset.image_name = [f'{i + 1:05}.png' for i in range(19000)]

    raw_attr_name = [i[0][0] for i in peta_data['peta'][0][0][1]]
    # (19000, 105)
    raw_label = peta_data['peta'][0][0][0][:, 4:]

    # (19000, 35)
    dataset.label = raw_label[:, :35]
    dataset.attr_name = raw_attr_name[:35]
    if reorder:
        dataset.label = dataset.label[:, np.array(group_order)]
        dataset.attr_name = [dataset.attr_name[i] for i in group_order]

    dataset.partition = EasyDict()
    dataset.partition.train = []
    dataset.partition.val = []
    dataset.partition.trainval = []
    dataset.partition.test = []

    dataset.weight_train = []
    dataset.weight_trainval = []

    for idx in range(5):
        train = peta_data['peta'][0][0][3][idx][0][0][0][0][:, 0] - 1
        val = peta_data['peta'][0][0][3][idx][0][0][0][1][:, 0] - 1
        test = peta_data['peta'][0][0][3][idx][0][0][0][2][:, 0] - 1
        trainval = np.concatenate((train, val), axis=0)

        dataset.partition.train.append(train)
        dataset.partition.val.append(val)
        dataset.partition.trainval.append(trainval)
        dataset.partition.test.append(test)

        weight_train = np.mean(dataset.label[train], axis=0)
        weight_trainval = np.mean(dataset.label[trainval], axis=0)

        dataset.weight_train.append(weight_train)
        dataset.weight_trainval.append(weight_trainval)
    with open(os.path.join(save_dir, 'dataset.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    save_dir = './data/PETA/'

    generate_data_description(save_dir, True)
