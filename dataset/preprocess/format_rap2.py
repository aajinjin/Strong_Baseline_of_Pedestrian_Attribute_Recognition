import os
import numpy as np
import random
import pickle
from scipy.io import loadmat
from easydict import EasyDict

np.random.seed(0)
random.seed(0)

group_order = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
               36, 37, 38, 39, 40, 41, 42, 43, 44, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 45, 46, 47, 48, 49, 50, 51, 52, 53]


def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


def generate_data_description(save_dir, reorder):
    data = loadmat(os.path.join(save_dir, 'RAP_annotation/RAP_annotation.mat'))
    data = data['RAP_annotation']
    dataset = EasyDict()
    dataset.description = 'rap2'
    dataset.reorder = 'group_order'
    dataset.root = os.path.join(save_dir, 'RAP_dataset')
    dataset.image_name = [data['name'][0][0][i][0][0] for i in range(84928)]
    raw_attr_name = [data['attribute'][0][0][i][0][0] for i in range(152)]
    raw_label = data['data'][0][0]
    selected_attr_idx = data['selected_attribute'][0][0][0] - 1

    dataset.label = raw_label[:, selected_attr_idx]
    dataset.attr_name = [raw_attr_name[i] for i in selected_attr_idx]
    if reorder:
        dataset.label = dataset.label[:, group_order]
        dataset.attr_name = [dataset.attr_name[i] for i in group_order]

    dataset.partition = EasyDict()
    dataset.partition.train = []
    dataset.partition.val = []
    dataset.partition.test = []
    dataset.partition.trainval = []

    dataset.weight_train = []
    dataset.weight_trainval = []

    for idx in range(5):
        train = data['partition_attribute'][0][0][0][idx]['train_index'][0][0][0] - 1
        val = data['partition_attribute'][0][0][0][idx]['val_index'][0][0][0] - 1
        test = data['partition_attribute'][0][0][0][idx]['test_index'][0][0][0] - 1
        trainval = np.concatenate([train, val])
        dataset.partition.train.append(train)
        dataset.partition.val.append(val)
        dataset.partition.test.append(test)
        dataset.partition.trainval.append(trainval)
        # cls_weight
        weight_train = np.mean(dataset.label[train], axis=0)
        weight_trainval = np.mean(dataset.label[trainval], axis=0)
        dataset.weight_train.append(weight_train)
        dataset.weight_trainval.append(weight_trainval)
    with open(os.path.join(save_dir, 'dataset.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    save_dir = './data/RAP2/'
    reorder = True
    generate_data_description(save_dir, reorder)
