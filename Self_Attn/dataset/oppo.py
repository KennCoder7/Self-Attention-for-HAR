import numpy as np
import torch
import torch.utils.data as Data
import args
import pandas as pd


def load_oppo_npy(which):
    if which == 'oppo_denull':
        train_x = np.load('E:\Python_project\DeepConvLSTM\data\oppo/train_x_denull.npy')
        train_x = np.expand_dims(train_x, axis=1)
        train_y = np.asarray(pd.get_dummies(np.load('E:\Python_project\DeepConvLSTM\data\oppo/train_y_denull.npy')))
        train_y = np.argmax(train_y, axis=-1)
        print(np.unique(train_y))
        test_x = np.load('E:\Python_project\DeepConvLSTM\data\oppo/test_x_denull.npy')
        test_x = np.expand_dims(test_x, axis=1)
        test_y = np.asarray(pd.get_dummies(np.load('E:\Python_project\DeepConvLSTM\data\oppo/test_y_denull.npy')))
        test_y = np.argmax(test_y, axis=-1)
        print(np.unique(test_y))
        print('#  Opportunity data loaded.')
        return train_x, train_y, test_x, test_y
    elif which == 'oppo_denull_seglen128':
        train_x = np.load('E:\Python_project\HAR_dataset_preprocess\opportunity\data\gestures\seglen128/train_x_denull.npy')
        train_x = np.expand_dims(train_x, axis=1)
        train_y = np.asarray(pd.get_dummies(np.load('E:\Python_project\HAR_dataset_preprocess\opportunity\data\gestures\seglen128/train_y_denull.npy')))
        train_y = np.argmax(train_y, axis=-1)
        print(np.unique(train_y))
        test_x = np.load('E:\Python_project\HAR_dataset_preprocess\opportunity\data\gestures\seglen128/test_x_denull.npy')
        test_x = np.expand_dims(test_x, axis=1)
        test_y = np.asarray(pd.get_dummies(np.load('E:\Python_project\HAR_dataset_preprocess\opportunity\data\gestures\seglen128/test_y_denull.npy')))
        test_y = np.argmax(test_y, axis=-1)
        print(np.unique(test_y))
        print('#  Opportunity data loaded.')
        return train_x, train_y, test_x, test_y
    else:
        pass


def get_oppo(which):
    train_x, train_y, test_x, test_y = load_oppo_npy(which)
    train_x = torch.from_numpy(train_x).type(torch.FloatTensor).cuda()
    train_y = torch.from_numpy(train_y).cuda()
    test_x = torch.from_numpy(test_x).type(torch.FloatTensor).cuda()
    test_y = torch.from_numpy(test_y).cuda()
    train = Data.TensorDataset(train_x, train_y)
    train_loader = Data.DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True
    )
    test = Data.TensorDataset(test_x, test_y)
    test_loader = Data.DataLoader(
        test,
        batch_size=args.batch_size,
        shuffle=True
    )
    return train_loader, test_loader

