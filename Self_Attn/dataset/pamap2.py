import numpy as np
import torch
import torch.utils.data as Data
import args


def load_pamap2_npy(which):
    if which == 'pamap2_hand':
        train_x = np.load('./data/PAMAP2/preprocessed2/X_train_norm.npy')[:, :, 0:3]
        train_x = np.expand_dims(train_x, axis=1)
        train_y = np.load('./data/PAMAP2/preprocessed2/y_train.npy')
        train_y = np.argmax(train_y, axis=-1)
        test_x = np.load('./data/PAMAP2/preprocessed2/X_test_norm.npy')[:, :, 0:3]
        test_x = np.expand_dims(test_x, axis=1)
        test_y = np.load('./data/PAMAP2/preprocessed2/y_test.npy')
        test_y = np.argmax(test_y, axis=-1)
        print('#  pamap2 2 hand_acc data loaded.')
        return train_x, train_y, test_x, test_y
    elif which == 'pamap2_ankle':
        train_x = np.load('./data/PAMAP2/preprocessed2/X_train_norm.npy')[:, :, 3:6]
        train_x = np.expand_dims(train_x, axis=1)
        train_y = np.load('./data/PAMAP2/preprocessed2/y_train.npy')
        train_y = np.argmax(train_y, axis=-1)
        test_x = np.load('./data/PAMAP2/preprocessed2/X_test_norm.npy')[:, :, 3:6]
        test_x = np.expand_dims(test_x, axis=1)
        test_y = np.load('./data/PAMAP2/preprocessed2/y_test.npy')
        test_y = np.argmax(test_y, axis=-1)
        print('#  pamap2 2 ankle_acc data loaded.')
        return train_x, train_y, test_x, test_y
    elif which == 'pamap2_chest':
        train_x = np.load('./data/PAMAP2/preprocessed2/X_train_norm.npy')[:, :, 6:9]
        train_x = np.expand_dims(train_x, axis=1)
        train_y = np.load('./data/PAMAP2/preprocessed2/y_train.npy')
        train_y = np.argmax(train_y, axis=-1)
        test_x = np.load('./data/PAMAP2/preprocessed2/X_test_norm.npy')[:, :, 6:9]
        test_x = np.expand_dims(test_x, axis=1)
        test_y = np.load('./data/PAMAP2/preprocessed2/y_test.npy')
        test_y = np.argmax(test_y, axis=-1)
        print('#  pamap2 2 chest_acc data loaded.')
        return train_x, train_y, test_x, test_y
    else:
        pass


def get_pamap2(which):
    train_x, train_y, test_x, test_y = load_pamap2_npy(which)
    train_x = torch.from_numpy(train_x).type(torch.FloatTensor).cuda()
    train_y = torch.from_numpy(train_y).type(torch.LongTensor).cuda()
    test_x = torch.from_numpy(test_x).type(torch.FloatTensor).cuda()
    test_y = torch.from_numpy(test_y).type(torch.LongTensor).cuda()
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

