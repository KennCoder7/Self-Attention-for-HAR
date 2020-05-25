import numpy as np
import pandas as pd
import time


def load_data(which='oppo'):
    if which == 'oppo_denull':
        train_x = np.load('/home/wangkun/project/DeepConvLSTM/data/oppo/train_x_denull.npy')
        train_y = np.asarray(pd.get_dummies(np.load('/home/wangkun/project/DeepConvLSTM/data/oppo/train_y_denull.npy')))
        test_x = np.load('/home/wangkun/project/DeepConvLSTM/data/oppo/test_x_denull.npy')
        test_y = np.asarray(pd.get_dummies(np.load('/home/wangkun/project/DeepConvLSTM/data/oppo/test_y_denull.npy')))
        print('#  Opportunity data loaded.')
        return train_x, train_y, test_x, test_y
    elif which == 'oppo_denull_local':
        train_x = np.load('E:\Python_project\DeepConvLSTM\data\oppo/train_x_denull.npy')
        train_y = np.asarray(pd.get_dummies(np.load('E:\Python_project\DeepConvLSTM\data\oppo/train_y_denull.npy')))
        test_x = np.load('E:\Python_project\DeepConvLSTM\data\oppo/test_x_denull.npy')
        test_y = np.asarray(pd.get_dummies(np.load('E:\Python_project\DeepConvLSTM\data\oppo/test_y_denull.npy')))
        print('#  Opportunity data loaded.')
        return train_x, train_y, test_x, test_y
    elif which == 'uci':
        train_x = np.load('/home/wangkun/project/EmbeddedHAR/data/UCI_HAR/np_train_x.npy')
        train_y = np.load('/home/wangkun/project/EmbeddedHAR/data/UCI_HAR/np_train_y.npy')
        test_x = np.load('/home/wangkun/project/EmbeddedHAR/data/UCI_HAR/np_test_x.npy')
        test_y = np.load('/home/wangkun/project/EmbeddedHAR/data/UCI_HAR/np_test_y.npy')
        print('#  UCI data loaded.')
        return train_x, train_y, test_x, test_y

    else:
        pass


def shuffle(data, labels):
    index = np.arange(len(data))
    np.random.shuffle(index)
    return data[index], labels[index]


def process_bar(title, current_state, total_state, bar_length=20):
    current_bar = int(current_state / total_state * bar_length)
    bar = ['['] + ['#'] * current_bar + ['-'] * (bar_length - current_bar) + [']']
    bar_show = ''.join(bar)
    print('\r{}{}%d%%'.format(title, bar_show) % ((current_state + 1) / total_state * 100), end='')
    if current_state == total_state - 1:
        bar = ['['] + ['#'] * bar_length + [']']
        bar_show = ''.join(bar)
        print('\r{}{}%d%%'.format(title, bar_show) % 100, end='')
        print('\r')
