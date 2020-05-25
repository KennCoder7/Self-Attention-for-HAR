from mnist import get_mnist
from svhn import get_svhn
from pamap2 import get_pamap2
from oppo import get_oppo


def load_data(dt_name):
    if dt_name == 'mnist':
        train_loader, test_loader = get_mnist()
    elif dt_name == 'svhn':
        train_loader, test_loader = get_svhn()
    elif dt_name == 'pamap2_hand':
        train_loader, test_loader = get_pamap2('pamap2_hand')
    elif dt_name == 'oppo':
        train_loader, test_loader = get_oppo('oppo_denull')
    else:
        train_loader, test_loader = get_mnist()
    counter = 0
    for idx, (x, y) in enumerate(train_loader):
        if idx == 0:
            x_shape = x.shape
            y_shape = y.shape
        counter += 1
    train_x_shape = [counter, x_shape, y_shape]
    counter = 0
    for idx, (x, y) in enumerate(test_loader):
        if idx == 0:
            x_shape = x.shape
            y_shape = y.shape
        counter += 1
    test_x_shape = [counter, x_shape, y_shape]

    print('Successfully load dataset[{}]'
          '\nThe train loader shape:[{}]'
          '\nThe test loader shape:[{}]'.format(
        dt_name, train_x_shape,
        test_x_shape
    ))
    return train_loader, test_loader
