import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from matplotlib import pyplot as plt
# from matplotlib import gridspec
import numpy as np
import args
from dataset.load_data import load_data
# from torchsummary import summary

# torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
batch_size = args.batch_size
epochs = args.epochs

train_loader, test_loader = load_data('oppo')

image_channels = 1
show_im = False
alpha = 1
enc_shape = []


class DeepConvLSTM(torch.nn.Module):
    def __init__(self, image_channels, n_classes):
        super(DeepConvLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=(5, 1), stride=(1, 1)),
            nn.ReLU(),
            # nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(5, 1), stride=(1, 1)),
            nn.ReLU(),
            # nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(5, 1), stride=(1, 1)),
            nn.ReLU(),
            # nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(5, 1), stride=(1, 1)),
            nn.ReLU(),
            # nn.BatchNorm2d(64),
        )
        self.lstm = nn.LSTM(
            input_size=113 * 64,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        cnn_x = self.cnn(x)
        # print(cnn_x.shape)
        cnn_x = cnn_x.transpose(dim0=1, dim1=2)
        # print(cnn_x.shape)
        cnn_x = cnn_x.reshape([-1, 8, 64 * 113])
        # print(cnn_x.shape)
        lstm_x, (h_n, c_n) = self.lstm(cnn_x)
        # print(lstm_x.shape)
        z = self.fc(lstm_x[:, -1, :])
        # print(z.shape)
        return F.log_softmax(z, dim=1)


cnn = DeepConvLSTM(image_channels=image_channels, n_classes=args.n_classes).cuda()
# model.load_state_dict(torch.load('vae.torch', map_location='cpu'))
optimizer = torch.optim.Adam([{"params": cnn.parameters()}],
                             lr=1e-3,
                             weight_decay=0.01)
# summary(cnn, (1, 128, 113), args.batch_size)
# summary(classifier, 256)
if __name__ == '__main__':
    print('The encoder model: \n', cnn)
    loss_ = nn.NLLLoss()
    for epoch in range(epochs):
        for idx, (train_x, train_y) in enumerate(train_loader):
            cnn.train()

            train_x = train_x.cuda()
            train_y = train_y.cuda()
            train_y = train_y.long()
            prb = cnn(train_x)
            # print(images.shape, recon_images.shape, labels.shape, pred.shape)
            loss = loss_(prb, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # if (epoch + 1) % 1 == 0:
        #     to_print = "Epoch[{}/{}] Loss: total-{:.3f} ". \
        #         format(epoch + 1,
        #                epochs,
        #                loss.item() / len(train_x))
        #     print(to_print)

        if (epoch + 1) % 1 == 0:
            cnn.eval()
            correct = 0  # 初始化预测正确的数据个数为0
            for test_x, test_y in test_loader:
                test_x, test_y = test_x.cuda(), test_y.cuda()
                prb = cnn(test_x)
                pred = prb.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(test_y.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加
            print('\nIn epoch[{}] Test set: Accuracy: {}/{} ({:.2f}%)\n'.format(
                epoch + 1, correct, len(test_loader.dataset),
                100.0 * float(correct) / float(len(test_loader.dataset))))
