import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from matplotlib import pyplot as plt
# from matplotlib import gridspec
import numpy as np
import args
from dataset.load_data import load_data
from torchsummary import summary
from self_attention_gan.sagan_models import Self_Attn

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


class Flatten(nn.Module):
    # 传入输入维度和输出维度
    def __init__(self):
        # 调用父类构造函数
        super(Flatten, self).__init__()

    # 实现forward函数
    def forward(self, input):
        # 保存batch维度，后面的维度全部压平
        return input.view(input.size(0), -1)


class SelfAttnCNN(torch.nn.Module):
    def __init__(self, image_channels, n_classes):
        super(SelfAttnCNN, self).__init__()
        self.self_attn_cnn = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=args.kernel_size, stride=args.stride),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            Self_Attn(64),
            nn.Conv2d(64, 64, kernel_size=args.kernel_size, stride=args.stride),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            Self_Attn(64),
            nn.Conv2d(64, 64, kernel_size=args.kernel_size, stride=args.stride),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            Self_Attn(64),
            nn.Conv2d(64, 64, kernel_size=args.kernel_size, stride=args.stride),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            Self_Attn(64),
        )
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        h = self.self_attn_cnn(x)
        # print(h.shape)
        z = self.fc(h)
        return F.log_softmax(z, dim=1)


cnn = SelfAttnCNN(image_channels=image_channels, n_classes=args.n_classes).cuda()
# model.load_state_dict(torch.load('vae.torch', map_location='cpu'))
# optimizer = torch.optim.Adam([{"params": cnn.parameters()}],
#                              lr=1e-3,
#                              weight_decay=0.01)
optimizer = torch.optim.SGD([{"params": cnn.parameters()}],
                            lr=1e-3,
                            momentum=0.05)
summary(cnn, (1, 128, 113), args.batch_size)
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
