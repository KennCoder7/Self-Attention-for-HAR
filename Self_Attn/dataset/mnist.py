import args
import torch
import torchvision
from torchvision import transforms


def get_mnist():
    transform = transforms.Compose(
        [transforms.Resize([32, 32]),
         transforms.Grayscale(3),
         transforms.ToTensor(),
         ])
    mnist = torchvision.datasets.MNIST('./data/', download=True, transform=transform, train=True)

    dataloader = torch.utils.data.DataLoader(mnist, batch_size=args.batch_size,
                                             shuffle=True, num_workers=2)

    mnist_t = torchvision.datasets.MNIST('./data/', download=True, transform=transform, train=False)

    test_loader = torch.utils.data.DataLoader(mnist_t, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)
    return dataloader, test_loader
