import args
import torch
import torchvision
from torchvision import transforms


def get_svhn():
    transform = transforms.Compose(
        [transforms.ToTensor()])
    svhn = torchvision.datasets.SVHN('./data/', download=True, transform=transform, split='train')

    dataloader = torch.utils.data.DataLoader(svhn, batch_size=args.batch_size,
                                             shuffle=True, num_workers=2)

    svhn_t = torchvision.datasets.SVHN('./data/', download=True, transform=transform, split='test')

    test_loader = torch.utils.data.DataLoader(svhn_t, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)
    return dataloader, test_loader