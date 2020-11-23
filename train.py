import argparse
from dataset import dataset_RECT
from models import SmoothOhemLoss, VGGNet
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os

parser = argparse.ArgumentParser('')
parser.add_argument('-retrain', dest='retrain', help='whether to retrain the model', action='store_true')
parser.add_argument('-bs', dest='batch_size', type=int, default=4)
parser.add_argument('-epoch', dest='epoch', type=int, default=200)
parser.add_argument('-dataset', dest='dataset', nargs='?', type=str, default='IC15')

use_cuda = torch.cuda.is_available()

momentum = 0.9
weight_decay = 5e-4
train_epochs = 400


def train(args):
    img_size = (512, 512)
    batch_size = args.batch_size
    if args.dataset == 'IC13':
        dataset = dataset_RECT.RectDataset(img_size, crop=True, light=True,
                                           rotate=False, filp=True, noise=True, color=True)
    else:
        dataset = dataset_RECT.RectDataset(img_size, crop=True, light=True,
                                           rotate=True, filp=True, noise=True, color=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VGGNet()

    if args.dataset == "IC15":
        save_model = "./patchs/icdar15_model/"
    elif args.dataset == "IC13":
        save_model = "./patchs/icdar13_model/"
    elif args.dataset == "IC17":
        save_model = "./patchs/icdar17_model/"
    elif args.dataset == "MS500":
        save_model = "./patchs/msra500_model/"

    if args.retrain:
        model_static = torch.load(save_model + 'vgg16-{}.mdl'.format(args.epoch))
        model.load_state_dict(model_static)
        learning_rate1 = 1e-4
        learning_rate2 = 1e-5
    else:
        learning_rate1 = 1e-3
        learning_rate2 = 1e-4

    if use_cuda:
        model.cuda()

    optimizer1 = torch.optim.Adam(model.parameters(),
                                  lr=learning_rate1,
                                  weight_decay=weight_decay)
    optimizer2 = torch.optim.Adam(model.parameters(),
                                  lr=learning_rate2,
                                  weight_decay=weight_decay)
    loss_reg = SmoothOhemLoss()

    loss_reg = loss_reg.cuda()
    model.train()
    print('Start training')
    total_loss = 0.0
    for epoch in range(args.epoch+1, train_epochs+1):
        if epoch < 200:
            optimizer = optimizer1
        else:
            optimizer = optimizer2
        for i, sample in enumerate(dataloader):
            img = sample['image'].cuda()
            img = Variable(img)
            pixel_mask = Variable(sample['pixel_mask'].cuda())
            pixel_ignore = Variable(sample['pixel_ignore'].cuda())
            pixel_weight = Variable(sample['pixel_weight'].cuda())
            pred_pixel  = model(img)
            pred_pixel = pred_pixel.squeeze(1)
            loss = loss_reg(pred_pixel, pixel_mask, pixel_ignore, pixel_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch {}/{}. The total loss is {:.5}.'
                      .format(epoch, train_epochs, loss))
            total_loss += loss.item()
        print('Epoch {}/{} on all dataset. The total loss is {:.5}.'
              .format(epoch, train_epochs, total_loss))

        total_loss = 0.0

        if epoch % 10 == 0:
            if not os.path.exists(save_model):
                os.mkdir(save_model)
            torch.save(model.state_dict(), save_model+args.arch+'-{}.mdl'.format(epoch))


if __name__ == '__main__':
    argin = parser.parse_args()
    train(argin)
