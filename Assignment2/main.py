from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchsummary import summary
import focal_loss
from focal_loss import FocalLoss
import numpy
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from argparse import Namespace
import time

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

torch.manual_seed(args.seed)

### Data Initialization and Loading
from data import initialize_data, data_transforms # data.py in the same folder
initialize_data(args.data) # extracts the zip files, makes a validation set

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from model import Net
from googLeNet import GoogLeNet
model = GoogLeNet().to(device)
#model = Net().to(device)

#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
loss = FocalLoss(class_num = 43, gamma=1.5, size_average = False)

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def plot_confusion_matrix(y_true,y_pred):
    cm_array = confusion_matrix(y_true,y_pred)
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    plt.imshow(cm_array, interpolation='nearest', cmap=plt.cm.jet)
    plt.title("Confusion matrix", fontsize=16)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('Number of images', rotation=270, labelpad=30, fontsize=12)
    xtick_marks = np.arange(len(true_labels))
    ytick_marks = np.arange(len(pred_labels))
    plt.xticks(xtick_marks, true_labels, rotation=90)
    plt.yticks(ytick_marks,pred_labels)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 12
    plt.rcParams["figure.figsize"] = fig_size



def train(epoch):
    model.train()
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
#    imshow(torchvision.utils.make_grid(local_contrast_norm(images.to(device), radius=12).cpu()))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(target).to(device)
        optimizer.zero_grad()
        output = model(data)
        #loss = F.nll_loss(output, target)
        #loss = F.cross_entropy(output,target)

        training_loss = loss(output,target)
        training_loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), training_loss))
    return training_loss, (100. * batch_idx) / len(train_loader)

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    f_score = 0
    y_true = []
    y_pred = []
    for data, target in val_loader:
        data, target = Variable(data, volatile=True).to(device), Variable(target).to(device)
        output = model(data)
        validation_loss += loss(output,target).data[0]
        #validation_loss += F.cross_entropy(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        y_true += target.cpu().numpy().tolist()
        y_pred += pred.cpu().numpy().tolist()
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    f_score = f1_score(y_true, y_pred, average = 'weighted')
    class_names = [ repr(i) for i in range(43)]
    plot_confusion_matrix(y_true, y_pred)
#    plt.show()
    plt.savefig("confusion_" + repr(time.time()) + ".png")
    plt.clf()
    print(classification_report(y_true, y_pred))
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    print("F score : "+ str(f_score) + " Balanced accuracy : "+ str(balanced_accuracy))
    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    return validation_loss, 100. * correct / len(val_loader.dataset)

validation_accuracies = []
validation_losses = []
training_accuracies = []
training_losses = []
for epoch in range(1, args.epochs + 1):
    #print(summary(model, (3,32,32)))
    training_loss, training_accuracy = train(epoch)
    training_losses.append(training_loss)
    training_accuracies.append(training_accuracy)
    validation_loss, validation_accuracy = validation()
    validation_losses.append(validation_loss)
    validation_accuracies.append(validation_accuracy)
    model_file = 'model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    print('\nSaved model to ' + model_file + '. You can run `python evaluate.py ' + model_file + '` to generate the Kaggle formatted csv file')
    plt.plot(training_accuracies)
    plt.savefig("train_accuracy_" + repr(time.time())+".png")
    plt.clf()
    plt.plot(validation_accuracies)
    plt.savefig("validation_accuracy_" + repr(time.time()) + ".png" )
    plt.clf()
    plt.plot(training_losses)
    plt.savefig("training_losses_" + repr(time.time()) + ".png")
    plt.clf()
    plt.plot(validation_losses)
    plt.savefig("validation_losses_" + repr(time.time()) + ".png")
    plt.clf()
