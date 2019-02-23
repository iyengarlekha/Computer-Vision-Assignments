import torch
import torch.nn as nn
import torch.nn.functional as F
import local_contrast_normalization
#from local_contrast_normalization import local_contrast_norm
from local_contrast_normalization import local_contrast_norm as lcn

nclasses = 43 # GTSRB as 43 classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#class Net(nn.Module):
#    def __init__(self):
#        super(Net, self).__init__()
#        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
#        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#        self.conv2_drop = nn.Dropout2d()
#        self.fc1 = nn.Linear(500, 50)
#        self.fc2 = nn.Linear(50, nclasses)

#    def forward(self, x):
#        x = F.relu(F.max_pool2d(self.conv1(x), 2))
#        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#        x = x.view(-1, 500)
#        x = F.relu(self.fc1(x))
#        x = F.dropout(x, training=self.training)
#        x = self.fc2(x)
#        return F.log_softmax(x)

#Architecture 1. Best perfrormance - 97.387
#----------------------------------------------------------------
#        Layer (type)               Output Shape         Param #
#================================================================
#            Conv2d-1          [-1, 100, 46, 46]           2,800
#            Conv2d-2          [-1, 150, 20, 20]         240,150
#         Dropout2d-3          [-1, 150, 20, 20]               0
#            Conv2d-4            [-1, 250, 8, 8]         337,750
#         Dropout2d-5            [-1, 250, 8, 8]               0
#            Linear-6                  [-1, 200]         200,200
#            Linear-7                   [-1, 43]           8,643
#================================================================
#Total params: 789,543
#Trainable params: 789,543
#Non-trainable params: 0
#----------------------------------------------------------------
#Input size (MB): 0.03
#Forward/backward pass size (MB): 2.78
#Params size (MB): 3.01
#Estimated Total Size (MB): 5.81
#----------------------------------------------------------------
#class Net(nn.Module):
#    def __init__(self):
#        super(Net, self).__init__()
#        self.conv1 = nn.Conv2d(3, 100, kernel_size=3)
#        self.conv2 = nn.Conv2d(100, 150, kernel_size=4)
#        self.conv2_drop = nn.Dropout2d()
#        self.conv3 = nn.Conv2d(150, 250, kernel_size=3)
#        self.conv3_drop = nn.Dropout2d()
#        self.fc1 = nn.Linear(1000, 200)
#        self.fc2 = nn.Linear(200, nclasses)
#
#    def forward(self, x):
#        x = F.relu(F.max_pool2d(self.conv1(x), 2))
#        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
#        x = x.view(-1, 250*4)
#        x = F.relu(self.fc1(x))
#        x = F.dropout(x, training=self.training)
#        x = self.fc2(x)
#        return F.log_softmax(x)



#Architecture 2. Best performance - 96.373
#----------------------------------------------------------------
#        Layer (type)               Output Shape         Param #
#================================================================
#            Conv2d-1          [-1, 100, 42, 42]          14,800
#            Conv2d-2          [-1, 150, 18, 18]         240,150
#         Dropout2d-3          [-1, 150, 18, 18]               0
#            Conv2d-4            [-1, 250, 6, 6]         600,250
#         Dropout2d-5            [-1, 250, 6, 6]               0
#            Linear-6                  [-1, 300]          75,300
#            Linear-7                   [-1, 43]          12,943
#================================================================
#Total params: 943,443
#Trainable params: 943,443
#Non-trainable params: 0
#----------------------------------------------------------------
#Input size (MB): 0.03
#Forward/backward pass size (MB): 2.23
#Params size (MB): 3.60
#Estimated Total Size (MB): 5.85
#----------------------------------------------------------------

#class Net(nn.Module):
#    def __init__(self):
#        super(Net, self).__init__()
#        self.conv1 = nn.Conv2d(3, 100, kernel_size=7)
#        self.conv2 = nn.Conv2d(100, 150, kernel_size=4)
#        self.conv2_drop = nn.Dropout2d()
#        self.conv3 = nn.Conv2d(150, 250, kernel_size=4)
#        self.conv3_drop = nn.Dropout2d()
#        self.fc1 = nn.Linear(250 , 300)
#        self.fc2 = nn.Linear(300, nclasses)
#
#    def forward(self, x):
#        x = F.relu(F.max_pool2d(self.conv1(x), 2))
#        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
#        x = x.view(-1, 250)
#        x = F.relu(self.fc1(x))
#        x = F.dropout(x, training=self.training)
#        x = self.fc2(x)
#        return F.log_softmax(x)



#Architecture 3
#----------------------------------------------------------------
#        Layer (type)               Output Shape         Param #
#================================================================
#            Conv2d-1          [-1, 200, 46, 46]           5,600
#            Conv2d-2          [-1, 250, 20, 20]         800,250
#         Dropout2d-3          [-1, 250, 20, 20]               0
#            Conv2d-4            [-1, 350, 8, 8]         787,850
#         Dropout2d-5            [-1, 350, 8, 8]               0
#            Linear-6                  [-1, 300]         420,300
#            Linear-7                   [-1, 43]          12,943
#================================================================
#Total params: 2,026,943
#Trainable params: 2,026,943
#Non-trainable params: 0
#----------------------------------------------------------------
#Input size (MB): 0.03
#Forward/backward pass size (MB): 5.10
#Params size (MB): 7.73
#Estimated Total Size (MB): 12.86
#----------------------------------------------------------------
#
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 200, kernel_size=3)
        self.conv2 = nn.Conv2d(200, 250, kernel_size=4)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(250, 350, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1400, 300)
        self.fc2 = nn.Linear(300, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 1400)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


#Architecture 4: Inception v1
#
#----------------------------------------------------------------
#        Layer (type)               Output Shape         Param #
#================================================================
#            Conv2d-1          [-1, 192, 32, 32]           5,376
#       BatchNorm2d-2          [-1, 192, 32, 32]             384
#              ReLU-3          [-1, 192, 32, 32]               0
#            Conv2d-4           [-1, 64, 32, 32]          12,352
#       BatchNorm2d-5           [-1, 64, 32, 32]             128
#              ReLU-6           [-1, 64, 32, 32]               0
#            Conv2d-7           [-1, 96, 32, 32]          18,528
#       BatchNorm2d-8           [-1, 96, 32, 32]             192
#              ReLU-9           [-1, 96, 32, 32]               0
#           Conv2d-10          [-1, 128, 32, 32]         110,720
#      BatchNorm2d-11          [-1, 128, 32, 32]             256
#             ReLU-12          [-1, 128, 32, 32]               0
#           Conv2d-13           [-1, 16, 32, 32]           3,088
#      BatchNorm2d-14           [-1, 16, 32, 32]              32
#             ReLU-15           [-1, 16, 32, 32]               0
#           Conv2d-16           [-1, 32, 32, 32]           4,640
#      BatchNorm2d-17           [-1, 32, 32, 32]              64
#             ReLU-18           [-1, 32, 32, 32]               0
#           Conv2d-19           [-1, 32, 32, 32]           9,248
#      BatchNorm2d-20           [-1, 32, 32, 32]              64
#             ReLU-21           [-1, 32, 32, 32]               0
#        MaxPool2d-22          [-1, 192, 32, 32]               0
#           Conv2d-23           [-1, 32, 32, 32]           6,176
#      BatchNorm2d-24           [-1, 32, 32, 32]              64
#             ReLU-25           [-1, 32, 32, 32]               0
#        Inception-26          [-1, 256, 32, 32]               0
#           Conv2d-27          [-1, 128, 32, 32]          32,896
#      BatchNorm2d-28          [-1, 128, 32, 32]             256
#             ReLU-29          [-1, 128, 32, 32]               0
#           Conv2d-30          [-1, 128, 32, 32]          32,896
#      BatchNorm2d-31          [-1, 128, 32, 32]             256
#             ReLU-32          [-1, 128, 32, 32]               0
#           Conv2d-33          [-1, 192, 32, 32]         221,376
#      BatchNorm2d-34          [-1, 192, 32, 32]             384
#             ReLU-35          [-1, 192, 32, 32]               0
#           Conv2d-36           [-1, 32, 32, 32]           8,224
#      BatchNorm2d-37           [-1, 32, 32, 32]              64
#             ReLU-38           [-1, 32, 32, 32]               0
#           Conv2d-39           [-1, 96, 32, 32]          27,744
#      BatchNorm2d-40           [-1, 96, 32, 32]             192
#             ReLU-41           [-1, 96, 32, 32]               0
#           Conv2d-42           [-1, 96, 32, 32]          83,040
#      BatchNorm2d-43           [-1, 96, 32, 32]             192
#             ReLU-44           [-1, 96, 32, 32]               0
#        MaxPool2d-45          [-1, 256, 32, 32]               0
#           Conv2d-46           [-1, 64, 32, 32]          16,448
#      BatchNorm2d-47           [-1, 64, 32, 32]             128
#             ReLU-48           [-1, 64, 32, 32]               0
#        Inception-49          [-1, 480, 32, 32]               0
#        MaxPool2d-50          [-1, 480, 16, 16]               0
#           Conv2d-51          [-1, 192, 16, 16]          92,352
#      BatchNorm2d-52          [-1, 192, 16, 16]             384
#             ReLU-53          [-1, 192, 16, 16]               0
#           Conv2d-54           [-1, 96, 16, 16]          46,176
#      BatchNorm2d-55           [-1, 96, 16, 16]             192
#             ReLU-56           [-1, 96, 16, 16]               0
#           Conv2d-57          [-1, 208, 16, 16]         179,920
#      BatchNorm2d-58          [-1, 208, 16, 16]             416
#             ReLU-59          [-1, 208, 16, 16]               0
#           Conv2d-60           [-1, 16, 16, 16]           7,696
#      BatchNorm2d-61           [-1, 16, 16, 16]              32
#             ReLU-62           [-1, 16, 16, 16]               0
#           Conv2d-63           [-1, 48, 16, 16]           6,960
#      BatchNorm2d-64           [-1, 48, 16, 16]              96
#             ReLU-65           [-1, 48, 16, 16]               0
#           Conv2d-66           [-1, 48, 16, 16]          20,784
#      BatchNorm2d-67           [-1, 48, 16, 16]              96
#             ReLU-68           [-1, 48, 16, 16]               0
#        MaxPool2d-69          [-1, 480, 16, 16]               0
#           Conv2d-70           [-1, 64, 16, 16]          30,784
#      BatchNorm2d-71           [-1, 64, 16, 16]             128
#             ReLU-72           [-1, 64, 16, 16]               0
#        Inception-73          [-1, 512, 16, 16]               0
#           Conv2d-74          [-1, 160, 16, 16]          82,080
#      BatchNorm2d-75          [-1, 160, 16, 16]             320
#             ReLU-76          [-1, 160, 16, 16]               0
#           Conv2d-77          [-1, 112, 16, 16]          57,456
#      BatchNorm2d-78          [-1, 112, 16, 16]             224
#             ReLU-79          [-1, 112, 16, 16]               0
#           Conv2d-80          [-1, 224, 16, 16]         226,016
#      BatchNorm2d-81          [-1, 224, 16, 16]             448
#             ReLU-82          [-1, 224, 16, 16]               0
#           Conv2d-83           [-1, 24, 16, 16]          12,312
#      BatchNorm2d-84           [-1, 24, 16, 16]              48
#             ReLU-85           [-1, 24, 16, 16]               0
#           Conv2d-86           [-1, 64, 16, 16]          13,888
#      BatchNorm2d-87           [-1, 64, 16, 16]             128
#             ReLU-88           [-1, 64, 16, 16]               0
#           Conv2d-89           [-1, 64, 16, 16]          36,928
#      BatchNorm2d-90           [-1, 64, 16, 16]             128
#             ReLU-91           [-1, 64, 16, 16]               0
#        MaxPool2d-92          [-1, 512, 16, 16]               0
#           Conv2d-93           [-1, 64, 16, 16]          32,832
#      BatchNorm2d-94           [-1, 64, 16, 16]             128
#             ReLU-95           [-1, 64, 16, 16]               0
#        Inception-96          [-1, 512, 16, 16]               0
#           Conv2d-97          [-1, 128, 16, 16]          65,664
#      BatchNorm2d-98          [-1, 128, 16, 16]             256
#             ReLU-99          [-1, 128, 16, 16]               0
#          Conv2d-100          [-1, 128, 16, 16]          65,664
#     BatchNorm2d-101          [-1, 128, 16, 16]             256
#            ReLU-102          [-1, 128, 16, 16]               0
#          Conv2d-103          [-1, 256, 16, 16]         295,168
#     BatchNorm2d-104          [-1, 256, 16, 16]             512
#            ReLU-105          [-1, 256, 16, 16]               0
#          Conv2d-106           [-1, 24, 16, 16]          12,312
#     BatchNorm2d-107           [-1, 24, 16, 16]              48
#            ReLU-108           [-1, 24, 16, 16]               0
#          Conv2d-109           [-1, 64, 16, 16]          13,888
#     BatchNorm2d-110           [-1, 64, 16, 16]             128
#            ReLU-111           [-1, 64, 16, 16]               0
#          Conv2d-112           [-1, 64, 16, 16]          36,928
#     BatchNorm2d-113           [-1, 64, 16, 16]             128
#            ReLU-114           [-1, 64, 16, 16]               0
#       MaxPool2d-115          [-1, 512, 16, 16]               0
#          Conv2d-116           [-1, 64, 16, 16]          32,832
#     BatchNorm2d-117           [-1, 64, 16, 16]             128
#            ReLU-118           [-1, 64, 16, 16]               0
#       Inception-119          [-1, 512, 16, 16]               0
#          Conv2d-120          [-1, 112, 16, 16]          57,456
#     BatchNorm2d-121          [-1, 112, 16, 16]             224
#            ReLU-122          [-1, 112, 16, 16]               0
#          Conv2d-123          [-1, 144, 16, 16]          73,872
#     BatchNorm2d-124          [-1, 144, 16, 16]             288
#            ReLU-125          [-1, 144, 16, 16]               0
#          Conv2d-126          [-1, 288, 16, 16]         373,536
#     BatchNorm2d-127          [-1, 288, 16, 16]             576
#            ReLU-128          [-1, 288, 16, 16]               0
#          Conv2d-129           [-1, 32, 16, 16]          16,416
#     BatchNorm2d-130           [-1, 32, 16, 16]              64
#            ReLU-131           [-1, 32, 16, 16]               0
#          Conv2d-132           [-1, 64, 16, 16]          18,496
#     BatchNorm2d-133           [-1, 64, 16, 16]             128
#            ReLU-134           [-1, 64, 16, 16]               0
#          Conv2d-135           [-1, 64, 16, 16]          36,928
#     BatchNorm2d-136           [-1, 64, 16, 16]             128
#            ReLU-137           [-1, 64, 16, 16]               0
#       MaxPool2d-138          [-1, 512, 16, 16]               0
#          Conv2d-139           [-1, 64, 16, 16]          32,832
#     BatchNorm2d-140           [-1, 64, 16, 16]             128
#            ReLU-141           [-1, 64, 16, 16]               0
#       Inception-142          [-1, 528, 16, 16]               0
#          Conv2d-143          [-1, 256, 16, 16]         135,424
#     BatchNorm2d-144          [-1, 256, 16, 16]             512
#            ReLU-145          [-1, 256, 16, 16]               0
#          Conv2d-146          [-1, 160, 16, 16]          84,640
#     BatchNorm2d-147          [-1, 160, 16, 16]             320
#            ReLU-148          [-1, 160, 16, 16]               0
#          Conv2d-149          [-1, 320, 16, 16]         461,120
#     BatchNorm2d-150          [-1, 320, 16, 16]             640
#            ReLU-151          [-1, 320, 16, 16]               0
#          Conv2d-152           [-1, 32, 16, 16]          16,928
#     BatchNorm2d-153           [-1, 32, 16, 16]              64
#            ReLU-154           [-1, 32, 16, 16]               0
#          Conv2d-155          [-1, 128, 16, 16]          36,992
#     BatchNorm2d-156          [-1, 128, 16, 16]             256
#            ReLU-157          [-1, 128, 16, 16]               0
#          Conv2d-158          [-1, 128, 16, 16]         147,584
#     BatchNorm2d-159          [-1, 128, 16, 16]             256
#            ReLU-160          [-1, 128, 16, 16]               0
#       MaxPool2d-161          [-1, 528, 16, 16]               0
#          Conv2d-162          [-1, 128, 16, 16]          67,712
#     BatchNorm2d-163          [-1, 128, 16, 16]             256
#            ReLU-164          [-1, 128, 16, 16]               0
#       Inception-165          [-1, 832, 16, 16]               0
#       MaxPool2d-166            [-1, 832, 8, 8]               0
#          Conv2d-167            [-1, 256, 8, 8]         213,248
#     BatchNorm2d-168            [-1, 256, 8, 8]             512
#            ReLU-169            [-1, 256, 8, 8]               0
#          Conv2d-170            [-1, 160, 8, 8]         133,280
#     BatchNorm2d-171            [-1, 160, 8, 8]             320
#            ReLU-172            [-1, 160, 8, 8]               0
#          Conv2d-173            [-1, 320, 8, 8]         461,120
#     BatchNorm2d-174            [-1, 320, 8, 8]             640
#            ReLU-175            [-1, 320, 8, 8]               0
#          Conv2d-176             [-1, 32, 8, 8]          26,656
#     BatchNorm2d-177             [-1, 32, 8, 8]              64
#            ReLU-178             [-1, 32, 8, 8]               0
#          Conv2d-179            [-1, 128, 8, 8]          36,992
#     BatchNorm2d-180            [-1, 128, 8, 8]             256
#            ReLU-181            [-1, 128, 8, 8]               0
#          Conv2d-182            [-1, 128, 8, 8]         147,584
#     BatchNorm2d-183            [-1, 128, 8, 8]             256
#            ReLU-184            [-1, 128, 8, 8]               0
#       MaxPool2d-185            [-1, 832, 8, 8]               0
#          Conv2d-186            [-1, 128, 8, 8]         106,624
#     BatchNorm2d-187            [-1, 128, 8, 8]             256
#            ReLU-188            [-1, 128, 8, 8]               0
#       Inception-189            [-1, 832, 8, 8]               0
#          Conv2d-190            [-1, 384, 8, 8]         319,872
#     BatchNorm2d-191            [-1, 384, 8, 8]             768
#            ReLU-192            [-1, 384, 8, 8]               0
#          Conv2d-193            [-1, 192, 8, 8]         159,936
#     BatchNorm2d-194            [-1, 192, 8, 8]             384
#            ReLU-195            [-1, 192, 8, 8]               0
#          Conv2d-196            [-1, 384, 8, 8]         663,936
#     BatchNorm2d-197            [-1, 384, 8, 8]             768
#            ReLU-198            [-1, 384, 8, 8]               0
#          Conv2d-199             [-1, 48, 8, 8]          39,984
#     BatchNorm2d-200             [-1, 48, 8, 8]              96
#            ReLU-201             [-1, 48, 8, 8]               0
#          Conv2d-202            [-1, 128, 8, 8]          55,424
#     BatchNorm2d-203            [-1, 128, 8, 8]             256
#            ReLU-204            [-1, 128, 8, 8]               0
#          Conv2d-205            [-1, 128, 8, 8]         147,584
#     BatchNorm2d-206            [-1, 128, 8, 8]             256
#            ReLU-207            [-1, 128, 8, 8]               0
#       MaxPool2d-208            [-1, 832, 8, 8]               0
#          Conv2d-209            [-1, 128, 8, 8]         106,624
#     BatchNorm2d-210            [-1, 128, 8, 8]             256
#            ReLU-211            [-1, 128, 8, 8]               0
#       Inception-212           [-1, 1024, 8, 8]               0
#       AvgPool2d-213           [-1, 1024, 1, 1]               0
#          Linear-214                   [-1, 43]          44,075
#================================================================
#Total params: 6,200,075
#Trainable params: 6,200,075
#Non-trainable params: 0
#----------------------------------------------------------------
#Input size (MB): 0.01
#Forward/backward pass size (MB): 81.42
#Params size (MB): 23.65
#Estimated Total Size (MB): 105.09
#----------------------------------------------------------------

#class Inception(nn.Module):
#    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
#        super(Inception, self).__init__()
#        # 1x1 conv branch
#        self.b1 = nn.Sequential(
#            nn.Conv2d(in_planes, n1x1, kernel_size=1),
#            nn.BatchNorm2d(n1x1),
#            nn.ReLU(True),
#        )
#
#        # 1x1 conv -> 3x3 conv branch
#        self.b2 = nn.Sequential(
#            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
#            nn.BatchNorm2d(n3x3red),
#            nn.ReLU(True),
#            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
#            nn.BatchNorm2d(n3x3),
#            nn.ReLU(True),
#        )
#
#        # 1x1 conv -> 5x5 conv branch
#        self.b3 = nn.Sequential(
#            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
#            nn.BatchNorm2d(n5x5red),
#            nn.ReLU(True),
#            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
#            nn.BatchNorm2d(n5x5),
#            nn.ReLU(True),
#            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
#            nn.BatchNorm2d(n5x5),
#            nn.ReLU(True),
#        )
#
#        # 3x3 pool -> 1x1 conv branch
#        self.b4 = nn.Sequential(
#            nn.MaxPool2d(3, stride=1, padding=1),
#            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
#            nn.BatchNorm2d(pool_planes),
#            nn.ReLU(True),
#        )
#
#    def forward(self, x):
#        y1 = self.b1(x)
#        y2 = self.b2(x)
#        y3 = self.b3(x)
#        y4 = self.b4(x)
#        return torch.cat([y1,y2,y3,y4], 1)
#
#
#class GoogLeNet(nn.Module):
#    def __init__(self):
#        super(GoogLeNet, self).__init__()
#        self.pre_layers = nn.Sequential(
#            nn.Conv2d(3, 192, kernel_size=3, padding=1),
#            nn.BatchNorm2d(192),
#            nn.ReLU(True),
#        )
#
#        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
#        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
#
#        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
#
#        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
#        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
#        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
#        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
#        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)
#
#        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
#        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
#
#        self.avgpool = nn.AvgPool2d(8, stride=1)
#        self.linear = nn.Linear(1024, 43)
#
#    def forward(self, x):
#        out = self.pre_layers(x)
#        out = self.a3(out)
#        out = self.b3(out)
#        out = self.maxpool(out)
#        out = self.a4(out)
#        out = self.b4(out)
#        out = self.c4(out)
#        out = self.d4(out)
#        out = self.e4(out)
#        out = self.maxpool(out)
#        out = self.a5(out)
#        out = self.b5(out)
#        out = self.avgpool(out)
#        out = out.view(out.size(0), -1)
#        out = self.linear(out)
#        return out



#Architechture 5 : STN
#----------------------------------------------------------------
#        Layer (type)               Output Shape         Param #
#================================================================
#         MaxPool2d-1            [-1, 3, 24, 24]               0
#            Conv2d-2          [-1, 250, 24, 24]          19,000
#              ReLU-3          [-1, 250, 24, 24]               0
#         MaxPool2d-4          [-1, 250, 12, 12]               0
#            Conv2d-5          [-1, 250, 12, 12]       1,562,750
#              ReLU-6          [-1, 250, 12, 12]               0
#         MaxPool2d-7            [-1, 250, 6, 6]               0
#            Linear-8                  [-1, 250]       2,250,250
#              ReLU-9                  [-1, 250]               0
#           Linear-10                    [-1, 6]           1,506
#       LocNet_ST1-11            [-1, 3, 48, 48]               0
#           Conv2d-12          [-1, 200, 46, 46]          29,600
#        MaxPool2d-13          [-1, 200, 11, 11]               0
#           Conv2d-14          [-1, 150, 11, 11]         750,150
#             ReLU-15          [-1, 150, 11, 11]               0
#        MaxPool2d-16            [-1, 150, 5, 5]               0
#           Conv2d-17            [-1, 200, 5, 5]         750,200
#             ReLU-18            [-1, 200, 5, 5]               0
#        MaxPool2d-19            [-1, 200, 2, 2]               0
#           Linear-20                  [-1, 300]         240,300
#             ReLU-21                  [-1, 300]               0
#           Linear-22                    [-1, 6]           1,806
#       LocNet_ST2-23          [-1, 200, 23, 23]               0
#           Conv2d-24          [-1, 250, 24, 24]         800,250
#        MaxPool2d-25            [-1, 250, 6, 6]               0
#           Conv2d-26            [-1, 150, 6, 6]         937,650
#             ReLU-27            [-1, 150, 6, 6]               0
#        MaxPool2d-28            [-1, 150, 3, 3]               0
#           Conv2d-29            [-1, 200, 3, 3]         750,200
#             ReLU-30            [-1, 200, 3, 3]               0
#        MaxPool2d-31            [-1, 200, 1, 1]               0
#           Linear-32                  [-1, 300]          60,300
#             ReLU-33                  [-1, 300]               0
#           Linear-34                    [-1, 6]           1,806
#       LocNet_ST3-35          [-1, 250, 12, 12]               0
#           Conv2d-36          [-1, 350, 13, 13]       1,400,350
#           Linear-37                  [-1, 400]       5,040,400
#           Linear-38                   [-1, 43]          17,243
#================================================================
#Total params: 14,613,761
#Trainable params: 14,613,761
#Non-trainable params: 0
#----------------------------------------------------------------
#Input size (MB): 0.03
#Forward/backward pass size (MB): 9.80
#Params size (MB): 55.75
#Estimated Total Size (MB): 65.57
#----------------------------------------------------------------


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.st1 = LocNet_ST1()
        self.conv1 = nn.Conv2d(3, 200, kernel_size=7, stride =1, padding =2)
        self.st2 = LocNet_ST2()
        self.conv2 = nn.Conv2d(200, 250, kernel_size=4, stride=1, padding =2)
        self.st3 = LocNet_ST3()
        self.conv3 = nn.Conv2d(250, 350, kernel_size=4, stride=1, padding =2)
        self.fc1 = nn.Linear(350*6*6, 400)
        self.fc_drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(400, nclasses)

    def forward(self, x):
        x = self.st1(x)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = local_contrast_norm(x,radius = 9)
#        x = local_contrast_norm(x,radius = 7)
        x = self.st2(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = local_contrast_norm(x,radius = 9)
#        x = local_contrast_norm(x,radius = 7)
        x = self.st3(x)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = local_contrast_norm(x,radius = 9)
#        x = local_contrast_norm(x,radius = 7)
        x = x.view(-1, 350*6*6)
        x = F.relu(self.fc1(x))
        x = self.fc_drop(x)
        x = self.fc2(x)
        return F.log_softmax(x)



class LocNet_ST1(nn.Module):
    def __init__(self):
        super(LocNet_ST1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 250, kernel_size=5, padding = 2)
        self.conv2 = nn.Conv2d(in_channels = 250, out_channels = 250, kernel_size=5, padding = 2)
        self.fc1 = nn.Linear(in_features = 250*6*6, out_features = 250)
        self.fc2 = nn.Linear(in_features = 250, out_features = 6)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(in_channels = 3, out_channels = 250, kernel_size=5, padding = 2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(in_channels = 250, out_channels = 250, kernel_size=5, padding = 2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size = 2)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(in_features = 250*6*6, out_features = 250),
            nn.ReLU(True),
            nn.Linear(in_features = 250, out_features = 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 250*6*6)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x


class LocNet_ST2(nn.Module):
    def __init__(self):
        super(LocNet_ST2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 200, out_channels = 150, kernel_size=5, padding = 2)
        self.conv2 = nn.Conv2d(in_channels = 150, out_channels = 200, kernel_size=5, padding = 2)
        self.fc1 = nn.Linear(in_features = 200*2*2, out_features = 300)
        self.fc2 = nn.Linear(in_features = 300, out_features = 6)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(in_channels = 200, out_channels = 150, kernel_size=5, padding = 2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(in_channels = 150, out_channels = 200, kernel_size=5, padding = 2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size = 2)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(in_features = 200*2*2, out_features = 300),
            nn.ReLU(True),
            nn.Linear(in_features = 300, out_features = 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 200*2*2)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x


class LocNet_ST3(nn.Module):
    def __init__(self):
        super(LocNet_ST3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 250, out_channels = 150, kernel_size=5, padding= 2)
        self.conv2 = nn.Conv2d(in_channels = 150, out_channels = 200, kernel_size=5, padding = 2)
        self.fc1 = nn.Linear(in_features = 200*1*1, out_features = 300)
        self.fc2 = nn.Linear(in_features = 300, out_features = 6)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(in_channels = 250, out_channels = 150, kernel_size=5, padding = 2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(in_channels = 150, out_channels = 200, kernel_size=5, padding = 2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size = 2)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(in_features = 200*1*1, out_features = 300),
            nn.ReLU(True),
            nn.Linear(in_features = 300, out_features = 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 200*1*1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
