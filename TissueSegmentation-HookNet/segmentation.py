from wholeslidedata.iterators import create_batch_iterator
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from multiprocessing import process
import torch.optim as optim
import sys
import cv2
import albumentations as A

# function to perform center crop for concatenating skip connections and the hook connection
def center_crop(x, size):
    transform = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(size=size)])
    return transform(x)

# define the conv-batchnorm-relu-conv-batchnorm-relu block
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.conv(x)

# define the HookeNet model
class HookNet(nn.Module):
    def __init__(self, in_channels, out_channels, hook_crop) -> None:
        
        super(HookNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2)
        self.preskips_context = nn.ModuleList()
        self.preskips_target = nn.ModuleList()
        self.hook_crop = hook_crop
        
        # channel sizes for DoubleConv blocks
        features=[32, 64, 128, 256]
        
        # encoder
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # bottleneck
        self.bottleneck = DoubleConv(256, 320)
        
        # decoder
        for feature in reversed(features):
            self.decoder.append(DoubleConv(feature*2, feature))
        
        self.preskips_context.append(nn.Conv2d(320, 256, 3))
        self.preskips_context.append(nn.Conv2d(256, 128, 3))
        self.preskips_context.append(nn.Conv2d(128, 64, 3))
        self.preskips_context.append(nn.Conv2d(64, 32, 3))
        
        self.preskips_target.append(nn.Conv2d(352, 256, 3))
        self.preskips_target.append(nn.Conv2d(256, 128, 3))
        self.preskips_target.append(nn.Conv2d(128, 64, 3))
        self.preskips_target.append(nn.Conv2d(64, 32, 3))
        
        # final conv
        self.finalconv = nn.Conv2d(32, out_channels, 1)
        
    def context_forward(self, x):
        
        # store skip connection inputs
        skip_connections = []
        
        # apply encoder DoubleConvs
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.maxpool(x)
        
        # reverse the skip connections
        skip_connections.reverse()
        
        # apply bottleneck
        x = self.bottleneck(x)
        
        # apply decoder DoubleConvs
        for decode, preskipconv, skip in zip(self.decoder, self.preskips_context, skip_connections):
            x = self.upsample(x)
            x = preskipconv(x)
            x = F.relu(torch.cat((center_crop(skip, (x.shape[2], x.shape[2])), x), dim=1))
            x = decode(x)
        
        # hook out to target branch
        hook = center_crop(x, (self.hook_crop, self.hook_crop))
        
        # final conv
        x = self.finalconv(x)
        
        # return x and hool
        return (hook, x)
    
    def target_forward(self, hook, x):
        
        # store skip connection inputs
        skip_connections = []
        
        # apply encoder DoubleConvs
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.maxpool(x)
        
        # reverse the skip connections
        skip_connections.reverse()
        
        # apply bottleneck
        x = self.bottleneck(x)
        
        # hook in from context branch
        x = torch.cat((hook, x), dim=1)
        
        # apply decoder DoubleConvs
        for decode, preskipconv, skip in zip(self.decoder, self.preskips_target, skip_connections):
            x = self.upsample(x)
            x = preskipconv(x)
            x = F.relu(torch.cat((center_crop(skip, (x.shape[2], x.shape[2])), x), dim=1))
            x = decode(x)
        
        # final conv
        x = self.finalconv(x)
        
        # return x
        return x
    
    def forward(self, context_input, target_input):
        
        # forward pass of context branch
        hook, context_output = self.context_forward(context_input)
        # forward pass of target branch
        target_output = self.target_forward(hook, target_input)
        # return the outputs
        return (context_output, target_output)

def dice_loss(output, target):

    eps = 0.0001

    intersection = output * target
    numerator = 2 * intersection.sum(0).sum(1).sum(1)
    denominator = output + target
    denominator = denominator.sum(0).sum(1).sum(1) + eps
    loss_per_channel = (1 - (numerator / denominator))

    return loss_per_channel.sum() / output.size(1)

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE


# number of batches
batches = 100
# batch iterator config
user_config = '/home/amandal36/Documents/TIL_Scoring/TissueSegmentation-HookNet/segmentation/user_config_seg.yml'
# number of cpus used to extract patches on multiple cores
cpus = 1
# create iterators for the dataset
training_iterator = create_batch_iterator(user_config=user_config, mode='training', cpus=cpus, buffer_dtype=np.uint8, number_of_batches=batches)
test_iterator = create_batch_iterator(user_config=user_config, mode='inference', cpus=cpus, buffer_dtype=np.uint8, number_of_batches=batches)    

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

aug_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5)
], additional_targets={'target_in' : 'image',
                      'context_in' : 'image',
                      'target_mask' : 'mask',
                      'context_mask' : 'mask'
                      })

"""
transformed = aug_transforms(image=np.zeros((10, 10, 3), dtype=np.uint8),
                                         target_in=x_batch[0,0,:,:,:],
                                         context_in=x_batch[0,1,:,:,:],
                                         target_mask=y_batch[0,0,:,:,:],
                                         context_mask=y_batch[0,1,:,:,:]
                                        )
"""

"""
plt.imshow(x_batch[0,1,:,:,:])
plt.show()
fig, axes = plt.subplots(1, 7)
for i in range(7):
    axes[i].imshow(y_batch[0,1,:,:,i], cmap='gray', vmax=1, vmin=0)
plt.show()

plt.imshow(transformed['context_in'][:,:,:])
plt.show()
fig, axes = plt.subplots(1, 7)
for i in range(7):
    axes[i].imshow(transformed['context_mask'][:,:,i], cmap='gray', vmax=1, vmin=0)
plt.show()
"""

#def hooknet parameters for training
net = HookNet(3, 7, 10)
#net.load_state_dict(torch.load('/home/amandal36/Documents/TIL_Scoring/TissueSegmentation-HookNet/segmentation//weights/weights_1000.pth'))
net.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr= 5e-6, weight_decay=0.001)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, cooldown=4, min_lr=1e-8, verbose=True)

train_losses = []
test_losses = []

textfile = open("train_dice_loss.txt", "w")
textfile.close()
textfile = open("test_dice_loss.txt", "w")
textfile.close()

for epoch in range(1000):
    
    running_train_loss = 0.0
    running_test_loss = 0.0
    
    with tqdm(training_iterator, unit="batch") as tepoch:
        for i, data in enumerate(tepoch, 0):
            tepoch.set_description(f"Epoch {epoch}")
            # normalize the data
            x_batch, y_batch = data[0], data[1]
            info = data[2]
        
            # add augmentations
            for batch in range(2):
                transformed = aug_transforms(image=np.zeros((10, 10, 3), dtype=np.uint8),
                                     target_in=x_batch[batch,0,:,:,:],
                                     context_in=x_batch[batch, 1,:,:,:],
                                     target_mask=y_batch[batch,0,:,:,:],
                                     context_mask=y_batch[batch,1,:,:,:]
                                    )
        
                x_batch[batch,0,:,:,:] = transformed['target_in']
                x_batch[batch,1,:,:,:] = transformed['context_in']
                y_batch[batch,0,:,:,:] = transformed['target_mask']
                y_batch[batch,1,:,:,:] = transformed['context_mask']
        
            # convert images to tensors
            x_batch = torch.from_numpy(x_batch).type(torch.FloatTensor).permute((0,1,4,2,3)).to(device)/255
            y_batch = torch.from_numpy(y_batch).type(torch.FloatTensor).permute((0,1,4,2,3)).to(device)
        
            # zero the parameter gradients
            optimizer.zero_grad()
        
            # forward + backwards + optimize
            context_out, target_out = net(x_batch[:,1,:,:,:].squeeze(), x_batch[:,0,:,:,:].squeeze())

            lossFunction = DiceBCELoss()

            target_loss_invasive_tumor = lossFunction(target_out[:,0,:,:].squeeze(), y_batch[:,0,0,:,:].squeeze())
            context_loss_invasive_tumor = lossFunction(context_out[:,0,:,:].squeeze(), y_batch[:,1,0,:,:].squeeze())
            loss_invasive_tumor = 0.75 * target_loss_invasive_tumor + 0.25 * context_loss_invasive_tumor

            target_loss_tumor_stroma = lossFunction(target_out[:,1,:,:].squeeze(), y_batch[:,0,1,:,:].squeeze())
            context_loss_tumor_stroma = lossFunction(context_out[:,1,:,:].squeeze(), y_batch[:,1,1,:,:].squeeze())
            loss_tumor_stroma = 0.75 * target_loss_tumor_stroma + 0.25 * context_loss_tumor_stroma

            target_loss_in_situ_tumor = lossFunction(target_out[:,2,:,:].squeeze(), y_batch[:,0,2,:,:].squeeze())
            context_loss_in_situ_tumor = lossFunction(context_out[:,2,:,:].squeeze(), y_batch[:,1,2,:,:].squeeze())
            loss_in_situ_tumor = 0.75 * target_loss_in_situ_tumor + 0.25 * context_loss_in_situ_tumor

            target_loss_healthy = lossFunction(target_out[:,3,:,:].squeeze(), y_batch[:,0,3,:,:].squeeze())
            context_loss_healthy = lossFunction(context_out[:,3,:,:].squeeze(), y_batch[:,1,3,:,:].squeeze())
            loss_healthy = 0.75 * target_loss_healthy + 0.25 * context_loss_healthy

            target_loss_necrosis = lossFunction(target_out[:,4,:,:].squeeze(), y_batch[:,0,4,:,:].squeeze())
            context_loss_necrosis = lossFunction(context_out[:,4,:,:].squeeze(), y_batch[:,1,4,:,:].squeeze())
            loss_necrosis = 0.75 * target_loss_necrosis + 0.25 * context_loss_necrosis

            target_loss_inflamed = lossFunction(target_out[:,5,:,:].squeeze(), y_batch[:,0,5,:,:].squeeze())
            context_loss_inflamed = lossFunction(context_out[:,5,:,:].squeeze(), y_batch[:,1,5,:,:].squeeze())
            loss_inflamed = 0.75 * target_loss_inflamed + 0.25 * context_loss_inflamed

            target_loss_rest = lossFunction(target_out[:,6,:,:].squeeze(), y_batch[:,0,6,:,:].squeeze())
            context_loss_rest = lossFunction(context_out[:,6,:,:].squeeze(), y_batch[:,1,6,:,:].squeeze())
            loss_rest = 0.75 * target_loss_rest + 0.25 * context_loss_rest

            #target_loss = dice_loss(F.softmax(target_out, dim=1), y_batch[:,0,:,:,:].squeeze())
            #context_loss = dice_loss(F.softmax(context_out, dim=1), y_batch[:,1,:,:,:].squeeze())

            # 0.3 + 0.3 + 0.2 + 0.05 + 0.05 + 0.05 + 0.05
            loss = 0.3 * loss_invasive_tumor + 0.3 * loss_tumor_stroma + 0.2 * loss_in_situ_tumor + 0.2 * loss_healthy + 0.05 * loss_necrosis + 0.05 * loss_inflamed + 0.05 *loss_rest
            loss.backward()
            optimizer.step()
        
            # stat tracking
            loss_val = loss.detach().cpu().item()
            running_train_loss += loss_val
            tepoch.set_postfix(loss=loss_val)
            #tepoch.set_postfix({'loss':str(loss_val), 'file1':info['sample_references'][0]['reference'].file_key, 'file2':info['sample_references'][1]['reference'].file_key})
        
            # cleanup
            del x_batch
            del y_batch
            del loss
            del target_out
            del context_out
            del target_loss_invasive_tumor
            del context_loss_invasive_tumor
            del loss_invasive_tumor
            del target_loss_tumor_stroma
            del context_loss_tumor_stroma
            del loss_tumor_stroma
            del target_loss_in_situ_tumor
            del context_loss_in_situ_tumor
            del loss_in_situ_tumor
            del target_loss_healthy
            del context_loss_healthy
            del loss_healthy
            del target_loss_necrosis
            del context_loss_necrosis
            del loss_necrosis
            del target_loss_inflamed
            del context_loss_inflamed
            del loss_inflamed
            del target_loss_rest
            del context_loss_rest
            del loss_rest
            
            torch.cuda.empty_cache()
        
    with torch.no_grad():
        for i, data in enumerate(test_iterator, 0):
            x_batch, y_batch = data[0], data[1]
            info = data[2]
            # convert images to tensors
            x_batch = torch.from_numpy(x_batch).type(torch.FloatTensor).permute((0,1,4,2,3)).to(device)/255
            y_batch = torch.from_numpy(y_batch).type(torch.FloatTensor).permute((0,1,4,2,3)).to(device)
    
            # forward + backwards + optimize
            context_out, target_out = net(x_batch[:,1,:,:,:].squeeze(), x_batch[:,0,:,:,:].squeeze())

            lossFunction = DiceBCELoss()

            target_loss_invasive_tumor = lossFunction(target_out[:,0,:,:].squeeze(), y_batch[:,0,0,:,:].squeeze())
            context_loss_invasive_tumor = lossFunction(context_out[:,0,:,:].squeeze(), y_batch[:,1,0,:,:].squeeze())
            loss_invasive_tumor = 0.75 * target_loss_invasive_tumor + 0.25 * context_loss_invasive_tumor

            target_loss_tumor_stroma = lossFunction(target_out[:,1,:,:].squeeze(), y_batch[:,0,1,:,:].squeeze())
            context_loss_tumor_stroma = lossFunction(context_out[:,1,:,:].squeeze(), y_batch[:,1,1,:,:].squeeze())
            loss_tumor_stroma = 0.75 * target_loss_tumor_stroma + 0.25 * context_loss_tumor_stroma

            target_loss_in_situ_tumor = lossFunction(target_out[:,2,:,:].squeeze(), y_batch[:,0,2,:,:].squeeze())
            context_loss_in_situ_tumor = lossFunction(context_out[:,2,:,:].squeeze(), y_batch[:,1,2,:,:].squeeze())
            loss_in_situ_tumor = 0.75 * target_loss_in_situ_tumor + 0.25 * context_loss_in_situ_tumor

            target_loss_healthy = lossFunction(target_out[:,3,:,:].squeeze(), y_batch[:,0,3,:,:].squeeze())
            context_loss_healthy = lossFunction(context_out[:,3,:,:].squeeze(), y_batch[:,1,3,:,:].squeeze())
            loss_healthy = 0.75 * target_loss_healthy + 0.25 * context_loss_healthy

            target_loss_necrosis = lossFunction(target_out[:,4,:,:].squeeze(), y_batch[:,0,4,:,:].squeeze())
            context_loss_necrosis = lossFunction(context_out[:,4,:,:].squeeze(), y_batch[:,1,4,:,:].squeeze())
            loss_necrosis = 0.75 * target_loss_necrosis + 0.25 * context_loss_necrosis

            target_loss_inflamed = lossFunction(target_out[:,5,:,:].squeeze(), y_batch[:,0,5,:,:].squeeze())
            context_loss_inflamed = lossFunction(context_out[:,5,:,:].squeeze(), y_batch[:,1,5,:,:].squeeze())
            loss_inflamed = 0.75 * target_loss_inflamed + 0.25 * context_loss_inflamed

            target_loss_rest = lossFunction(target_out[:,6,:,:].squeeze(), y_batch[:,0,6,:,:].squeeze())
            context_loss_rest = lossFunction(context_out[:,6,:,:].squeeze(), y_batch[:,1,6,:,:].squeeze())
            loss_rest = 0.75 * target_loss_rest + 0.25 * context_loss_rest
    
            loss = 0.3 * loss_invasive_tumor + 0.3 * loss_tumor_stroma + 0.2 * loss_in_situ_tumor + 0.2 * loss_healthy + 0.05 * loss_necrosis + 0.05 * loss_inflamed + 0.05 *loss_rest
            
            # stat tracking
            loss_val = loss.detach().cpu().item()
            running_test_loss += loss_val
    
            #tepoch.set_postfix({'loss':str(loss_val), 'file1':info['sample_references'][0]['reference'].file_key, 'file2':info['sample_references'][1]['reference'].file_key})
    
            # cleanup
            del x_batch
            del y_batch
            del loss
            del target_out
            del context_out
            del target_loss_invasive_tumor
            del context_loss_invasive_tumor
            del loss_invasive_tumor
            del target_loss_tumor_stroma
            del context_loss_tumor_stroma
            del loss_tumor_stroma
            del target_loss_in_situ_tumor
            del context_loss_in_situ_tumor
            del loss_in_situ_tumor
            del target_loss_healthy
            del context_loss_healthy
            del loss_healthy
            del target_loss_necrosis
            del context_loss_necrosis
            del loss_necrosis
            del target_loss_inflamed
            del context_loss_inflamed
            del loss_inflamed
            del target_loss_rest
            del context_loss_rest
            del loss_rest
    
    textfile = open("train_dice_loss.txt", "a")
    textfile.write(str(running_train_loss/len(training_iterator)) + "\n")
    textfile.close()

    train_losses.append(running_train_loss/len(training_iterator))

    textfile = open("test_dice_loss.txt", "a")
    textfile.write(str(running_test_loss/len(test_iterator)) + "\n")
    textfile.close()

    test_losses.append(running_test_loss/len(test_iterator))
    print(train_losses[-1])
    print(test_losses[-1])
    # LR scheduler step
    #scheduler.step(loss_val)
    # save checkpoint weights every 20 epochs 
    if ((epoch + 1) % 20 == 0):
        torch.save(net.state_dict(), '/home/amandal36/Documents/TIL_Scoring/TissueSegmentation-HookNet/segmentation/weights_new_new/weights_' + str(epoch+1) + '.pth')