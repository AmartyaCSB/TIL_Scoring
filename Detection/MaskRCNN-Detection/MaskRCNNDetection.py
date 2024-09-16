import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from pycocotools.coco import COCO
import numpy as np
import cv2
from matplotlib import pyplot as plt


def imshow(img):
    """function to show an image"""
    img = img/2 + 0.5
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    plt.show()
    
def display_data_sample(img, target):
    """function to display one data point"""
    # display the image
    imshow(img)
    print()
    # check for annotations
    if len(target['boxes']):
        # add all the masks and display
        mask = np.sum(target['masks'].numpy(), axis=0)
        mask[mask>255] = 255
        plt.imshow(mask, cmap='gray', vmin=0, vmax=1)
        plt.show()
    else:
        print('no annotations')

class TILDetectionDataset(Dataset):
    """COCO Detection format dataset for TIL detection"""
    
    def __init__(self, root, annFile):
        self.root = root
        self.annFile = annFile
        self.coco = COCO(annFile)
    
    def __len__(self):
        """return the number of images"""
        return len(self.coco.imgs)
    
    def __getitem__(self, idx):
        """
            returns (img, target)
            img    : normalized image tensor
            target : dict with keys :
                            boxes (FloatTensor[N, 4])
                            labels (Int64Tensor[N])
                            masks (UInt8Tensor[N, H, W])
        """
        
        # read the image
        filename = self.root + self.coco.imgs[idx]['file_name']
        img = torchvision.io.read_image(filename)
        img = torch.as_tensor(img, dtype=torch.float)
        img = img/255
        _, H,W = img.shape
        # normalize the image
        #img = torchvision.transforms.functional.normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # get the annotations in img
        annIds = self.coco.getAnnIds(imgIds=idx)
        anns = self.coco.loadAnns(annIds)
        # generate boxes
        boxes = []
        for ann in anns:
            xmin = ann['bbox'][0]
            ymin = ann['bbox'][1]
            xmax = xmin + ann['bbox'][2]
            ymax = ymin + ann['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # generate labels
        labels = torch.ones((len(anns)), dtype=torch.int64)
        # create blank image for mask
        masks = np.zeros((len(anns), H, W), dtype=np.uint8)
        # generate masks
        for idx, ann in enumerate(anns):
            xmin = ann['bbox'][0]
            ymin = ann['bbox'][1]
            xmax = xmin + ann['bbox'][2]
            ymax = ymin + ann['bbox'][3]
            masks[idx] = cv2.rectangle(masks[idx], (xmin, ymin), (xmax, ymax), 255,-1)
            masks[idx] = masks[idx]/255
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        # craete dictionary
        retDict = {
            'boxes':boxes,
            'labels':labels,
            'masks':masks}
        return (img, retDict)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# create dataset
dataset = TILDetectionDataset('/home/krishna/Documents/mip_project/data/wsirois/roi-level-annotations/tissue-cells/', '/home/krishna/Documents/mip_project/data/wsirois/roi-level-annotations/tissue-cells/tiger-coco.json')

# create train and test split from the data
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# create the model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, num_classes=2, pretrained_backbone=False)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr= 1e-4, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=2, verbose=True)

train_losses = []
test_losses = []

for epoch in range(200):  # loop over the dataset multiple times
    
    running_train_loss = 0.0
    running_test_loss = 0.0

    with tqdm(train_dataset, unit="batch") as tepoch:
        # train
        for i, (img, target) in enumerate(tepoch, 0):
            tepoch.set_description(f"Epoch {epoch}")
            
            # get the inputs; data is a list of [inputs, labels]
            if len(target['boxes']) > 0 and len(target['boxes']) < 500:
                target_gpu = {}
                for key in target.keys():
                    target_gpu[key] = target[key].to(device)
                
                img = img.to(device)
                
                target_gpu = [target_gpu]
                img = [img]
            
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward
                outputs = model(img, target_gpu)
                loss = outputs['loss_classifier'] + outputs['loss_box_reg'] + outputs['loss_mask']
                loss.backward()
                
                # save the loss value
                loss_val = float(loss.detach().cpu().item())
                running_train_loss += loss_val
                tepoch.set_postfix(loss=loss_val)
                
                # optimizer step
                optimizer.step()
                
                # memory management
                for im in img:
                    del im
                    torch.cuda.empty_cache()
                
                for t in target_gpu:
                    del target_gpu
                    torch.cuda.empty_cache()
                
                del loss
                torch.cuda.empty_cache()
                
                del outputs
                torch.cuda.empty_cache()
                
                del target
                torch.cuda.empty_cache()
        
    # test
    with torch.no_grad():
        for i, (img, target) in enumerate(test_dataset, 0):
            if len(target['boxes']) > 0 and len(target['boxes']) < 500:
                
                target_gpu = {}
                for key in target.keys():
                    target_gpu[key] = target[key].to(device)
                
                img = img.to(device)
                
                target_gpu = [target_gpu]
                img = [img]
                
                outputs = model(img, target_gpu)
                
                loss = outputs['loss_classifier'] + outputs['loss_box_reg'] + outputs['loss_mask']
                
                loss_val = float(loss.detach().cpu().item())
                running_test_loss += loss_val
                
                # memory management
                for im in img:
                    del im
                    torch.cuda.empty_cache()
                
                for t in target_gpu:
                    del target_gpu
                    torch.cuda.empty_cache()
                
                del loss
                torch.cuda.empty_cache()
                
                del outputs
                torch.cuda.empty_cache()
                
                del target
                torch.cuda.empty_cache()
                
    # compute epoch losses
    train_losses.append(running_train_loss/(len(train_dataset)))
    test_loss = running_test_loss/(len(test_dataset))
    test_losses.append(test_loss)
    # LR scheduler step
    scheduler.step(test_loss)
    
    # save checkpoint weights every 50 epochs 
    if (epoch % 20 == 0):
        torch.save(model.state_dict(), '/home/krishna/Documents/mip_project/code/model_weights/detection_' + str(epoch) + '.pth')
        
    
torch.save(model.state_dict(), '/home/krishna/Documents/mip_project/code/model_weights/detection_final.pth')
np.save('/home/krishna/Documents/mip_project/code/model_weights/train_losses.npy', np.array(train_losses))
np.save('/home/krishna/Documents/mip_project/code/model_weights/test_losses.npy', np.array(test_losses))
