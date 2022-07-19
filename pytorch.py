import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import datasets, transforms
import os
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import segmentation_models_pytorch as smp


tsr_img = torchvision.io.read_image('data/train/labels/0011.png')
tsr_img.shape

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

batch_size = 16

transform = transforms.Compose([transforms.ToTensor()])

class UlcerData(Dataset):
    def __init__(self, imagePath, maskPath, transforms):
        self.imagePath = imagePath
        self.maskPath = maskPath
        self.transforms = transforms
        self.all_images = os.listdir(imagePath)
        self.all_labels = os.listdir(maskPath)
        
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        img_loc = os.path.join(self.imagePath, self.all_images[idx])
        label_loc = os.path.join(self.maskPath, self.all_labels[idx])
        image = Image.open(img_loc) 
        label = Image.open(label_loc)
        return (self.transforms(image),self.transforms(label))



dataset = UlcerData("data/train/images","data/train/labels",transform)
train_loader = DataLoader(dataset,batch_size=batch_size)
print(len(train_loader))

#for image, label in train_loader:
    #pass
    #feed model with data

ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['foot']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'

# create segmentation model with pretrained encoder
model = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

model = model.to(device)

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])

train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

for i in range(0, 40):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)

