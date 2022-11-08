import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import torchvision,torch
from torch.utils.data import DataLoader

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()
    

def stratified_split():
    data = pd.read_csv('../Signature Matching/input/custom/data.csv').sample(frac=1)
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,[0,1]],data.iloc[:,[2]],test_size=0.1,stratify=data.iloc[:,[2]])
    pd.concat([X_train,y_train],axis=1).to_csv('../Signature Matching/input/custom/train_data.csv',index=False)
    pd.concat([X_test,y_test],axis=1).to_csv('../Signature Matching/input/custom/test_data.csv',index=False)
    

def visualize_dataset(dataset):
    vis_dataloader = DataLoader(dataset,
                        shuffle=True,
                        batch_size=8)
    dataiter = iter(vis_dataloader)


    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0],example_batch[1]),0)
    print(example_batch[0].size())
    imshow(torchvision.utils.make_grid(concatenated))
    print(example_batch[2].numpy())

