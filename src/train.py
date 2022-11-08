import config

import torch
from torch import optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from siamese_network import SiameseNetworkDataset,SiameseNetwork
from loss import ContrastiveLoss

def fit(model,train_dataloader,criterion,optimizer):
    model.train()
    counter = []
    loss_history = [] 
    iteration_number= 0
    
    for epoch in range(0,config.train_number_epochs):
        for i, data in enumerate(train_dataloader,0):
            img0, img1 , label = data
            # img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
            optimizer.zero_grad()
            output1,output2 = model(img0,img1)
            loss_contrastive = criterion(output1,output2,label)
            loss_contrastive.backward()
            optimizer.step()
            if i %50 == 0 :
                print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
                iteration_number +=10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
    return model

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the the dataset from raw image folders
    siamese_dataset = SiameseNetworkDataset(config.training_csv,config.data_dir,
                                            transform=config.trans)
    
    # Load the dataset as pytorch tensors using dataloader
    train_dataloader = DataLoader(siamese_dataset,
                            shuffle=True,
                            num_workers=8,
                            batch_size=config.train_batch_size)
    
    # Declare Siamese Network
    model = SiameseNetwork()#.cuda()
    # Decalre Loss Function
    criterion = ContrastiveLoss()
    # Declare Optimizer
    optimizer = optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.99, eps=1e-8, weight_decay=0.0005, momentum=0.9)
    
    # Train the model
    model = fit(model,train_dataloader,criterion,optimizer)
    torch.save(model.state_dict(), "../output/model.pt")
    print("Model Saved Successfully")