from PIL import Image

import torch,torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import config
from siamese_network import SiameseNetwork
from utilities import imshow

def predict(sig1,sig2):
    sig1 = Image.open(sig1)
    sig2 = Image.open(sig2)
    sig1 = sig1.convert("L")
    sig2 = sig2.convert("L")
    sig1 = config.trans(sig1)
    sig2 = config.trans(sig2)
    sig1 = sig1.unsqueeze(0)
    sig2 = sig2.unsqueeze(0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load("../output/model_margin_2.pt",map_location=torch.device('cpu')))
    output1,output2 = model(sig1.to(device),sig2.to(device))
    eucledian_distance = F.pairwise_distance(output1, output2)
    
    concatenated = torch.cat((sig1,sig2),0)
    imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(eucledian_distance.item()))