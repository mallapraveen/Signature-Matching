import config
import torch,torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from siamese_network import SiameseNetwork,SiameseNetworkDataset
from utilities import imshow

def inference_test():
    # Load the saved model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load("../output/model.pt",map_location=torch.device('cpu')))
    model.eval()
    # Load the test dataset
    test_dataset = SiameseNetworkDataset(training_csv=config.testing_csv,training_dir=config.data_dir,
                                            transform=config.trans)

    test_dataloader = DataLoader(test_dataset,num_workers=6,batch_size=1,shuffle=True)
    
    # Print the sample outputs to view its dissimilarity
    counter=0
    list_0 = torch.FloatTensor([[1]])
    list_1 = torch.FloatTensor([[0]])
    for i, data in enumerate(test_dataloader,0): 
        x0, x1 , label = data
        concatenated = torch.cat((x0,x1),0)
        output1,output2 = model(x0.to(device),x1.to(device))
        eucledian_distance = F.pairwise_distance(output1, output2)
        if label==list_0:
            label="Orginial"
        else:
            label="Forged"
        imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f} Label: {}'.format(eucledian_distance.item(),label))
        counter=counter+1
        break
        if counter ==20:
            break