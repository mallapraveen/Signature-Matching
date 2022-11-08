from torch import rand
import config,os,random
from siamese_network import SiameseNetworkDataset
from predict import predict

from train import train
from inference import inference_test
from utilities import visualize_dataset
import torchvision.transforms as transforms

if __name__ == '__main__':
    
    siamese_dataset = SiameseNetworkDataset(config.training_csv,config.data_dir,
                                            transform=transforms.Compose([transforms.Resize((105,105)),
                                                                        transforms.ToTensor()
                                                                        ]))
    
    visualize_dataset(siamese_dataset)
    
    # train()
    # inference_test()
    
    path = '../Old Code/s1'
    
    fold = random.choice(os.listdir(path))
    
    org_fold = os.listdir(os.path.join(path,str(fold),'o'))
    
    forg_fold = os.listdir(os.path.join(path,str(fold),'f'))
    
    org = os.path.join(path,str(fold), 'o', random.choice(org_fold))
    
    forg = os.path.join(path,str(fold),'f', random.choice(forg_fold))
    
    predict(org,forg)