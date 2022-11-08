import torchvision.transforms as transforms

data_dir = "../input/custom/full"
training_csv = "../input/custom/train_data.csv"
testing_csv = "../input/custom/test_data.csv"


train_batch_size = 32
train_number_epochs = 20

# trans = transforms.Compose([transforms.Resize((105,105)),transforms.ToTensor(),
#                             transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

trans = transforms.Compose([transforms.Resize((105,105)),transforms.ToTensor()])