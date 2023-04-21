import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset files: train3000.json, train6000.json, etc.
# each file contains approx. 3000 dictionaries { labels, latent_space }

# labels size: 20
# latent space size: 512

# a json contains:
# { "labels": {"Woman": 0.03828, "Man": 0.96172, "asian": 0.0, "indian": 0.0, "black": 0.0,
#              "white": 0.98292, "middle eastern": 0.01689, "latino hispanic": 0.00019, "Bald": 0.0,
#              "Bangs": 0.10947, "Black_Hair": 0.8164, "Blond_Hair": 0.02662, "Brown_Hair": 0.1449,
#              "Eyeglasses": 1.0, "Gray_Hair": 0.0347, "Mustache": 0.04369, "No_Beard": 0.94032,
#              "Straight_Hair": 0.3993, "Wavy_Hair": 0.49615, "Young": 0.93806},
# "latent_space": [[-1.3095457553863525, 0.8847733736038208, 0.3035028874874115, -2.1871345043182373, ...]]}

files = ['3000', '6000', '9000', '12000', '15000', '18000', '21000']

class LatentSpaceDataset(Dataset):
    def __init__(self, data_path, train=True):
        self.data_path = data_path
        self.data = []
       # self.features = torch.tensor([d['features'] for d in data], dtype=torch.float32)
        #self.latent_spaces = torch.tensor([d['latent_space'] for d in data], dtype=torch.float32)

        if train:
            for nr in files:
                with open(data_path + nr + '.json', 'r') as f:
                    j_data = json.load(f)
                    for _dict in j_data:
                        self.data.append(_dict)
        # for test dataset
        else:
            with open(data_path, 'r') as f:
                j_data = json.load(f)
                for _dict in j_data:
                    self.data.append(_dict)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        features = list(self.data[index]['labels'].values())
        #print(index, features)
        latent_space = self.data[index]['latent_space'][0]
        #print(index, latent_space)

        return torch.tensor(features), torch.tensor(latent_space)


# Define linear regression model
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(LinearRegression, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size*2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size*2, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


def train():
    train_dataset = LatentSpaceDataset('ls_dataset/train')
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)

    model = LinearRegression(input_size=20, output_size=512, hidden_size=128)
    model.to(device)
    criterion = nn.MSELoss()
    #criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    epochs = 5000

    for epoch in range(epochs):
        running_loss = 0.0
        for features, latent_spaces in train_dataloader:
            features = features.to(device)
            latent_spaces = latent_spaces.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, latent_spaces)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch %d Loss: %.3f' % (epoch + 1, running_loss / len(train_dataloader)))

        if (epoch + 1) % 100 == 0:
            checkpoint_path = f"ls_weights/model_epoch_{epoch + 1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch + 1}")

    test(model)


def test(model):
    test_dataset = LatentSpaceDataset('ls_dataset/test.json', train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)
    criterion = nn.MSELoss()
    model.eval()

    total_loss = 0.0
    with torch.no_grad():
        for features, latent_spaces in test_dataloader:
            features = features.to(device)
            latent_spaces = latent_spaces.to(device)
            outputs = model(features)
            loss = criterion(outputs, latent_spaces)
            total_loss += loss.item()

    print('Test Loss: %.3f' % (total_loss / len(test_dataloader)))


def read_json():
    files = ['3000', '6000', '9000', '12000', '15000', '18000', '21000']

    for nr in files:

        with open('ls_dataset/train' + nr + '.json', 'r') as f:
            file1k = json.load(f)


        print(len(file1k))
    with open('ls_dataset/test.json', 'r') as f:
        test_ = json.load(f)
    print(len(test_))


if __name__ == "__main__":
    #read_json()
    train()
