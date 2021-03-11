import argparse, time, copy, torch
import pandas as pd
from torch.autograd import Variable
from densenet import densenet169
from utils import plot_training, n_p, get_count
from train import train_model, get_metrics
from pipeline import get_study_level_data, get_all_data, get_dataloaders

parser = argparse.ArgumentParser(description='Train a deep learner to determine whether an abnormality exists in an upper or lower extremity x-ray')
parser.add_argument('study_type', help='study type x-rays to train the model on',
                    choices=['mura', 'lera', 'elbow', 'finger', 'forearm', 'hand', 'humerus', 'shoulder', 'wrist', 'ankle', 'foot', 'hip', 'knee'])
parser.add_argument('-t', '--transfer', help='use transfer learning using saved model', type=str)
args = parser.parse_args()

if args.study_type == 'mura':
    print('Training using all MURA study types')
    # load all MURA dict data
    study_data = get_all_data(dataset='MURA-v1.1')
elif args.study_type == 'lera':
    print('Training using all LERA study types')
    # load all LERA dict data
    study_data = get_all_data(dataset='LERA')
else:
    print('Training using study type', args.study_type)
    # #### load study level dict data specified in argument
    study_data = get_study_level_data(study_type='XR_' + args.study_type.upper())

# #### Create dataloaders pipeline
data_cat = ['train', 'valid'] # data categories
dataloaders = get_dataloaders(study_data, batch_size=1)
dataset_sizes = {x: len(study_data[x]) for x in data_cat}

# #### Build model
# tai = total abnormal images, tni = total normal images
tai = {x: get_count(study_data[x], 'positive') for x in data_cat}
tni = {x: get_count(study_data[x], 'negative') for x in data_cat}
Wt1 = {x: n_p(tni[x] / (tni[x] + tai[x])) for x in data_cat}
Wt0 = {x: n_p(tai[x] / (tni[x] + tai[x])) for x in data_cat}

print('tai:', tai)
print('tni:', tni, '\n')
print('Wt0 train:', Wt0['train'])
print('Wt0 valid:', Wt0['valid'])
print('Wt1 train:', Wt1['train'])
print('Wt1 valid:', Wt1['valid'])

class Loss(torch.nn.modules.Module):
    def __init__(self, Wt1, Wt0):
        super(Loss, self).__init__()
        self.Wt1 = Wt1
        self.Wt0 = Wt0

    def forward(self, inputs, targets, phase):
        loss = - (self.Wt1[phase] * targets * inputs.log() + self.Wt0[phase] * (1 - targets) * (1 - inputs).log())
        return loss

if args.transfer:
    model = densenet169()
    model.load_state_dict(torch.load('models/' + args.transfer), strict=False)
else:
    model = densenet169(pretrained=True)

model = model.cuda()

criterion = Loss(Wt1, Wt0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, verbose=True)

# if __name__ == '__main__':
# #### Train model
model = train_model(model, criterion, optimizer, dataloaders, scheduler, dataset_sizes, num_epochs=5)

torch.save(model.state_dict(), 'models/model.pt')

get_metrics(model, criterion, dataloaders, dataset_sizes)
