import time
import copy
import math
import torch
from torchnet import meter
from torch.autograd import Variable
from utils import plot_training
# from ignite.engine import Engine
from ignite.metrics import Precision, Recall
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef

data_cat = ['train', 'valid'] # data categories

def train_model(model, criterion, optimizer, dataloaders, scheduler,
                dataset_sizes, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    costs = {x:[] for x in data_cat} # for storing costs per epoch
    accs = {x:[] for x in data_cat} # for storing accuracies per epoch
    print('Train batches:', len(dataloaders['train']))
    print('Valid batches:', len(dataloaders['valid']), '\n')
    for epoch in range(num_epochs):
        confusion_matrix = {x: meter.ConfusionMeter(2, normalized=False)
                            for x in data_cat}
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in data_cat:
            model.train(phase=='train')
            running_loss = 0.0
            running_corrects = 0

            # engine = None
            epoch_precision = Precision(average=False, is_multilabel=False)
            epoch_recall = Recall(average=False, is_multilabel=False)

            overall_labels = []
            overall_preds = []

            # Iterate over data.
            for i, data in enumerate(dataloaders[phase]):
                # get the inputs
                print(i, end='\r')
                inputs = data['images'][0]
                labels = data['label'].type(torch.FloatTensor)

                overall_labels.append(labels)

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                outputs = torch.mean(outputs)
                loss = criterion(outputs, labels, phase)
                running_loss += loss.data[0]
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                preds = (outputs.data > 0.5).type(torch.cuda.FloatTensor)
                preds = preds.view(1)

                if i%99 == 0:
                    print('\nOverall labels:', overall_labels)
                    print('Overall preds:', overall_preds)

                overall_preds.append(preds)

                # engine = Engine((preds, labels))
                epoch_precision.update((preds, labels))
                epoch_recall.update((preds, labels))

                running_corrects += torch.sum(preds == labels.data)
                confusion_matrix[phase].add(preds, labels.data)
            epoch_loss = running_loss.item() / dataset_sizes[phase]
            epoch_acc = running_corrects.item() / dataset_sizes[phase]
            costs[phase].append(epoch_loss)
            accs[phase].append(epoch_acc)

            # epoch_precision = Precision(average=False, is_multilabel=False)
            # epoch_recall = Recall(average=False, is_multilabel=False)
            # epoch_precision.attach(engine, 'precision')
            # epoch_recall.attach(engine, 'recall')

            epoch_f1 = (2*epoch_precision.compute()*epoch_recall.compute())/(epoch_precision.compute()+epoch_recall.compute())
            # matrix_copy = confusion_matrix[phase].value()

            # mcc = (matrix_copy[1][1]*matrix_copy[0][0])/math.sqrt((matrix_copy[0][1]))

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            print('\nScikit {} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'.format(phase, precision_score(overall_labels, overall_preds), recall_score(overall_labels, overall_preds), f1_score(overall_labels, overall_preds)))
            print('Confusion Meter:\n', confusion_matrix[phase].value())
            print('\n{} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'.format(phase, epoch_precision.compute(), epoch_recall.compute(), epoch_f1))

            # deep copy the model
            if phase == 'valid':
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
        time_elapsed = time.time() - since
        print('Time elapsed: {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: {:4f}'.format(best_acc))
    plot_training(costs, accs)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def get_metrics(model, criterion, dataloaders, dataset_sizes, phase='valid'):
    '''
    Loops over phase (train or valid) set to determine acc, loss and
    confusion meter of the model.
    '''
    confusion_matrix = meter.ConfusionMeter(2, normalized=False)
    running_loss = 0.0
    running_corrects = 0

    # engine = None
    precision = Precision(average=False, is_multilabel=False)
    recall = Recall(average=False, is_multilabel=False)

    overall_labels = []
    overall_preds = []

    for i, data in enumerate(dataloaders[phase]):
        print(i, end='\r')
        labels = data['label'].type(torch.FloatTensor)
        inputs = data['images'][0]

        overall_labels.append(labels)

        # wrap them in Variable
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        # forward
        outputs = model(inputs)
        outputs = torch.mean(outputs)
        loss = criterion(outputs, labels, phase)
        # statistics
        running_loss += loss.data[0] * inputs.size(0)
        preds = (outputs.data > 0.5).type(torch.cuda.FloatTensor)
        preds = preds.view(1)

        if i%99 == 0:
            print('\nOverall labels:', overall_labels)
            print('Overall preds:', overall_preds)

        overall_preds.append(preds)

        # engine = Engine((preds, labels))
        precision.update((preds, labels))
        recall.update((preds, labels))

        running_corrects += torch.sum(preds == labels.data)
        confusion_matrix.add(preds, labels.data)

    loss = running_loss.item() / dataset_sizes[phase]
    acc = running_corrects.item() / dataset_sizes[phase]

    # precision = Precision(average=False, is_multilabel=False)
    # recall = Recall(average=False, is_multilabel=False)
    # precision.attach(engine, 'precision')
    # recall.attach(engine, 'recall')

    f1 = (2*precision.compute()*recall.compute())/(precision.compute()+recall.compute())

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, loss, acc))
    print('\nScikit {} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'.format(phase, precision_score(overall_labels, overall_preds), recall_score(overall_labels, overall_preds), f1_score(overall_labels, overall_preds)))
    print('Confusion Meter:\n', confusion_matrix.value())
    print('\n{} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'.format(phase, precision.compute(), recall.compute(), f1))
