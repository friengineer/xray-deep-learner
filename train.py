import time, copy, math, torch
from torchnet import meter
from torch.autograd import Variable
from utils import plot_training
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score

data_cat = ['train', 'valid']

# train the model
def train_model(model, criterion, optimizer, dataloaders, scheduler,
                dataset_sizes, num_epochs):
    """Train the model using the given parameters."""
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # store costs and accuracies per epoch
    costs = {x:[] for x in data_cat}
    accs = {x:[] for x in data_cat}
    print('Train batches:', len(dataloaders['train']))
    print('Valid batches:', len(dataloaders['valid']), '\n')

    for epoch in range(num_epochs):
        confusion_matrix = {x: meter.ConfusionMeter(2, normalized=False)
                            for x in data_cat}
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # each epoch has a training and validation phase
        for phase in data_cat:
            model.train(phase=='train')
            running_loss = 0.0
            running_corrects = 0
            # store ground truth and predicted labels
            overall_labels = []
            overall_preds = []

            # iterate over data
            for i, data in enumerate(dataloaders[phase]):
                # get the inputs
                print(i, end='\r')
                inputs = data['images'][0]
                labels = data['label'].type(torch.FloatTensor)

                overall_labels.append(labels.item())

                # send inputs to GPU and wrap them in Variable
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)

                outputs = torch.mean(outputs)
                loss = criterion(outputs, labels, phase)
                running_loss += loss.data[0]

                # backward and optimise only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                preds = (outputs.data > 0.5).type(torch.cuda.FloatTensor)
                preds = preds.view(1)

                overall_preds.append(preds.item())

                running_corrects += torch.sum(preds == labels.data)
                confusion_matrix[phase].add(preds, labels.data)

            # calcualte epoch metrics
            epoch_loss = running_loss.item() / dataset_sizes[phase]
            epoch_acc = running_corrects.item() / dataset_sizes[phase]
            costs[phase].append(epoch_loss)
            accs[phase].append(epoch_acc)

            print('\nConfusion Meter:\n', confusion_matrix[phase].value())
            print('\n{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('\n{} Precision: {:.4f} Recall: {:.4f}'.format(phase,
                                                          precision_score(overall_labels, overall_preds),
                                                          recall_score(overall_labels, overall_preds)))
            print('{} F1: {:.4f} MCC: {:.4f}'.format(phase, f1_score(overall_labels, overall_preds),
                                              matthews_corrcoef(overall_labels, overall_preds)))
            print('{} AUROC: {:.4f}\n'.format(phase, roc_auc_score(overall_labels, overall_preds)))

            # deep copy the model if it has the best accuracy
            if phase == 'valid':
                scheduler.step(epoch_loss)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        # calculate epoch training time
        time_elapsed = time.time() - since
        print('Time elapsed: {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
        print()

    # calculate total training time
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: {:4f}'.format(best_acc))
    # plot graph of training and validation costs and accuracies
    plot_training(costs, accs)
    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


def get_metrics(model, criterion, dataloaders, dataset_sizes, phase='valid'):
    """
    Loops over phase (train or valid) set to determine accuracy, loss and
    confusion meter of the model.
    """
    confusion_matrix = meter.ConfusionMeter(2, normalized=False)
    running_loss = 0.0
    running_corrects = 0
    # store ground truth and predicted labels
    overall_labels = []
    overall_preds = []

    # iterate over data
    for i, data in enumerate(dataloaders[phase]):
        # get the inputs
        print(i, end='\r')
        labels = data['label'].type(torch.FloatTensor)
        inputs = data['images'][0]

        overall_labels.append(labels.item())

        # send inputs to GPU and wrap them in Variable
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

        overall_preds.append(preds.item())

        running_corrects += torch.sum(preds == labels.data)
        confusion_matrix.add(preds, labels.data)

    # caculate metrics
    loss = running_loss.item() / dataset_sizes[phase]
    acc = running_corrects.item() / dataset_sizes[phase]

    print('\nConfusion Meter:\n', confusion_matrix.value())
    print('\n{} Loss: {:.4f} Acc: {:.4f}'.format(phase, loss, acc))
    print('\n{} Precision: {:.4f} Recall: {:.4f}'.format(phase, precision_score(overall_labels, overall_preds),
                                                  recall_score(overall_labels, overall_preds)))
    print('{} F1: {:.4f} MCC: {:.4f}'.format(phase, f1_score(overall_labels, overall_preds),
                                      matthews_corrcoef(overall_labels, overall_preds)))
    print('{} AUROC: {:.4f}\n'.format(phase, roc_auc_score(overall_labels, overall_preds)))
