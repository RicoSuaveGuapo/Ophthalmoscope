import time
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from model import FundusModel
from dataset import FundusDataset


def confusionMatrix(model_path, model_name, mode, output_class, plotclass, image_size, dataset=FundusDataset):

    model = FundusModel(model_name = model_name, output_class=output_class, hidden_dim=256)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print('model loaded')
    model = model.cuda()
    
    with torch.no_grad():
        dataset   = FundusDataset(mode=mode, image_size=image_size)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True, 
                                pin_memory=True, num_workers=2*os.cpu_count())

        targets = torch.Tensor().type(torch.long)
        predicts = torch.Tensor().type(torch.long).cuda()
        for i, data in enumerate(dataloader, start=0):
            input, target = data[0].cuda(), data[1]
            targets = torch.cat((targets, target))
            output = model(input) 
            _, predicted = torch.max(output, 1)
            predicts = torch.cat((predicts, predicted))
            print(f'-- {i} batch--')

        correct_count = (predicts == targets.cuda()).sum().item()
        accuracy = (100 * correct_count/len(dataset))
        print(f'\n Accuracy on {mode} set: %.2f %% \n' % (accuracy) )
        targets = targets.numpy()
        predicts = predicts.cpu().numpy()
        c_matrix = confusion_matrix(targets, predicts, normalize='true',
                                    labels=[i for i in range(plotclass)])    
    return c_matrix


if __name__ == '__main__':
    start_time = time.time()
    image_size = 600
    mode = 'val'
    output_class = 5
    plotclass = 5
    trial = 18
    model_name = 'se_resnext101_32x4d'
    # model_name = 'resnet18'

    path = os.path.join('model_save', f'{trial}_{model_name}_best.pth')
    # path = os.path.join('model_save', '1_resnet18.pth')

    c_matrix = confusionMatrix(
        model_path=path, model_name=model_name, 
        mode=mode, output_class=output_class,
        image_size = image_size, plotclass=plotclass)
    # print(type(c_matrix))

    import matplotlib.pyplot as plt

    figure = plt.figure()
    axes = figure.add_subplot(111)
    axes.matshow(c_matrix)
    axes.set_title(f'Confusion Matrix: {mode} set')
    axes.set(xlabel = 'Predicted',ylabel = 'Truth')
    axes.set_xticks(np.arange(0, plotclass-1))
    axes.set_yticks(np.arange(0, plotclass-1))
    caxes = axes.matshow(c_matrix, interpolation ='nearest') 
    figure.colorbar(caxes)
    
    for row_i, row in enumerate(c_matrix):
        for col_i, col in enumerate(row):
            axes.text(col_i-0.3,row_i+0.2,f'{col:.2f}',color='white')

    print(f'--- %.1f sec ---' % (time.time() - start_time))
    plt.savefig(f'confusion_matrix_{mode}.png')
    
