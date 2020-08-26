import torch
import torch.nn as nn
import torch.nn.functional as F

import pretrainedmodels
from efficientnet_pytorch import EfficientNet

from utils import Mish


class FundusModel(nn.Module):
    def __init__(self, model_name, hidden_dim, dropout=0.5, activation='relu', output_class = 10):
        super().__init__()

        if activation.lower() == 'relu':
            activation = F.relu
        elif activation.lower() == 'mish':
            activation = Mish()

        self.model_name = model_name
        self.hidden_dim = hidden_dim

        if model_name.startswith('efficientnet'):
            self.cnn_model = EfficientNet.from_name(model_name)
            dim_feats= self.cnn_model._fc.in_features

        else:
            self.cnn_model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet') 
            dim_feats = self.cnn_model.last_linear.in_features  
        
        self.linear1 = nn.Linear(dim_feats, hidden_dim)

        output_classes = output_class
        self.linear2 = nn.Linear(hidden_dim, output_classes)
        
        self.dropout = nn.Dropout(p=dropout) if dropout else None
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.act = activation

    def features(self, input):
        if self.model_name.startswith('efficientnet'):
            return self.cnn_model.extract_features(input)
        else:
            return self.cnn_model.features(input)

    def logits(self, feature):
        if self.model_name.startswith('efficientnet'):
            output = self.pool(feature)
            output = output.view(output.size(0), -1)
            output = self.linear1(output)
            output = self.act(output)

            if self.dropout:
                output = self.dropout(output)
            output = self.linear2(output)

        elif self.model_name.startswith('resnet'):
            output = self.pool(feature)
            output = output.view(output.size(0), -1)
            output = self.linear1(output)
            output = self.act(output)
            
            if self.dropout:
                output = self.dropout(output)
            output = self.linear2(output)
            
        elif self.model_name.startswith('densenet'):
            output = self.act(feature, inplace = True)
            output = self.pool(output)
            output = output.view(output.size(0), -1)
            output = self.linear1(output)
            output = self.act(output)
            
            if self.dropout:
                output = self.dropout(output)
            output = self.linear2(output)
            
        elif self.model_name.startswith('se_resnet'):
            output = self.pool(feature)

            if self.dropout:
              output = self.dropout(output)

            output = output.view(output.size(0), -1)
            output = self.linear1(output)
            output = self.act(output)

            if self.dropout:
              output = self.dropout(output)

            output = self.linear2(output)

        elif self.model_name.startswith('se_resnext'):
            output = self.pool(feature)

            if self.dropout:
              output = self.dropout(output)

            output = output.view(output.size(0), -1)
            output = self.linear1(output)
            output = self.act(output)

            if self.dropout:
              output = self.dropout(output)
            output = self.linear2(output)
        
        return output

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)

        return x
if __name__ == '__main__':
    model = FundusModel(model_name='efficientnet-b1', hidden_dim=128, output_class=10)
    img = torch.randn((1,3,256,256))
    x = model(img)
    print(x)
    print(x.shape)
    _, y = torch.max(x, 1)
    print(y)
    print(y.shape)