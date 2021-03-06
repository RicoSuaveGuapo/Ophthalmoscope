import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

class Attension(nn.Module):
    def __init__(self, base_model, pt_depth, feature_size, output_class=5, freeze=False):
        super().__init__()
        # base model parameters should freeze
        self.base_model = base_model
        for name, par in self.base_model.named_parameters():
            if name.startswith('cnn_model'):
                par.requires_grad = freeze
        self.base_feature = base_model.features
        self.attn_1 = nn.Conv2d(in_channels = pt_depth, out_channels= 64, kernel_size=1)
        self.attn_2 = nn.Conv2d(in_channels = 64, out_channels= 16, kernel_size=1)
        self.attn_3 = nn.Conv2d(in_channels = 16, out_channels= 8, kernel_size=1)
        self.attn_4 = nn.Conv2d(in_channels = 8, out_channels= 1, kernel_size=1)
        self.attn_5 = nn.Conv2d(in_channels = 1, out_channels= pt_depth, kernel_size=1, bias=False)
        self.attn_5.weight = torch.nn.Parameter(torch.ones(pt_depth,1,1,1))
        self.attn_5.weight.requires_grad = False
        self.act = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.drop_1 = nn.Dropout()
        self.drop_2 = nn.Dropout(0.25)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bnor = nn.BatchNorm2d(pt_depth)
        self.lin_1 = nn.Linear(pt_depth, 128)
        self.lin_2 = nn.Linear(128, output_class)

    def attension(self, input):
        x = self.base_feature(input)
        Bx = self.bnor(x)
        x = self.drop_1(Bx)
        x = self.act(self.attn_1(x))
        x = self.act(self.attn_2(x))
        x = self.act(self.attn_3(x))
        x = self.sig(self.attn_4(x))
        at_x = self.attn_5(x) # linear activation
        x = torch.mul(at_x, Bx)
        gap_features = self.gap(x)
        gap_features = gap_features.view(-1,gap_features.size(1))
        gap_mask = self.gap(at_x)
        gap_mask = gap_mask.view(-1, gap_mask.size(1))

        return gap_features, gap_mask

    def missing(self, gap_features, gap_mask):
        x = gap_features/gap_mask
        x = self.drop_2(x)
        x = x.view(-1, x.size(1))
        x = self.drop_2(self.act(self.lin_1(x)))
        x = self.lin_2(x)
        return x

    def forward(self, input):
        gap_features, gap_mask = self.attension(input)
        output = self.missing(gap_features, gap_mask)
        return output

if __name__ == '__main__':
    model = FundusModel(model_name='se_resnext101_32x4d', hidden_dim=256, output_class=5)
    model = model.cuda()
    img = torch.randn((1,3,256,256)).cuda()
    feature_map = model.features(img)
    _, pt_depth, feature_size, _ = feature_map.shape
    att_model = Attension(base_model=model, pt_depth=pt_depth, feature_size=feature_size, output_class=5)

    att_model = att_model.cuda()
    
    x = att_model(img)
    # print(x)

    # x = model(img)
    # feature_map = model.features(img)
    # print(features.shape)
    # print(x)
    # print(x.shape)
    # _, y = torch.max(x, 1)
    # print(y)
    # print(y.shape)
