import torch.nn as nn

def logistic_regression_model(in_num):
    return nn.Sequential(
        nn.Flatten(),
        # nn.Dropout(0.8),
        nn.Linear(in_features=in_num, out_features=1, bias=True),
        nn.Sigmoid()
    )