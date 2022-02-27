import torch
from torch import nn
from copy import deepcopy
from torchvision.models import resnet
from functools import partial

import torch.nn.functional as F
from copy import deepcopy
import pickle

# SplitBatchNorm: simulate multi-gpu behavior of BatchNorm in one gpu by splitting alone the batch dimension
# implementation adapted from https://github.com/davidcpage/cifar10-fast/blob/master/torch_backend.py
class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        
    def forward(self, input):
        N, C, H, W = input.shape

        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split, 
                                                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits), True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(input, self.running_mean, self.running_var, 
                                            self.weight, self.bias, False, self.momentum, self.eps)

class ContinualNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_groups=8, **kw):
        super().__init__(num_features, **kw)
        self.G = num_groups
    
    def forward(self, input):
        
        if self.training or not self.track_running_stats:
            out_gn = nn.functional.group_norm(input, self.G, None, None, self.eps)
            outcome = nn.functional.batch_norm(out_gn, self.running_mean, self.running_var, self.weight, self.bias, True, self.momentum, self.eps)
            return outcome
        
        else:
            out_gn = nn.functional.group_norm(input, self.G, None, None, self.eps)
            outcome = nn.functional.batch_norm(out_gn, self.running_mean, self.running_var, self.weight, self.bias, False, self.momentum, self.eps)
            return outcome
        
class LLL_Net(nn.Module):
    """Basic class for implementing networks"""

    def __init__(self, model, remove_existing_head=False):
        head_var = model.head_var
        assert type(head_var) == str
        assert not remove_existing_head or hasattr(model, head_var), \
            "Given model does not have a variable called {}".format(head_var)
        assert not remove_existing_head or type(getattr(model, head_var)) in [nn.Sequential, nn.Linear], \
            "Given model's head {} does is not an instance of nn.Sequential or nn.Linear".format(head_var)
        super(LLL_Net, self).__init__()
        self._initialize()
        self.model = model
        last_layer = getattr(self.model, head_var)

        if remove_existing_head:
            if type(last_layer) == nn.Sequential:
                self.out_size = last_layer[-1].in_features
                # strips off last linear layer of classifier
                del last_layer[-1]
            elif type(last_layer) == nn.Linear:
                self.out_size = last_layer.in_features
                # converts last layer into identity
                # setattr(self.model, head_var, nn.Identity())
                # WARNING: this is for when pytorch version is <1.2
                setattr(self.model, head_var, nn.Sequential())
        else:
            self.out_size = last_layer.out_features

        self.heads = nn.ModuleList()
        self.task_cls = []
        self.task_offset = []
        self._initialize_weights()

    def add_head(self, num_outputs):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the corresponding offsets"""
        self.heads.append(nn.Linear(self.out_size, num_outputs))
        # we re-compute instead of append in case an approach makes changes to the heads
        self.task_cls = torch.tensor([head.out_features for head in self.heads])
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])

    def forward(self, x, return_features=False):
        """Applies the forward pass

        Simplification to work on multi-head only -- returns all head outputs in a list
        Args:
            x (tensor): input images
            return_features (bool): return the representations before the heads
        """
        
        x = self.model(x)
        assert (len(self.heads) > 0), "Cannot access any head"
        y = []
        for head in self.heads:
            y.append(head(x))
        if return_features:
            return y, x
        else:
            return y

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        """Freeze all parameters from the main model, but not the heads"""
        for param in self.model.parameters():
            param.requires_grad = False

    def freeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _initialize(self):
        # for saving gamma, beta, running_mean, running_var, initialize structure
        self.weight_1, self.bias_1, self.running_mean_1, self.running_var_1, self.mean_1, self.var_1 = [], [], [], [], [], []
        self.weight_2, self.bias_2, self.running_mean_2, self.running_var_2, self.mean_2, self.var_2 = [], [], [], [], [], []
        self.weight_3, self.bias_3, self.running_mean_3, self.running_var_3, self.mean_3, self.var_3 = [], [], [], [], [], []
        self.weight_4, self.bias_4, self.running_mean_4, self.running_var_4, self.mean_4, self.var_4 = [], [], [], [], [], []
        self.weight_5, self.bias_5, self.running_mean_5, self.running_var_5, self.mean_5, self.var_5 = [], [], [], [], [], []
        self.weight_6, self.bias_6, self.running_mean_6, self.running_var_6, self.mean_6, self.var_6 = [], [], [], [], [], []
        self.weight_7, self.bias_7, self.running_mean_7, self.running_var_7, self.mean_7, self.var_7 = [], [], [], [], [], []
        self.weight_8, self.bias_8, self.running_mean_8, self.running_var_8, self.mean_8, self.var_8 = [], [], [], [], [], []
        self.weight_9, self.bias_9, self.running_mean_9, self.running_var_9, self.mean_9, self.var_9 = [], [], [], [], [], []
        self.weight_10, self.bias_10, self.running_mean_10, self.running_var_10, self.mean_10, self.var_10 = [], [], [], [], [], []
        self.weight_11, self.bias_11, self.running_mean_11, self.running_var_11, self.mean_11, self.var_11 = [], [], [], [], [], []
        self.weight_12, self.bias_12, self.running_mean_12, self.running_var_12, self.mean_12, self.var_12 = [], [], [], [], [], []
        self.weight_13, self.bias_13, self.running_mean_13, self.running_var_13, self.mean_13, self.var_13 = [], [], [], [], [], []
        self.weight_14, self.bias_14, self.running_mean_14, self.running_var_14, self.mean_14, self.var_14 = [], [], [], [], [], []
        self.weight_15, self.bias_15, self.running_mean_15, self.running_var_15, self.mean_15, self.var_15 = [], [], [], [], [], []
        self.weight_16, self.bias_16, self.running_mean_16, self.running_var_16, self.mean_16, self.var_16 = [], [], [], [], [], []
        self.weight_17, self.bias_17, self.running_mean_17, self.running_var_17, self.mean_17, self.var_17 = [], [], [], [], [], []
        self.weight_18, self.bias_18, self.running_mean_18, self.running_var_18, self.mean_18, self.var_18 = [], [], [], [], [], []
        self.weight_19, self.bias_19, self.running_mean_19, self.running_var_19, self.mean_19, self.var_19 = [], [], [], [], [], []
        self.weight_20, self.bias_20, self.running_mean_20, self.running_var_20, self.mean_20, self.var_20 = [], [], [], [], [], []

    def save_bn_parameters(self):
        print("Copying BN parameters...")
        self.weight_1.append(deepcopy(self.model.bn1.weight))
        self.bias_1.append(deepcopy(self.model.bn1.bias))
        self.running_mean_1.append(deepcopy(self.model.bn1.running_mean))
        self.running_var_1.append(deepcopy(self.model.bn1.running_var))
        
        self.weight_2.append(deepcopy(self.model.layer1[0].bn1.weight))
        self.bias_2.append(deepcopy(self.model.layer1[0].bn1.bias))
        self.running_mean_2.append(deepcopy(self.model.layer1[0].bn1.running_mean))
        self.running_var_2.append(deepcopy(self.model.layer1[0].bn1.running_var))
        
        self.weight_3.append(deepcopy(self.model.layer1[0].bn2.weight))
        self.bias_3.append(deepcopy(self.model.layer1[0].bn2.bias))
        self.running_mean_3.append(deepcopy(self.model.layer1[0].bn2.running_mean))
        self.running_var_3.append(deepcopy(self.model.layer1[0].bn2.running_var))
        
        self.weight_4.append(deepcopy(self.model.layer1[1].bn1.weight))
        self.bias_4.append(deepcopy(self.model.layer1[1].bn1.bias))
        self.running_mean_4.append(deepcopy(self.model.layer1[1].bn1.running_mean))
        self.running_var_4.append(deepcopy(self.model.layer1[1].bn1.running_var))
        
        self.weight_5.append(deepcopy(self.model.layer1[1].bn2.weight))
        self.bias_5.append(deepcopy(self.model.layer1[1].bn2.bias))
        self.running_mean_5.append(deepcopy(self.model.layer1[1].bn2.running_mean))
        self.running_var_5.append(deepcopy(self.model.layer1[1].bn2.running_var))
        
        self.weight_6.append(deepcopy(self.model.layer2[0].bn1.weight))
        self.bias_6.append(deepcopy(self.model.layer2[0].bn1.bias))
        self.running_mean_6.append(deepcopy(self.model.layer2[0].bn1.running_mean))
        self.running_var_6.append(deepcopy(self.model.layer2[0].bn1.running_var))
        
        self.weight_7.append(deepcopy(self.model.layer2[0].bn2.weight))
        self.bias_7.append(deepcopy(self.model.layer2[0].bn2.bias))
        self.running_mean_7.append(deepcopy(self.model.layer2[0].bn2.running_mean))
        self.running_var_7.append(deepcopy(self.model.layer2[0].bn2.running_var))
        
        self.weight_8.append(deepcopy(self.model.layer2[0].downsample[1].weight))
        self.bias_8.append(deepcopy(self.model.layer2[0].downsample[1].bias))
        self.running_mean_8.append(deepcopy(self.model.layer2[0].downsample[1].running_mean))
        self.running_var_8.append(deepcopy(self.model.layer2[0].downsample[1].running_var))
        
        self.weight_9.append(deepcopy(self.model.layer2[1].bn1.weight))
        self.bias_9.append(deepcopy(self.model.layer2[1].bn1.bias))
        self.running_mean_9.append(deepcopy(self.model.layer2[1].bn1.running_mean))
        self.running_var_9.append(deepcopy(self.model.layer2[1].bn1.running_var))
        
        self.weight_10.append(deepcopy(self.model.layer2[1].bn2.weight))
        self.bias_10.append(deepcopy(self.model.layer2[1].bn2.bias))
        self.running_mean_10.append(deepcopy(self.model.layer2[1].bn2.running_mean))
        self.running_var_10.append(deepcopy(self.model.layer2[1].bn2.running_var))
        
        self.weight_11.append(deepcopy(self.model.layer3[0].bn1.weight))
        self.bias_11.append(deepcopy(self.model.layer3[0].bn1.bias))
        self.running_mean_11.append(deepcopy(self.model.layer3[0].bn1.running_mean))
        self.running_var_11.append(deepcopy(self.model.layer3[0].bn1.running_var))
        
        self.weight_12.append(deepcopy(self.model.layer3[0].bn2.weight))
        self.bias_12.append(deepcopy(self.model.layer3[0].bn2.bias))
        self.running_mean_12.append(deepcopy(self.model.layer3[0].bn2.running_mean))
        self.running_var_12.append(deepcopy(self.model.layer3[0].bn2.running_var))
        
        self.weight_13.append(deepcopy(self.model.layer3[0].downsample[1].weight))
        self.bias_13.append(deepcopy(self.model.layer3[0].downsample[1].bias))
        self.running_mean_13.append(deepcopy(self.model.layer3[0].downsample[1].running_mean))
        self.running_var_13.append(deepcopy(self.model.layer3[0].downsample[1].running_var))
        
        self.weight_14.append(deepcopy(self.model.layer3[1].bn1.weight))
        self.bias_14.append(deepcopy(self.model.layer3[1].bn1.bias))
        self.running_mean_14.append(deepcopy(self.model.layer3[1].bn1.running_mean))
        self.running_var_14.append(deepcopy(self.model.layer3[1].bn1.running_var))
        
        self.weight_15.append(deepcopy(self.model.layer3[1].bn2.weight))
        self.bias_15.append(deepcopy(self.model.layer3[1].bn2.bias))
        self.running_mean_15.append(deepcopy(self.model.layer3[1].bn2.running_mean))
        self.running_var_15.append(deepcopy(self.model.layer3[1].bn2.running_var))
        
        self.weight_16.append(deepcopy(self.model.layer4[0].bn1.weight))
        self.bias_16.append(deepcopy(self.model.layer4[0].bn1.bias))
        self.running_mean_16.append(deepcopy(self.model.layer4[0].bn1.running_mean))
        self.running_var_16.append(deepcopy(self.model.layer4[0].bn1.running_var))
        
        self.weight_17.append(deepcopy(self.model.layer4[0].bn2.weight))
        self.bias_17.append(deepcopy(self.model.layer4[0].bn2.bias))
        self.running_mean_17.append(deepcopy(self.model.layer4[0].bn2.running_mean))
        self.running_var_17.append(deepcopy(self.model.layer4[0].bn2.running_var))
        
        self.weight_18.append(deepcopy(self.model.layer4[0].downsample[1].weight))
        self.bias_18.append(deepcopy(self.model.layer4[0].downsample[1].bias))
        self.running_mean_18.append(deepcopy(self.model.layer4[0].downsample[1].running_mean))
        self.running_var_18.append(deepcopy(self.model.layer4[0].downsample[1].running_var))
        
        self.weight_19.append(deepcopy(self.model.layer4[1].bn1.weight))
        self.bias_19.append(deepcopy(self.model.layer4[1].bn1.bias))
        self.running_mean_19.append(deepcopy(self.model.layer4[1].bn1.running_mean))
        self.running_var_19.append(deepcopy(self.model.layer4[1].bn1.running_var))
        
        self.weight_20.append(deepcopy(self.model.layer4[1].bn2.weight))
        self.bias_20.append(deepcopy(self.model.layer4[1].bn2.bias))
        self.running_mean_20.append(deepcopy(self.model.layer4[1].bn2.running_mean))
        self.running_var_20.append(deepcopy(self.model.layer4[1].bn2.running_var))
        
    def log_bn_parameters(self, full_exp_name=None):
        print("Logging BN parameters")
        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_1.txt", 'wb') as fp:
            pickle.dump(self.weight_1, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_1.txt", 'wb') as fp:
            pickle.dump(self.bias_1, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_1.txt", 'wb') as fp:
            pickle.dump(self.running_mean_1, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_1.txt", 'wb') as fp:
            pickle.dump(self.running_var_1, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_2.txt", 'wb') as fp:
            pickle.dump(self.weight_2, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_2.txt", 'wb') as fp:
            pickle.dump(self.bias_2, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_2.txt", 'wb') as fp:
            pickle.dump(self.running_mean_2, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_2.txt", 'wb') as fp:
            pickle.dump(self.running_var_2, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_3.txt", 'wb') as fp:
            pickle.dump(self.weight_3, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_3.txt", 'wb') as fp:
            pickle.dump(self.bias_3, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_3.txt", 'wb') as fp:
            pickle.dump(self.running_mean_3, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_3.txt", 'wb') as fp:
            pickle.dump(self.running_var_3, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_4.txt", 'wb') as fp:
            pickle.dump(self.weight_4, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_4.txt", 'wb') as fp:
            pickle.dump(self.bias_4, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_4.txt", 'wb') as fp:
            pickle.dump(self.running_mean_4, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_4.txt", 'wb') as fp:
            pickle.dump(self.running_var_4, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_5.txt", 'wb') as fp:
            pickle.dump(self.weight_5, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_5.txt", 'wb') as fp:
            pickle.dump(self.bias_5, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_5.txt", 'wb') as fp:
            pickle.dump(self.running_mean_5, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_5.txt", 'wb') as fp:
            pickle.dump(self.running_var_5, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_6.txt", 'wb') as fp:
            pickle.dump(self.weight_6, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_6.txt", 'wb') as fp:
            pickle.dump(self.bias_6, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_6.txt", 'wb') as fp:
            pickle.dump(self.running_mean_6, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_6.txt", 'wb') as fp:
            pickle.dump(self.running_var_6, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_7.txt", 'wb') as fp:
            pickle.dump(self.weight_7, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_7.txt", 'wb') as fp:
            pickle.dump(self.bias_7, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_7.txt", 'wb') as fp:
            pickle.dump(self.running_mean_7, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_7.txt", 'wb') as fp:
            pickle.dump(self.running_var_7, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_8.txt", 'wb') as fp:
            pickle.dump(self.weight_8, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_8.txt", 'wb') as fp:
            pickle.dump(self.bias_8, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_8.txt", 'wb') as fp:
            pickle.dump(self.running_mean_8, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_8.txt", 'wb') as fp:
            pickle.dump(self.running_var_8, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_9.txt", 'wb') as fp:
            pickle.dump(self.weight_9, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_9.txt", 'wb') as fp:
            pickle.dump(self.bias_9, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_9.txt", 'wb') as fp:
            pickle.dump(self.running_mean_9, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_9.txt", 'wb') as fp:
            pickle.dump(self.running_var_9, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_10.txt", 'wb') as fp:
            pickle.dump(self.weight_10, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_10.txt", 'wb') as fp:
            pickle.dump(self.bias_10, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_10.txt", 'wb') as fp:
            pickle.dump(self.running_mean_10, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_10.txt", 'wb') as fp:
            pickle.dump(self.running_var_10, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_11.txt", 'wb') as fp:
            pickle.dump(self.weight_11, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_11.txt", 'wb') as fp:
            pickle.dump(self.bias_11, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_11.txt", 'wb') as fp:
            pickle.dump(self.running_mean_11, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_11.txt", 'wb') as fp:
            pickle.dump(self.running_var_11, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_12.txt", 'wb') as fp:
            pickle.dump(self.weight_12, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_12.txt", 'wb') as fp:
            pickle.dump(self.bias_12, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_12.txt", 'wb') as fp:
            pickle.dump(self.running_mean_12, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_12.txt", 'wb') as fp:
            pickle.dump(self.running_var_12, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_13.txt", 'wb') as fp:
            pickle.dump(self.weight_13, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_13.txt", 'wb') as fp:
            pickle.dump(self.bias_13, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_13.txt", 'wb') as fp:
            pickle.dump(self.running_mean_13, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_13.txt", 'wb') as fp:
            pickle.dump(self.running_var_13, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_14.txt", 'wb') as fp:
            pickle.dump(self.weight_14, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_14.txt", 'wb') as fp:
            pickle.dump(self.bias_14, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_14.txt", 'wb') as fp:
            pickle.dump(self.running_mean_14, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_14.txt", 'wb') as fp:
            pickle.dump(self.running_var_14, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_15.txt", 'wb') as fp:
            pickle.dump(self.weight_15, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_15.txt", 'wb') as fp:
            pickle.dump(self.bias_15, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_15.txt", 'wb') as fp:
            pickle.dump(self.running_mean_15, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_15.txt", 'wb') as fp:
            pickle.dump(self.running_var_15, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_16.txt", 'wb') as fp:
            pickle.dump(self.weight_16, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_16.txt", 'wb') as fp:
            pickle.dump(self.bias_16, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_16.txt", 'wb') as fp:
            pickle.dump(self.running_mean_16, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_16.txt", 'wb') as fp:
            pickle.dump(self.running_var_16, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_17.txt", 'wb') as fp:
            pickle.dump(self.weight_17, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_17.txt", 'wb') as fp:
            pickle.dump(self.bias_17, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_17.txt", 'wb') as fp:
            pickle.dump(self.running_mean_17, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_17.txt", 'wb') as fp:
            pickle.dump(self.running_var_17, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_18.txt", 'wb') as fp:
            pickle.dump(self.weight_18, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_18.txt", 'wb') as fp:
            pickle.dump(self.bias_18, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_18.txt", 'wb') as fp:
            pickle.dump(self.running_mean_18, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_18.txt", 'wb') as fp:
            pickle.dump(self.running_var_18, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_19.txt", 'wb') as fp:
            pickle.dump(self.weight_19, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_19.txt", 'wb') as fp:
            pickle.dump(self.bias_19, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_19.txt", 'wb') as fp:
            pickle.dump(self.running_mean_19, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_19.txt", 'wb') as fp:
            pickle.dump(self.running_var_19, fp)

        with open("../results/" + full_exp_name + "/bn_parameters/"+"weight_20.txt", 'wb') as fp:
            pickle.dump(self.weight_20, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"bias_20.txt", 'wb') as fp:
            pickle.dump(self.bias_20, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_mean_20.txt", 'wb') as fp:
            pickle.dump(self.running_mean_20, fp)
        with open("../results/" + full_exp_name + "/bn_parameters/"+"running_var_20.txt", 'wb') as fp:
            pickle.dump(self.running_var_20, fp)
