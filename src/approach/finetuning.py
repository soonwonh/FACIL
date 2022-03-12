import torch
import numpy as np
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """Class implementing the finetuning baseline"""

    def __init__(self, model, device, nepochs=100, lr_scheduler = 'multisteplr', lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False, logger=None, exemplars_dataset=None, 
                 all_outputs=False, batch_size=64, fix_batch=False, batch_ratio=3, model_freeze = False, change_mu=False, noise=0, cn=8, split_group=False):
        super(Appr, self).__init__(model, device, nepochs, lr_scheduler, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger, exemplars_dataset,
                                   batch_size, fix_batch, batch_ratio, model_freeze, change_mu, noise, cn, split_group)
        self.all_out = all_outputs
        print("all_outputs : ", all_outputs)

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--all-outputs', action='store_true', required=False,
                            help='Allow all weights related to all outputs to be modified (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1 and not self.all_out:
            # if there are no exemplars, previous heads are not modified
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        exemplars_loader = None
        current_loader = None
        
        if t>0:
            if self.fix_batch:
                current_batch = int(self.batch_size * self.batch_ratio / (self.batch_ratio + 1))
                current_loader = torch.utils.data.DataLoader(trn_loader.dataset,
                                                     batch_size=current_batch,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory,
                                                     worker_init_fn=np.random.seed(0))
                exemplars_batch = int(self.batch_size / (self.batch_ratio + 1))
                exemplars_loader = torch.utils.data.DataLoader(self.exemplars_dataset,
                                                     batch_size=exemplars_batch,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory,
                                                     worker_init_fn=np.random.seed(0))
            else:
                trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                         batch_size=trn_loader.batch_size,
                                                         shuffle=True,
                                                         num_workers=trn_loader.num_workers,
                                                         pin_memory=trn_loader.pin_memory,
                                                         worker_init_fn=np.random.seed(0))
            
        
         # FINETUNING TRAINING -- contains the epochs loop
        if len(self.exemplars_dataset) > 0 and t > 0:
            if self.fix_batch:
                print("fix_batch")
                super().train_loop(t, current_loader, val_loader, exemplars_loader)
        else:
            super().train_loop(t, trn_loader, val_loader)
        
        # add exemplars to train_loader
        if self.fix_batch:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)


        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        if self.all_out or len(self.exemplars_dataset) > 0:
            return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
