#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 11/3/18

from abc import ABCMeta, abstractmethod
import torch.nn as nn


class NMTModel(nn.Module):
    """"
    base class for all Sequence to sequence (NMT) models
    """
    # TODO: move stuff here that is common to all
    pass

    @property
    @abstractmethod
    def model_dim(self):
        pass
