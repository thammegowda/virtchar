#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu]
# Created: 11/3/18


import os
import logging as log
import yaml

debug_mode = os.environ.get('NMT_DEBUG', False)
log.basicConfig(level=log.DEBUG if debug_mode else log.INFO)
log.debug(f"NMT_DEBUG={debug_mode}")

import torch
device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)
cpu_device = torch.device('cpu')

log.debug(f'device: {device}')
profiler = None
if os.environ.get('NMT_PROFILER') == 'memory':
    import memory_profiler
    profiler = memory_profiler.profile
    log.info('Setting memory profiler')


def my_tensor(*args, **kwargs):
    return torch.tensor(*args, device=device, **kwargs)


def profile(func, *args):
    """
    :param func: function to profile
    :param args: any addtional args for profiler
    :return:
    """

    if not profiler:
        return func
    return profiler(func, *args)


def load_conf(inp):
    with open(inp, encoding='utf-8') as fh:
        return yaml.load(fh)


from virtchar.tool import dataprep
from virtchar.tool.dataprep import BatchIterable, Batch
from virtchar.tool.exp import DialogExperiment
from virtchar.model import t2t


