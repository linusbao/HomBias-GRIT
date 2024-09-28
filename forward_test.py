# import torch
# #import torch_geoemetric
# import os
# from torch_geometric.data import DataLoader

# from torch_geometric.graphgym.cmd_args import parse_args
# from torch_geometric.graphgym.config import (cfg, dump_cfg,
#                                              set_cfg, load_cfg,
#                                              makedirs_rm_exist)
# from torch_geometric.graphgym.loader import create_loader
# from torch_geometric.graphgym.logger import set_printing
# from torch_geometric.graphgym.optim import create_optimizer, \
#     create_scheduler, OptimizerConfig
# from torch_geometric.graphgym.model_builder import create_model
# from torch_geometric.graphgym.train import GraphGymDataModule, train
# from torch_geometric.graphgym.utils.comp_budget import params_count
# from torch_geometric.graphgym.utils.device import auto_select_device
# from torch_geometric.graphgym.register import train_dict
# from torch_geometric import seed_everything

# import torch_geometric.transforms as T

# from pathlib import Path
# import sys
# import os
# sys.path.append(Path(__file__).parent)
# import grit

# from pathlib import Path
# import sys
# import os
# sys.path.append(os.path.join(Path(__file__).parent, 'grit'))
# from utils import custom_create_loader
# sys.path.append(os.path.join(Path(__file__).parent.parent, 'hombasis-gt', 'qm9','data_GraphGym_QM9'))
# from CustomDataset import CustomDataset

import datetime
import os
import torch
import logging

# from pathlib import Path
# import sys
# import os
# sys.path.append(Path(__file__).parent.parent)
# print(f'path: {Path(__file__).parent.parent}')

import grit  # noqa, register custom modules
from grit.optimizer.extra_optimizers import ExtendedSchedulerConfig

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             # set_agg_dir,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import create_optimizer, \
    create_scheduler, OptimizerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import train
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything

from grit.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from grit.logger import create_logger

#EDITTED added for fixing large parallel jobs
import os.path as osp
from torch_geometric.io import fs
import random

def custom_set_out_dir(cfg, cfg_fname, name_tag):
    """Set custom main output directory path to cfg.
    Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
        cfg_fname (string): Filename for the yaml format configuration file
        name_tag (string): Additional name tag to identify this execution of the
            configuration file, specified in :obj:`cfg.name_tag`
    """
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)



args = parse_args()

set_cfg(cfg)
cfg.set_new_allowed(True)
load_cfg(cfg, args)

custom_set_out_dir(cfg, 'test1', 'test2')
dump_cfg(cfg)

# datamodule = GraphGymDataModule()
# loader = datamodule.train_dataloader()
# loaders = create_loader() if not (hasattr(cfg.dataset, "follow_edge_batch") and cfg.dataset.follow_edge_batch) else custom_create_loader()
loaders = create_loader()
loader = loaders[0]
for batch in loader:
    batch=batch

print(f'Batch: {batch}')

#for viewing counts
# print(f'batch.counts: {batch.counts}')
# print(f'batch.counts.size(): {batch.counts.size()}')

setattr(cfg, "accelerator", "cpu")
model = create_model()

print('ATTEMPTING FORWARD PASS')
pred, _ = model(batch)
print('FORWARD PASS SUCCESSFULL')
print(f"predictions: {pred}")

#uncomment if need param count as well
# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print("Total trainable parameters:", total_params)
