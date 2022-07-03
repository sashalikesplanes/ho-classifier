import numpy as np
import torch
from tsai.all import *
from scipy import io
import wandb
from fastai.callback.wandb import *
from getdata import get_fofu_data
from pathlib import Path


for_export = True

    
wandb.init()
config = wandb.config

path = Path('learners', config['export_name'])

# Either the condtions will be specified, or take the defaults
try:
    conds = config['desired_conditions']
    labels = config['condition_labels']
except KeyError:
    # Defaults
    conds = ['CL', 'CM', 'CH', 'PL', 'PM', 'PH', 'PRL', 'PRM', 'PRH']
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    
# 1D lists with each entry holding a single X, Y pair, splits indicated if it is validation or training
X, Y, splits = get_fofu_data(valid_pct=config["valid_pct"], data_file=config["data_file"], variables=config["variables"], random_labels=config["random_labels"], desired_conditions=conds, condition_labels=labels, valid_subject=config["valid_subject"])

# Datasets needed for FastAI library
tfms = [None, Categorize()]
dsets = TSDatasets(X, Y, tfms=tfms, splits=splits, inplace=True)

# Get data loaders
batch_tfms = None #[TSStandardize(by_var=config["standardize_by_var"], by_sample=config["standardize_by_sample"])]
# Needed for fast ai library, here you can set the things that apply to whole data
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=config["batch_size"], batch_tfms=batch_tfms, shuffle=True, num_workers=8)
    

# Get model
cbs = []
model = InceptionTimePlus(dls.vars, dls.c, nf=config["nf"], bn=config["bn"], ks=config["ks"], bottleneck=config["bottleneck"],
                          conv_dropout=config["conv_dropout"], depth=config["depth"])
# Get learner
learner = Learner(dls, model, opt_func=Adam, metrics=accuracy, wd=config["wd"], cbs=cbs)
# Execute fit one cycle training
learner.fit_one_cycle(config["epochs"], lr_max=config["lr"])

learner.save_all(path='learners', dls_fname='dls', model_fname="model", learner_fname="learner") 
print("########################################################################")
print("EXPORT SUCCESS!!!")
print("########################################################################")
wandb.finish()
