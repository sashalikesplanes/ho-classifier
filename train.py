import numpy as np
import torch
from tsai.all import *
from scipy import io
import wandb
from fastai.callback.wandb import *
from getdata import get_fofu_data

# Define a custom WandBCallback in order to not log after each batch
class CustomWandbCallback(WandbCallback):
    def after_batch(self):
        if self.training:
            self._wandb_step += 1
            self._wandb_epoch += 1/self.n_iter
            wandb.log({}, step=self._wandb_step)
    def after_epoch(self):
        super(CustomWandbCallback, self).after_epoch()
        wandb.log({n:s for n,s in zip(self.recorder.metric_names, self.recorder.log) if n not in ['train_loss', 'epoch', 'time']}, step=self._wandb_step)

    
wandb.init()
config = wandb.config
# Get the data sets

X, Y, splits = get_fofu_data(valid_pct=config["valid_pct"], data_file=config["data_file"])

tfms = [None, Categorize()]
dsets = TSDatasets(X, Y, tfms=tfms, splits=splits, inplace=True)

# Get data loaders
batch_tfms = None #[TSStandardize(by_var=config["standardize_by_var"], by_sample=config["standardize_by_sample"])]
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=config["batch_size"], batch_tfms=batch_tfms, shuffle=True, num_workers=8)
    

# Get model
cbs = [CustomWandbCallback(log=None, log_preds=True, log_model=False, dataset_name='valid runs random, standardized, 50Hz')]
model = InceptionTimePlus(dls.vars, dls.c, nf=config["nf"], bn=config["bn"], ks=config["ks"], bottleneck=config["bottleneck"],
                          conv_dropout=config["conv_dropout"], depth=config["depth"])
# Get learner
learner = Learner(dls, model, opt_func=Adam, metrics=accuracy, wd=config["wd"], cbs=cbs)
# Execute fit one cycle training
learner.fit_one_cycle(config["epochs"], lr_max=config["lr"])
print()
wandb.finish()