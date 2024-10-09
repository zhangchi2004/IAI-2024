from dataloader import get_dataloader
from model import MLP, CNN, LSTM
import yaml
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import EarlyStopping
import torch.nn as nn
import torch.nn.functional as F
import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model", type=str,default="MLP")
    return parser.parse_args()

def get_model(model_name, config):
    if model_name == "CNN":
        return CNN(config)
    elif model_name == "LSTM":
        return LSTM(config)
    else: # default: MLP
        return MLP(config)


def train():
    config_path = parse().config
    model_name = parse().model
    print(model_name)
    
    with open(config_path,'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    
    devices = torch.cuda.device_count()
    pad_len = config["pad_len"]
    data_path = config["data_path"]
    model = get_model(model_name,config)
    
    batch_size = config["batch_size"]
    wandb_path = config["wandb_path"]
    wandb_logger = WandbLogger(
        save_dir = wandb_path,
        project = "4",
        config = config,
        name = model_name +" "+ str(batch_size),
    )
    max_steps = config["max_steps"]
    # early_stop_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=10, verbose=False, mode="max")
    trainer = Trainer(
        accelerator = "gpu",
        devices = devices,
        logger = wandb_logger,
        max_steps = max_steps,
        check_val_every_n_epoch = 1,
        log_every_n_steps = 50,
        strategy = DDPStrategy(find_unused_parameters=True),
        callbacks=[early_stop_callback],
    )
    
    train_loader = get_dataloader(data_path, "train", batch_size, pad_len)
    print(len(train_loader))
    val_loader = get_dataloader(data_path, "validation", batch_size, pad_len)
    print(len(val_loader))
    test_loader = get_dataloader(data_path, "test", batch_size, pad_len)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    
if __name__ == "__main__":
    train()
