import filelock
import os
from typing import Any, Dict, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn
from torch import optim
from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext

import data
from data import download_pach_repo


TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]

class MRIUnetTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext):
        self.context = context
        self.config = self.context.get_data_config()
        
        #self.download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"
        #self.download_directory = self.config["data_dir"]
        data_config = self.context.get_data_config()
        print(data_config)
        #data_dir = os.path.join(self.download_directory, 'data')
        download_dir = self.config["download_directory"]
        data_dir = self.config["data_dir"]
        print("download_dir = " + download_dir)
        print("data_dir = " + data_dir)
        print("HEYOOOOO")

        full_dir = "/"
        full_dir = os.path.join(full_dir, download_dir.strip("/"), data_dir.strip("/"))

        print("full_dir = " + full_dir)

        des = self.download_data(data_config, download_dir)

        print(str(des))
        print("!!! Got here")
        #print(self.download_directory)
        print(self.context.get_hparam("split_seed"))
        print(self.context.get_hparam("validation_ratio"))

        print("Download Directory = " + download_dir)

        self.train_dataset, self.val_dataset = data.get_train_val_datasets(download_dir,
                data_dir,
                self.context.get_hparam("split_seed"),
                self.context.get_hparam("validation_ratio"))
        
#        self.download_directory = torch.hub.get_dir()
        print("!!! and Got here")
        print(download_dir)


        if not os.path.exists(full_dir):
            os.makedirs(full_dir)
            
        with filelock.FileLock(os.path.join(full_dir, "download.lock")):
            model = torch.hub.load(self.config["repo"],
                                   self.config["model"],
                                   in_channels=self.context.get_hparam("input_channels"),
                                   out_channels=self.context.get_hparam("output_channels"),
                                   init_features=self.context.get_hparam("init_features"),
                                   pretrained=self.context.get_hparam("pretrained"))
        
        self.model = self.context.wrap_model(model)
        self.optimizer = self.context.wrap_optimizer(optim.Adam(self.model.parameters(),
                                                                lr=self.context.get_hparam("learning_rate"),
                                                                weight_decay=self.context.get_hparam("weight_decay")))
        

    def iou(self, pred, label):
        intersection = (pred * label).sum()
        union = pred.sum() + label.sum() - intersection
        if pred.sum() == 0 and label.sum() == 0:
            return 1
        return intersection / union
    
    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int):
        imgs, masks = batch
        output = self.model(imgs)
        loss = torch.nn.functional.binary_cross_entropy(output, masks)
        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)
        iou = self.iou((output>0.5).int(), masks)
        return {"loss": loss, "IoU": iou}
        

    def evaluate_batch(self, batch: TorchData):
        imgs, masks = batch
        output = self.model(imgs)
        loss = torch.nn.functional.binary_cross_entropy(output, masks)
        iou = self.iou((output>0.5).int(), masks)
        return {"val_loss": loss, "val_IoU": iou}

    def build_training_data_loader(self):
        return DataLoader(self.train_dataset, batch_size=self.context.get_per_slot_batch_size(), shuffle=True)

    def build_validation_data_loader(self):
        return DataLoader(self.val_dataset, batch_size=self.context.get_per_slot_batch_size())




    # -------------------------------------------------------------------------
    
    ### MODEL_DEF from image-classification below

    # REMOVED MOST METHODS, DUPLICATES ABOVE
    
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def download_data(self, data_config, data_dir):
   


        files = download_pach_repo(
            data_config['pachyderm']['host'],
            data_config['pachyderm']['port'],
            data_config["pachyderm"]["repo"],
            data_config["pachyderm"]["branch"],
            data_dir,
            data_config["pachyderm"]["token"]
        )
        print(f'Data dir set to : {data_dir}')
        return [des for src, des in files ]

    # -------------------------------------------------------------------------
