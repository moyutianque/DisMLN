import os
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import models
import dataset
from pytorch_lightning import loggers as pl_loggers

class BaseTrainer:
    def __init__(self, config):
        logger = pl_loggers.TensorBoardLogger(config.LOG_DIR, name=config.TRAINER.MODEL_NAME)
        self.model = getattr(models, config.TRAINER.MODEL_NAME)(config.TRAINER.MODEL)
        # train_dset = getattr(dataset, config.DATASET.NAME)(config.DATASET, config.DATASET.TRAIN.NAME)
        # valid_dset = getattr(dataset, config.DATASET.NAME)(config.DATASET, config.DATASET.EVAL.NAME)
        whole_set = getattr(dataset, config.DATASET.NAME)(config.DATASET, config.DATASET.TRAIN.NAME)
        train_dset, valid_dset = random_split(whole_set, [100000, len(whole_set)-100000])
        self.train_loader = DataLoader(train_dset, batch_size=config.DATASET.TRAIN.batch_size, num_workers=config.DATASET.TRAIN.num_workers, shuffle=True, drop_last=True, collate_fn=whole_set.collate_fc)
        self.valid_loader = DataLoader(valid_dset, batch_size=config.DATASET.EVAL.batch_size, num_workers=config.DATASET.EVAL.num_workers, shuffle=False, drop_last=False, collate_fn=whole_set.collate_fc)

        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config.LOG_DIR, "ckpts"), save_top_k=2, monitor="val_recall")
        self.trainer = pl.Trainer(
            max_epochs=config.TRAINER.EPOCHS, progress_bar_refresh_rate=20, 
            default_root_dir=config.LOG_DIR,
            logger=logger,
            callbacks=[checkpoint_callback],
            gpus=1,
            val_check_interval=500 # how oftern validation
        )

    def train(self):
        self.trainer.fit(self.model, self.train_loader, self.valid_loader)


