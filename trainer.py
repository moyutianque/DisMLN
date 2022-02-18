import os
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import models
import dataset
from pytorch_lightning import loggers as pl_loggers

class BaseTrainer:
    def __init__(self, config, run_type):
        self.model_config = config.TRAINER.MODEL
        logger = pl_loggers.TensorBoardLogger(config.LOG_DIR, name=config.TRAINER.MODEL_NAME)
        self.model = getattr(models, config.TRAINER.MODEL_NAME)(config.TRAINER.MODEL)
        
        if run_type == 'train':
            train_dset = getattr(dataset, config.DATASET.NAME)(config.DATASET, config.DATASET.TRAIN.NAME)
            valid_dset = getattr(dataset, config.DATASET.NAME)(config.DATASET, config.DATASET.EVAL.NAME)
            self.train_loader = DataLoader(train_dset, batch_size=config.DATASET.TRAIN.batch_size, num_workers=config.DATASET.TRAIN.num_workers, shuffle=True, drop_last=True, collate_fn=train_dset.collate_fc)
            self.valid_loader = DataLoader(valid_dset, batch_size=config.DATASET.EVAL.batch_size, num_workers=config.DATASET.EVAL.num_workers, shuffle=False, drop_last=False, collate_fn=valid_dset.collate_fc)

            # whole_set = getattr(dataset, config.DATASET.NAME)(config.DATASET, config.DATASET.TRAIN.NAME)
            # train_len = int(len(whole_set) * 0.9)
            # train_dset, valid_dset = random_split(whole_set, [train_len, len(whole_set)-train_len])
            # self.train_loader = DataLoader(train_dset, batch_size=config.DATASET.TRAIN.batch_size, num_workers=config.DATASET.TRAIN.num_workers, shuffle=True, drop_last=True, collate_fn=whole_set.collate_fc)
            # self.valid_loader = DataLoader(valid_dset, batch_size=config.DATASET.EVAL.batch_size, num_workers=config.DATASET.EVAL.num_workers, shuffle=False, drop_last=False, collate_fn=whole_set.collate_fc)

        if run_type == 'inference':
            test_dset = getattr(dataset, config.DATASET.NAME)(config.DATASET, config.DATASET.TEST.NAME)
            self.test_loader = DataLoader(test_dset, batch_size=config.DATASET.TEST.batch_size, num_workers=config.DATASET.EVAL.num_workers, shuffle=False, drop_last=False, collate_fn=test_dset.collate_fc)

        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config.LOG_DIR, config.TRAINER.MODEL_NAME, "ckpts"), save_top_k=2, monitor="success_rate")
        self.trainer = pl.Trainer(
            max_epochs=config.TRAINER.EPOCHS, progress_bar_refresh_rate=20, 
            default_root_dir=config.LOG_DIR,
            logger=logger,
            callbacks=[checkpoint_callback],
            gpus=1,
            val_check_interval=config.TRAINER.EVAL_INTERVAL # how oftern validation
        )

    def train(self):
        self.trainer.fit(self.model, self.train_loader, self.valid_loader)

    def inference(self, ckpt_path):
        model = self.model.load_from_checkpoint(ckpt_path, config=self.model_config)
        results = self.trainer.test(model=model, dataloaders=self.test_loader, verbose=True)


