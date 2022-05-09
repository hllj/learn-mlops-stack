import os
import torch
import hydra
import wandb
import logging

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf

from data import DataModule
from model import ColaModel


logger = logging.getLogger(__name__)

class SamplesVisualisationLogger(pl.Callback):
    def __init__(self, datamodule):
        super().__init__()

        self.datamodule = datamodule

    def on_validation_end(self, trainer, pl_module):
        val_batch = next(iter(self.datamodule.val_dataloader()))
        sentences = val_batch["sentence"]

        outputs = pl_module(val_batch["input_ids"], val_batch["attention_mask"])
        preds = torch.argmax(outputs.logits, 1)
        labels = val_batch["label"]

        df = pd.DataFrame(
            {"Sentence": sentences, "Label": labels.numpy(), "Predicted": preds.numpy()}
        )

        wrong_df = df[df["Label"] != df["Predicted"]]
        trainer.logger.experiment.log(
            {
                "examples": wandb.Table(dataframe=wrong_df, allow_mixed_types=True),
                "global_step": trainer.global_step,
            }
        )

@hydra.main(config_path="./configs", config_name="config")
def main(cfg):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    logger.info(f"Using the model:{cfg.model.name}")
    logger.info(f"Using the tokenizer:{cfg.model.tokenizer}")
    cola_data = DataModule(
        cfg.model.tokenizer, cfg.processing.batch_size, cfg.processing.max_length
    )
    cola_model = ColaModel()
    root_dir = hydra.utils.get_original_cwd()
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(root_dir, cfg.training.checkpoint.checkpoint_path),
        filename=cfg.training.checkpoint.checkpoint_filename,
        monitor="valid/loss",
        mode="min",
        save_top_k=cfg.training.checkpoint.save_top_k,
        save_last=cfg.training.checkpoint.save_last,
    )

    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", patience=3, verbose=True, mode="min"
    )
    wandb_logger = WandbLogger(
        project=cfg.experiment.project,
        entity=cfg.experiment.entity,
        group=cfg.experiment.group,
        name=f"{cfg.experiment.name}_{'%03d' % cfg.experiment.id}",
    )
    trainer = pl.Trainer(
        # gpus=1,
        # fast_dev_run=False,
	    max_epochs=cfg.training.trainer.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=cfg.training.trainer.log_every_n_steps,
        deterministic=cfg.training.trainer.deterministic,
        limit_train_batches=cfg.training.trainer.limit_train_batches,
        limit_val_batches=cfg.training.trainer.limit_val_batches
    )
    trainer.fit(cola_model, cola_data)
    wandb.finish()


if __name__ == "__main__":
    main()
