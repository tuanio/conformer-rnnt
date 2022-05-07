import time
from model.conformer import Conformer
import os
import logging
import sys
import argparse
import torch

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import hydra
from omegaconf import OmegaConf, DictConfig
from utils import TextProcess
from datasets.librispeech import LibriSpeechDataset
from datasets.vivos import VivosDataset
from datamodule import LibrispeechDataModule, VivosDataModule
from modelmodule import ConformerModule

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Config path")
    parser.add_argument("-cp", help="config path")  # config path
    parser.add_argument("-cn", help="config name")  # config name

    args = parser.parse_args()

    @hydra.main(config_path=args.cp, config_name=args.cn)
    def main(cfg: DictConfig):

        text_process = TextProcess(**cfg.text_process)
        cfg.model.num_classes = len(text_process.vocab)

        if cfg.datasets.dataset_selected == "vivos":
            train_set = VivosDataset(**cfg.datasets.vivos, subset="train")
            test_set = VivosDataset(**cfg.datasets.vivos, subset="test")

            dm = VivosDataModule(train_set, test_set, text_process, batch_size)

        elif cfg.datasets.dataset_selected == "librispeech":
            
            datasets_cfg = cfg.datasets.librispeech

            train_set = LibriSpeechDataset(
                data_type=datasets_cfg.train,
                clean_path=datasets_cfg.clean_path,
                other_path=datasets_cfg.other_path,
                db_path=datasets_cfg.db_path,
            )

            val_set = LibriSpeechDataset(
                data_type=datasets_cfg.val,
                clean_path=datasets_cfg.clean_path,
                other_path=datasets_cfg.other_path,
                db_path=datasets_cfg.db_path,
            )

            test_set = LibriSpeechDataset(
                data_type=datasets_cfg.test,
                clean_path=datasets_cfg.clean_path,
                other_path=datasets_cfg.other_path,
                db_path=datasets_cfg.db_path,
            )

            dm = LibrispeechDataModule(
                train_set=train_set,
                val_set=val_set,
                test_set=test_set,
                predict_set=test_set,
                encode_string=text_process.text2int,
                batch_size=cfg.training.batch_size,
                dataloader_numworkers=cfg.training.dataloader_numworkers,
            )

        model = ConformerModule(
            cfg, blank=text_process.list_vocab.index("<p>"), text_process=text_process,
        )

        tb_logger = pl.loggers.tensorboard.TensorBoardLogger(**cfg.tb_logger)

        trainer = pl.Trainer(logger=tb_logger, **cfg.trainer)

        if cfg.ckpt.train:
            print("Training model")
            if cfg.ckpt.have_ckpt:
                trainer.fit(model, datamodule=dm, ckpt_path=cfg.ckpt.ckpt_path)
            else:
                try:
                    trainer.fit(model=model, datamodule=dm)
                except Exception as e:
                    print(str(e))

            trainer.save_checkpoint("conformer_rnnt.ckpt", weights_only=True)

            # export
            try:
                input_sample = next(iter(dm.train_dataloader()))
                model.to_onnx("conformer_rnnt.onnx", input_sample, export_params=True)
            except Exception as e:
                print(str(e))

        else:
            print("Train mode turn off!")

        print("Testing model")
        if cfg.ckpt.have_ckpt:
            trainer.test(model, datamodule=dm, ckpt_path=cfg.ckpt.ckpt_path)
        else:
            trainer.test(model, datamodule=dm)

    main()
