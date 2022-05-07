import torch
from torch.utils.data import DataLoader
import torchaudio.transforms as T
import torchaudio.functional as F
from model.conformer import Conformer
import pytorch_lightning as pl
import torchmetrics
import sys


class LibrispeechDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_set,
        val_set,
        test_set,
        predict_set,
        encode_string,
        batch_size: int = 32,
        dataloader_numworkers: int = 4,
    ):
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.predict_set = predict_set
        self.batch_size = batch_size
        self.encode_string = encode_string
        self.num_workers = dataloader_numworkers

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_set,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers
        )

    def _collate_fn(self, batch):
        inputs = [i[0] for i in batch]
        input_lengths = torch.IntTensor([i[1] for i in batch])
        targets = [self.encode_string(i[2]) for i in batch]
        target_lengths = torch.IntTensor([i[3] for i in batch])

        inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True).to(
            dtype=torch.int
        )

        return inputs, input_lengths, targets, target_lengths


class ConformerModule(pl.LightningModule):
    def __init__(self, cfg, blank=0, text_preprocess=None, batch_size=4):
        super().__init__()
        self.cfg = cfg
        self.lr = cfg.training.lr
        self.conformer = Conformer(**cfg.model)
        self.cal_loss = T.RNNTLoss(blank=blank)
        self.cal_wer = torchmetrics.WordErrorRate()
        self.text_preprocess = text_preprocess
        self.save_hyperparameters()

    def forward(self, inputs, input_lengths):
        return self.conformer.recognize(inputs, input_lengths)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, **self.cfg.optim)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, **self.cfg.sched
        )
        return [optimizer], [lr_scheduler]

    def get_batch(self, batch):
        inputs, input_lengths, targets, target_lengths = batch

        batch_size = inputs.size(0)

        zeros = torch.zeros((batch_size, 1)).to(device=self.device)
        compute_targets = torch.cat((zeros, targets), dim=1).to(
            device=self.device, dtype=torch.int
        )
        compute_target_lengths = (target_lengths + 1).to(device=self.device)

        return (
            inputs,
            input_lengths,
            targets,
            target_lengths,
            compute_targets,
            compute_target_lengths,
        )

    def training_step(self, batch, batch_idx):
        (
            inputs,
            input_lengths,
            targets,
            target_lengths,
            compute_targets,
            compute_target_lengths,
        ) = self.get_batch(batch)

        outputs, output_lengths = self.conformer(
            inputs, input_lengths, compute_targets, compute_target_lengths
        )

        loss = self.cal_loss(outputs, targets, output_lengths, target_lengths)

        self.log("train_loss", loss)
        self.log("lr", self.lr)

        return loss

    def validation_step(self, batch, batch_idx):
        (
            inputs,
            input_lengths,
            targets,
            target_lengths,
            compute_targets,
            compute_target_lengths,
        ) = self.get_batch(batch)

        outputs, output_lengths = self.conformer(
            inputs, input_lengths, compute_targets, compute_target_lengths
        )

        loss = self.cal_loss(outputs, targets, output_lengths, target_lengths)

        predicts = self.forward(inputs, input_lengths)
        predicts = [self.text_preprocess.int2text(sent) for sent in predicts]
        targets = [self.text_preprocess.int2text(sent) for sent in targets]

        list_wer = torch.tensor(
            [self.cal_wer(i, j).item() for i, j in zip(predicts, targets)]
        )
        wer = torch.mean(list_wer)

        self.log_output(predicts[0], targets[0], wer)

        self.log("val_loss", loss)
        self.log("val_batch_wer", wer)

        return loss, wer

    def test_step(self, batch, batch_idx):
        (
            inputs,
            input_lengths,
            targets,
            target_lengths,
            compute_targets,
            compute_target_lengths,
        ) = self.get_batch(batch)

        outputs, output_lengths = self.conformer(
            inputs, input_lengths, compute_targets, compute_target_lengths
        )

        loss = self.cal_loss(outputs, targets, output_lengths, target_lengths)

        predicts = self.forward(inputs, input_lengths)
        predicts = [self.text_preprocess.int2text(sent) for sent in predicts]
        targets = [self.text_preprocess.int2text(sent) for sent in targets]

        list_wer = torch.tensor(
            [self.cal_wer(i, j).item() for i, j in zip(predicts, targets)]
        )
        wer = torch.mean(list_wer)

        self.log_output(predicts[0], targets[0], wer)

        self.log("test_loss", loss)
        self.log("test_batch_wer", wer)

        return loss, wer

    def log_output(self, predict, target, wer):
        print("=" * 50)
        print("Sample Predicts: ", predict)
        print("Sample Targets:", target)
        print("Mean WER:", wer)
        print("=" * 50)
