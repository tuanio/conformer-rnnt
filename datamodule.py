import torch
from torch.utils.data import DataLoader, Dataset
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
            num_workers=self.num_workers,
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


class VivosDataModule(pl.LightningDataModule):
    def __init__(
        self,
        trainset: Dataset,
        testset: Dataset,
        text_process: TextProcess,
        batch_size: int,
    ):
        super().__init__()

        self.trainset = trainset
        self.valset = testset
        self.testset = testset
        self.batch_size = batch_size

        self.text_process = text_process

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def _collate_fn(self, batch):
        """
        Take feature and input, transform and then padding it
        """
        specs = [i[0] for i in batch]
        input_lengths = torch.LongTensor([i.size(0) for i in specs])
        trans = [i[1] for i in batch]
        target_lengths = torch.LongTensor([len(s) for s in trans])

        # batch, time, feature
        specs = torch.nn.utils.rnn.pad_sequence(specs, batch_first=True)
        specs = specs.unsqueeze(1)  # batch, channel, time, feature

        trans = [self.text_process.text2int(s) for s in trans]
        trans = torch.nn.utils.rnn.pad_sequence(trans, batch_first=True).to(
            dtype=torch.long
        )

        return specs, input_lengths, trans, target_lengths