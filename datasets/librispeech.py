import os
import torch
import torchaudio
from pathlib import Path
import sys
import pickle


class LibriSpeechDataset(torch.utils.data.Dataset):

    ext_txt = ".trans.txt"
    ext_audio = ".flac"

    def __init__(
        self,
        data_type="train100",
        clean_path="../input/librispeech-clean/LibriSpeech/",
        other_path="../input/librispeech-500-hours/LibriSpeech/",
        n_fft=159,
    ):
        """
            data_type \in ['train100', 'train360', 'train460', 'train960', 'dev', 'test']
        """

        self.spect_func = torchaudio.transforms.Spectrogram(n_fft=n_fft)

        self.list_url = [clean_path + "train-clean-100"]

        if data_type == "train360":
            self.list_url = [clean_path + "train-clean-360"]
        elif data_type == "train460":
            self.list_url += [clean_path + "train-clean-360"]
        elif data_type == "train960":
            self.list_url += [clean_path + "train-clean-360"]
            self.list_url += [other_path + "train-clean-360"]
        elif data_type == "dev":
            self.list_url = [clean_path + "dev-clean"]
        elif data_type == "test":
            self.list_url = [clean_path + "test-clean"]

        self._walker = []
        for path in self.list_url:
            walker = [
                (str(p.stem), path) for p in Path(path).glob("*/*/*" + self.ext_audio)
            ]
            self._walker.extend(walker)
        self._walker = sorted(self._walker)

    def __len__(self):
        return len(self._walker)

    def __getitem__(self, n):
        fileid, path = self._walker[n]
        return self.load_librispeech_item(fileid, path)

    def load_librispeech_item(self, fileid, path):
        """
            transform audio pack to spectrogram
        """

        speaker_id, chapter_id, utterance_id = fileid.split("-")

        file_text = speaker_id + "-" + chapter_id + self.ext_txt
        file_text = os.path.join(path, speaker_id, chapter_id, file_text)

        fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id
        file_audio = fileid_audio + self.ext_audio
        file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)

        # Load audio
        waveform, sample_rate = torchaudio.load(file_audio)

        # Load text
        with open(file_text) as ft:
            for line in ft:
                fileid_text, transcript = line.strip().split(" ", 1)
                if fileid_audio == fileid_text:
                    break
            else:
                # Translation not found
                raise FileNotFoundError("Translation not found for " + fileid_audio)

        spectrogram = self.spect_func(waveform)
        spectrogram = spectrogram.squeeze().permute(1, 0)

        input_lengths = spectrogram.size(0)  # time
        target_lengths = len(transcript)

        return spectrogram, input_lengths, transcript, target_lengths
