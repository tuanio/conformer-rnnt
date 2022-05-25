import torch
import os
from torch.utils.data import Dataset
from pathlib import Path
import torchaudio


class ComposeDataset(Dataset):
    """
        this dataset aim to load:
            - vivos
            - vin big data
            - vietnamese podcasts
    """

    def __init__(
        self,
        vivos_root: str = "",
        vivos_subset: str = "train",
        vlsp_root: str = "",
        podcasts_root: str = "",
        n_fft: int = 400,
    ):

        super().__init__()
        self.feature_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft)

        self.walker = self.init_vivos(vivos_root, vivos_subset)

        if vivos_subset == "train":
            self.walker += self.init_vlsp(vlsp_root)

    def init_vivos(self, root, subset):
        assert subset in ["train", "test"], "subset not found"

        path = os.path.join(root, subset)
        waves_path = os.path.join(path, "waves")
        transcript_path = os.path.join(path, "prompts.txt")

        # walker oof
        walker = list(Path(waves_path).glob("*/*"))

        with open(transcript_path, "r", encoding="utf-8") as f:
            transcripts = f.read().strip().split("\n")
            transcripts = [line.split(" ", 1) for line in transcripts]
            filenames = [i[0] for i in transcripts]
            trans = [i[1] for i in transcripts]
            transcripts = dict(zip(filenames, trans))

        def load_el_from_path(filepath):
            filename = filepath.name.split(".")[0]
            trans = transcripts[filename].lower()
            return (filepath, trans)

        walker = [load_el_from_path(filepath) for filepath in walker]
        return walker

    def init_vlsp(self, root):
        walker = list(Path(root).glob("*.wav"))

        def load_el_from_path(filepath):
            filename = filepath.name.split(".")[0] + ".txt"
            with open(Path(root) / filename, "r", encoding="utf-8") as f:
                trans = f.read().strip().lower()
                trans.replace("<unk>", "").strip()
            return filepath, trans

        walker = [load_el_from_path(filepath) for filepath in walker]

        return walker

    def __len__(self):
        return len(self.walker)

    def __getitem__(self, idx):
        filepath, trans = self.walker[idx]

        wave, sr = torchaudio.load(filepath)

        specs = self.feature_transform(wave)  # channel, feature, time
        specs = specs.permute(0, 2, 1)  # channel, time, feature
        specs = specs.squeeze()  # time, feature

        return specs, trans
