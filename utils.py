import torch


class TextProcess:

    aux_vocab = ["<pad>", "<sos>", "<eos>"]

    origin_list_vocab = {
        "en": aux_vocab + list("abcdefghijklmnopqrstuvwxyz '"),
        "vi": aux_vocab
        + list(
            "abcdefghijklmnopqrstuvwxyzàáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ "
        ),
    }

    origin_vocab = {
        lang: dict(zip(vocab, range(len(vocab))))
        for lang, vocab in origin_list_vocab.items()
    }

    def __init__(self, lang):
        self.lang = lang
        assert self.lang in ["vi", "en"], "Language not found"
        self.vocab = self.origin_vocab[lang]
        self.list_vocab = self.origin_list_vocab[lang]

    def text2int(self, s: str):
        return torch.Tensor([self.vocab[i] for i in s.lower()])

    def int2text(self, s: torch.Tensor):
        return "".join([self.list_vocab[i] for i in s if i > 2])
