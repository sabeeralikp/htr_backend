"""
The module contains definitions related to a Transformer model for handwritten text recognition.

This module includes the following components:
- `PositionalEncoding`: A module that applies positional encoding to the input tensor.
- `count_parameters`: A function that counts the number of trainable parameters in a model.
- `TransformerModel`: A class that defines the Transformer model for handwritten text recognition.
- `process_image`: A function that preprocesses an image for input to the model.
- `labels_to_text`: A function that converts a sequence of label indices to a string.
- `prediction`: A function that generates a prediction for an input image using the model.
- `char_error_rate`: A function that calculates the character error rate between two sequences.
- `test`: A function that performs testing and evaluation on a dataset.
- `pred_with_image`: A function that generates a prediction for an input image.

Note:
- The `TransformerModel` class assumes the presence of packages like:
    `torch`, `torch.nn`, `torchvision`, `math`, `os`, `cv2`, `numpy`, `string`, and `editdistance`.
- The module also assumes the availability of a trained model file and necessary data files.
- The provided code includes various utility functions and dependencies required for handwritten text recognition.
- Additional documentation and usage examples for individual functions and classes can be found within the code.
"""

import torch
from torch.cuda import is_available
import torch.nn as nn
import torchvision.models as models
import math
import os
import cv2
import numpy as np
import string
import editdistance


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.

    Args:
        d_model (int): The dimension of the input features.
        dropout (float, optional): Dropout rate to apply to the encoded features (default: 0.1).
        max_len (int, optional): The maximum length of the input sequence (default: 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: any) -> any:
        x = x + self.scale * self.pe[: x.size(0), :]
        return self.dropout(x)


def count_parameters(model: any) -> any:
    """
    Count the total number of trainable parameters in a given model.

    Args:
        model (torch.nn.Module): The model to count the parameters for.

    Returns:
        int: The total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TransformerModel(nn.Module):
    """
    Transformer-based model for a specific task.

    Args:
        bb_name (str): Name of the backbone model, e.g., 'resnet50'.
        outtoken (int): Number of output tokens.
        hidden (int): Dimensionality of the hidden state.
        enc_layers (int): Number of encoder layers in the transformer (default: 1).
        dec_layers (int): Number of decoder layers in the transformer (default: 1).
        nhead (int): Number of attention heads in the transformer (default: 1).
        dropout (float): Dropout rate (default: 0.1).
        pretrained (bool): Whether to use pretrained weights for the backbone model (default: False).
    """

    def __init__(
        self,
        bb_name: any,
        outtoken: any,
        hidden: any,
        enc_layers: int = 1,
        dec_layers: int = 1,
        nhead: int = 1,
        dropout: float = 0.1,
        pretrained: bool = False,
    ) -> any:
        super(TransformerModel, self).__init__()
        self.backbone = models.__getattribute__(bb_name)(pretrained=pretrained)
        self.backbone.fc = nn.Conv2d(2048, int(hidden / 2), 1)

        self.pos_encoder = PositionalEncoding(hidden, dropout)
        self.decoder = nn.Embedding(outtoken, hidden)
        self.pos_decoder = PositionalEncoding(hidden, dropout)
        self.transformer = nn.Transformer(
            d_model=hidden,
            nhead=nhead,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="relu",
        )

        self.fc_out = nn.Linear(hidden, outtoken)
        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None

        print("backbone: {}".format(bb_name))
        print("layers: {}".format(enc_layers))
        print("heads: {}".format(nhead))
        print("dropout: {}".format(dropout))
        print(f"{count_parameters(self):,} trainable parameters")

    def generate_square_subsequent_mask(self, sz: any) -> any:
        """
        Generate a square subsequent mask.

        Args:
            sz (int): Size of the mask.

        Returns:
            torch.Tensor: Square subsequent mask.
        """
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def make_len_mask(self, inp: any) -> any:
        """
        Make a length mask.

        Args:
            inp (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Length mask.
        """
        return (inp == 0).transpose(0, 1)

    def forward(self, src: any, trg: any) -> any:
        """
        params
        ---
        src : Tensor [64, 3, 64, 256] : [B,C,H,W]
            B - batch, C - channel, H - height, W - width
        trg : Tensor [13, 64] : [L,B]
            L - max length of label
        Returns:
            torch.Tensor: Output tensor.
        """
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(
                trg.device
            )
        x = self.backbone.conv1(src)

        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        # [64, 2048, 2, 8] : [B,C,H,W]
        x = self.backbone.layer4(x)
        # [64, 256, 2, 8] : [B,C,H,W]
        x = self.backbone.fc(x)
        # [64, 8, 256, 2] : [B,W,C,H]
        x = x.permute(0, 3, 1, 2)
        # [64, 8, 512] : [B,W,CH]
        x = x.flatten(2)
        # [8, 64, 512] : [W,B,CH]
        x = x.permute(1, 0, 2)
        src_pad_mask = self.make_len_mask(x[:, :, 0])
        # [8, 64, 512]
        src = self.pos_encoder(x)
        trg_pad_mask = self.make_len_mask(trg)
        trg = self.decoder(trg)
        trg = self.pos_decoder(trg)
        # [13, 64, 512] : [L,B,CH]
        output = self.transformer(
            src,
            trg,
            src_mask=self.src_mask,
            tgt_mask=self.trg_mask,
            memory_mask=self.memory_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=trg_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        # [13, 64, 92] : [L,B,H]
        output = self.fc_out(output)

        return output


PATH_TEST_DIR = "words/val/"
PATH_TEST_LABELS = "words/val.csv"

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cyrillic = ["EOS"]
s = ".ംഃഅആഇഈഉഊഋഎഏഐഒഓഔകഖഗഘങചഛജഝഞടഠഡഢണതഥദധനപഫബഭമയരറലളഴവശഷസഹാിീുൂൃെേൈൊോ്ൗൺൻർൽൾ"
cyrillic[:0] = s
cyrillic.insert(0, "SOS")
cyrillic.insert(0, "PAD")
char2idx = {char: idx for idx, char in enumerate(cyrillic)}
idx2char = {idx: char for idx, char in enumerate(cyrillic)}
hidden = 512
enc_layers = 2
dec_layers = 2
nhead = 4
dropout = 0.0
height = 64
width = 256

model = TransformerModel(
    "resnet50",
    len(cyrillic),
    hidden=hidden,
    enc_layers=enc_layers,
    dec_layers=dec_layers,
    nhead=nhead,
    dropout=dropout,
).to(dev)

model.load_state_dict(
    torch.load("htr_api/model/0.26470837907215855.pth", map_location=dev)
)


def process_image(img: any) -> any:
    """
    params:
    ---
    img : np.array
    returns
    ---
    img : np.array
    """
    w, h, _ = img.shape
    new_w = height
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h, _ = img.shape

    img = img.astype("float32")

    new_h = width
    if h < new_h:
        add_zeros = np.full((w, new_h - h, 3), 255)
        img = np.concatenate((img, add_zeros), axis=1)

    if h > new_h:
        img = cv2.resize(img, (new_h, new_w))

    return img


def labels_to_text(s: any, idx2char: any) -> any:
    """
    params
    ---
    idx2char : dict
        keys : int
            indicies of characters
        values : str
            characters
    returns
    ---
    S : str
    """
    S = "".join([idx2char[i] for i in s])
    if S.find("EOS") == -1:
        return S
    else:
        return S[: S.find("EOS")]


def prediction(model: any, test_dir: any, char2idx: any, idx2char: any) -> any:
    """
    Generate a prediction using the trained model.

    Args:
        model (nn.Module): Trained model.
        test_dir (any): Directory of the test data.
        char2idx (any): Mapping from characters to indices.
        idx2char (any): Mapping from indices to characters.

    Returns:
        any: Predicted output.
    """
    # os.makedirs("/output", exist_ok=True)
    model.eval()

    with torch.no_grad():
        img = test_dir
        if img is None:
            print(test_dir)
        img = process_image(img).astype("uint8")
        img = img / img.max()
        img = np.transpose(img, (2, 0, 1))
        src = torch.FloatTensor(img).unsqueeze(0).to(dev)
        out_indexes = [
            char2idx["SOS"],
        ]
        for _ in range(100):
            trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(dev)

            output = model(src, trg_tensor)
            out_token = output.argmax(2)[-1].item()
            out_indexes.append(out_token)
            if out_token == char2idx["EOS"]:
                break
        pred = labels_to_text(out_indexes[1:], idx2char)

    return pred


def char_error_rate(p_seq1: any, p_seq2: any) -> any:
    """
    params
    ---
    p_seq1 : str
    p_seq2 : str
    returns
    ---
    cer : float
    """
    p_vocab = set(p_seq1 + p_seq2)
    p2c = dict(zip(p_vocab, range(len(p_vocab))))
    c_seq1 = [chr(p2c[p]) for p in p_seq1]
    c_seq2 = [chr(p2c[p]) for p in p_seq2]
    return editdistance.eval("".join(c_seq1), "".join(c_seq2)) / max(
        len(c_seq1), len(c_seq2)
    )


def test(
    model: any,
    image_dir: any,
    label_dir: any,
    char2idx: any,
    idx2char: any,
    case: bool = True,
    punct: bool = False,
) -> any:
    """
    Evaluate the performance of a model on a test dataset.

    Args:
        model (nn.Module): Trained model.
        image_dir (any): Directory of the test images.
        label_dir (any): Directory of the corresponding labels.
        char2idx (any): Mapping from characters to indices.
        idx2char (any): Mapping from indices to characters.
        case (bool, optional): Flag indicating whether to consider case sensitivity. Defaults to True.
        punct (bool, optional): Flag indicating whether to consider punctuation. Defaults to False.

    Returns:
        any: Tuple containing character accuracy and string accuracy.
    """
    img2label = dict()
    raw = open(label_dir, "r", encoding="utf-8").read()
    temp = raw.split("\n")
    for t in temp:
        x = t.split("\t")
        if x == [""]:
            print(x)
            continue
        img2label[image_dir + x[0]] = x[1]
    preds = prediction(model, image_dir, char2idx, idx2char)
    N = len(preds)

    wer = 0
    cer = 0

    for item in preds.items():
        print(item)
        img_name = item[0]
        true_trans = img2label[image_dir + img_name]
        predicted_trans = item[1]

        if "ё" in true_trans:
            true_trans = true_trans.replace("ё", "е")
        if "ё" in predicted_trans["predicted_label"]:
            predicted_trans["predicted_label"] = predicted_trans[
                "predicted_label"
            ].replace("ё", "е")

        if not case:
            true_trans = true_trans.lower()
            predicted_trans["predicted_label"] = predicted_trans[
                "predicted_label"
            ].lower()

        if not punct:
            true_trans = true_trans.translate(str.maketrans("", "", string.punctuation))
            predicted_trans["predicted_label"] = predicted_trans[
                "predicted_label"
            ].translate(str.maketrans("", "", string.punctuation))

        if true_trans != predicted_trans["predicted_label"]:
            print("true:", true_trans)
            print("predicted:", predicted_trans)
            print(
                "cer:", char_error_rate(predicted_trans["predicted_label"], true_trans)
            )
            print("---")
            wer += 1
            cer += char_error_rate(predicted_trans["predicted_label"], true_trans)

    character_accuracy = 1 - cer / N
    string_accuracy = 1 - (wer / N)
    return character_accuracy, string_accuracy


# print(word_accur, char_accur)
def pred_with_image(image: any) -> any:
    """
    Generate a prediction for the given image.

    Args:
        image (any): Input image.

    Returns:
        any: Predicted label for the image.
    """
    pred = prediction(model, image, char2idx, idx2char)
    return pred
