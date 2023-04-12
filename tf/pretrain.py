import argparse
import pathlib
import json

from typing import Dict

import tensorflow as tf

from transformer.model import Transformer
from transformer.data import load_pten_tokenizers, load_pten_data, make_batches
from transformer.utils import CustomSchedule, masked_loss, masked_accuracy

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()


def train(hparams: Dict):
    train_examples, val_examples = load_pten_data()
    tokenizers = load_pten_tokenizers()
    hparams["input_vocab_size"] = tokenizers.pt.get_vocab_size().numpy()
    hparams["target_vocab_size"] = tokenizers.en.get_vocab_size().numpy()

    transformer = Transformer(**hparams)
    learning_rate = CustomSchedule(hparams["d_model"])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )
    transformer.compile(
        loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy]
    )

    train_batches = make_batches(train_examples, tokenizers)
    val_batches = make_batches(val_examples, tokenizers)
    transformer.fit(train_batches, epochs=20, validation_data=val_batches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="BCQ TF example")
    parser.add_argument("--hparams", default=SCRIPT_DIR / "hparams/small.json")
    args = parser.parse_args()
    with open(args.hparams, "r", encoding="utf-8") as f:
        hparams_dict = json.load(f)
    train(hparams_dict)
