import argparse
import pathlib
import json

from typing import Dict

import tensorflow as tf

from transformer.model import Transformer
from transformer.data import load_pten_tokenizers, load_pten_data, make_batches
from transformer.utils import CustomSchedule, masked_loss, masked_accuracy, Translator

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()


def train(hparams: Dict):
    train_examples, val_examples, test_examples = load_pten_data()
    tokenizers = load_pten_tokenizers()
    train_batches = make_batches(train_examples, tokenizers)
    val_batches = make_batches(val_examples, tokenizers)

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

    ckpt_filepath = "trained_models/pretrained/pten_best.ckpt"
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_filepath,
        save_weights_only=True,
        monitor="masked_accuracy",
        mode="max",
        save_best_only=True,
    )
    print(f"Training, checkpoints will be saved to {ckpt_filepath}")
    transformer.fit(
        train_batches,
        epochs=20,
        validation_data=val_batches,
        callbacks=[ckpt_callback],
    )

    transformer.load_weights(ckpt_filepath)
    translator = Translator(tokenizers, transformer)
    output_filepath = "output.detok.txt"
    ref_filepath = "ref.detok.txt"
    print(
        "Testing, results and references will be saved to {output_filepath} and {ref_filepath}"
    )
    with open(output_filepath, "w") as out_f, open(ref_filepath, "w") as ref_f:
        for pt_example, en_example in test_examples:
            en_output, _, _ = translator(pt_example)
            out_f.write(f"{en_output.numpy().decode('utf-8').strip()}\n")
            ref_f.write(f"{en_example.strip()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="BCQ TF example")
    parser.add_argument("--hparams", default=SCRIPT_DIR / "hparams/small.json")
    args = parser.parse_args()
    with open(args.hparams, "r", encoding="utf-8") as f:
        hparams_dict = json.load(f)
    train(hparams_dict)
