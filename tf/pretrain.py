import os
import argparse
import pathlib
import json
import nltk

from typing import Dict

import tensorflow as tf

from transformer.model import Transformer
from transformer.data import load_pten_tokenizers, load_pten_data, make_batches
from transformer.utils import CustomSchedule, masked_loss, masked_accuracy, Translator

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()


def prepare_dataset():
    train_examples, val_examples, test_examples = load_pten_data()
    tokenizers = load_pten_tokenizers()
    train_batches = make_batches(train_examples, tokenizers)
    val_batches = make_batches(val_examples, tokenizers)
    return test_examples, tokenizers, train_batches, val_batches


def prepare_model(hparams, tokenizers):
    hparams["input_vocab_size"] = tokenizers.pt.get_vocab_size().numpy()
    hparams["target_vocab_size"] = tokenizers.en.get_vocab_size().numpy()
    model = Transformer(**hparams)
    return model


def test_model(test_examples, tokenizers, model, ckpt_dir):
    latest = tf.train.latest_checkpoint(ckpt_dir)
    model.load_weights(latest)
    translator = Translator(tokenizers, model)
    output_filepath = "output.detok.txt"
    ref_filepath = "ref.detok.txt"
    refs = []
    hyps = []
    with open(output_filepath, "w") as out_f, open(ref_filepath, "w") as ref_f:
        assert 0
        for pt_example, en_example in test_examples:
            en_output, _, _ = translator(pt_example)
            pt_example = pt_example.numpy().decode('utf-8').strip()
            en_output = en_output.numpy().decode('utf-8').strip()
            en_example = en_example.numpy().decode('utf-8').strip()
            out_f.write(f"{en_output}\n")
            ref_f.write(f"{en_example}\n")
            refs.append([[tokenizers.en.tokenize(en_example)]])
            hyps.append([tokenizers.en.tokenize(en_output)])
            print(f"PT: {pt_example}\nEN: {en_example}\nOUT: {en_output}")
    print(
        f"Testing, results and references are written to {output_filepath} and {ref_filepath}"
    )
    bleu_score = nltk.translate.bleu_score.corpus_bleu(refs, hyps)
    return bleu_score


def train(hparams: Dict):
    test_examples, tokenizers, train_batches, val_batches = prepare_dataset()

    model = prepare_model(hparams, tokenizers)

    learning_rate = CustomSchedule(hparams["d_model"])
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )
    model.compile(
        loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy]
    )

    ckpt_dir = "trained_models/pretrained"
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(ckpt_dir, "pten.ckpt.ep{epoch:04d}"),
        save_weights_only=True,
        monitor="masked_accuracy",
        mode="max",
        save_best_only=True,
    )
    print(f"Training, checkpoints will be saved to {ckpt_dir}")

    latest = tf.train.latest_checkpoint(ckpt_dir)
    initial_epoch = None
    if latest:
        model.load_weights(latest)
        initial_epoch = int(latest[-4:])
        print(f"Training will initialize from previous checkpoint: {latest}")

    model.fit(
        train_batches,
        initial_epoch=initial_epoch,
        epochs=30,
        validation_data=val_batches,
        callbacks=[ckpt_callback],
    )

    bleu_score = test_model(test_examples, tokenizers, model, ckpt_dir)
    print(f"Test BLEU: {bleu_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="BCQ TF example")
    parser.add_argument("--hparams", default=SCRIPT_DIR / "hparams/small.json")
    args = parser.parse_args()
    with open(args.hparams, "r", encoding="utf-8") as f:
        hparams_dict = json.load(f)
    train(hparams_dict)
