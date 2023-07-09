import os
import pathlib
import nltk

from typing import Dict

import tensorflow as tf

from transformer.model import Transformer
from transformer.data import load_pten_tokenizers, load_pten_data, make_batches
from transformer.utils import CustomSchedule, masked_loss, masked_accuracy, Translator

from quantize_utils import CheckpointQuantizer

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


def test_model(test_examples, tokenizers, model, ckpt_dir, quantize=False):
    latest = tf.train.latest_checkpoint(ckpt_dir)
    model.load_weights(latest)
    translator = Translator(tokenizers, model)
    if not quantize:
        output_filepath = "output.txt"
        ref_filepath = "ref.txt"
    else:
        output_filepath = "output.q.txt"
        ref_filepath = "ref.q.txt"
    refs = []
    hyps = []
    with open(output_filepath, "w") as out_f, open(ref_filepath, "w") as ref_f:
        for pt_example, en_example in test_examples:
            en_output, _, _ = translator(pt_example)
            pt_example = pt_example.numpy().decode('utf-8').strip()
            en_output = en_output.numpy().decode('utf-8').strip()
            en_example = en_example.numpy().decode('utf-8').strip()
            out_f.write(f"{en_output}\n")
            ref_f.write(f"{en_example}\n")
            refs.append([en_example.split()])
            hyps.append(en_output.split())
            print(f"PT: {pt_example}\nEN: {en_example}\nOUT: {en_output}")
    print(
        f"Testing, results and references are written to {output_filepath} and {ref_filepath}"
    )
    bleu_score = nltk.translate.bleu_score.corpus_bleu(refs, hyps)
    return bleu_score


def train(hparams: Dict, pretrain_dir: str = "trained_models/pretrained", quantize_dir: str = "trained_models/quantized", epochs: int = 30, quantize: bool = False):
    test_examples, tokenizers, train_batches, val_batches = prepare_dataset()

    model = prepare_model(hparams, tokenizers)

    learning_rate = CustomSchedule(hparams["d_model"])
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )
    model.compile(
        loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy]
    )

    initial_epoch = 0
    ckpt_dir = pretrain_dir
    if not quantize:
        latest = tf.train.latest_checkpoint(pretrain_dir)
        if latest:
            model.load_weights(latest)
            initial_epoch = int(latest[-4:])
            print(
                f"Pre-training will initialize from previous checkpoint: {latest}")

        callback_cls = tf.keras.callbacks.ModelCheckpoint
        print(f"Pre-training. Checkpoints will be saved to '{ckpt_dir}'")
    else:
        ckpt_dir = quantize_dir
        # TODO: If previous ckpt from quantization exists - continue from last quantized checkpoint.
        latest = tf.train.latest_checkpoint(quantize_dir)
        
        # If this is the start of quantization - continue from pretrained weigths
        if not latest:
            latest = tf.train.latest_checkpoint(pretrain_dir)
            if not latest:
                raise RuntimeError(
                    "Cannot quantize model, no pretrained weights found in: {pretrain_dir}")
            print(
                f"Quantization will initialize from finetuned checkpoint in: {latest}")
        else:
            initial_epoch = int(latest[-4:])
            print(
                f"Continuing quantization from latest checkpoint in: {latest} @ epoch {initial_epoch}")

        # Loads either pretrained or finetuned weights
        model.load_weights(latest)

        callback_cls = CheckpointQuantizer
        print(f"Quantization - checkpoints will be saved to '{ckpt_dir}'")

    ckpt_callback = callback_cls(
        filepath=os.path.join(ckpt_dir, "pten.ckpt.ep{epoch:04d}"),
        save_freq="epoch",
        save_weights_only=True,
        monitor="masked_accuracy",
        mode="max",
        save_best_only=True,
    )
    if quantize:
        ckpt_callback.validation_data = val_batches

    if initial_epoch < epochs:
        model.fit(
            train_batches,
            initial_epoch=initial_epoch,
            epochs=epochs,
            steps_per_epoch=800, # NOTE: pNR equivalent
            validation_data=val_batches,
            callbacks=[ckpt_callback],
        )

    bleu_score = test_model(test_examples, tokenizers, model, ckpt_dir)
    print(f"Test BLEU: {bleu_score}")
