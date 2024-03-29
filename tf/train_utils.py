import os
import pathlib
import nltk

from typing import Dict
from glob import glob

import tensorflow as tf

from transformer.model import Transformer
from transformer.data import load_pten_tokenizers, load_pten_data, make_batches
from transformer.utils import CustomSchedule, masked_loss, masked_accuracy, Translator

from quantize_utils import CheckpointQuantizer, load_quantized_weights

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


def test_model(test_examples, tokenizers, model, ckpt_dir, num_tc=200, quantize=False):    
    if not quantize:
        latest = tf.train.latest_checkpoint(ckpt_dir)
        model.load_weights(latest)
        output_filepath = "output.txt"
        ref_filepath = "ref.txt"
    else:
        latest = latest_quantized_model_ckpt(ckpt_dir)
        load_quantized_weights(model, latest)
        output_filepath = "output.q.txt"
        ref_filepath = "ref.q.txt"
    
    translator = Translator(tokenizers, model)
    refs = []
    hyps = []
    
    with open(output_filepath, "w") as out_f, open(ref_filepath, "w") as ref_f:
        for i, (pt_example, en_example) in enumerate(test_examples):
            if i >= num_tc:
                break    
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

def latest_quantized_model_ckpt(quantize_dir):
    candidates = sorted(glob(quantize_dir + "/*.bcq.npy"))
    if not candidates:
        return None
    return candidates[-1]

def train(hparams: Dict, pretrain_dir: str = "trained_models/pretrained", quantize_dir: str = "trained_models/quantized", epochs: int = 26, quantize: bool = False):
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
        # Pretraining
        latest = tf.train.latest_checkpoint(pretrain_dir)
        if latest:
            model.load_weights(latest)
            initial_epoch = int(latest[-4:])
            print(
                f"Pre-training will initialize from previous checkpoint: {latest}")

        callback_cls = tf.keras.callbacks.ModelCheckpoint
        print(f"Pre-training. Checkpoints will be saved to '{ckpt_dir}'")
    else:
        # Quantization
        ckpt_dir = quantize_dir
        latest = latest_quantized_model_ckpt(quantize_dir)
        # If this is the start of quantization - continue from pretrained weigths
        if not latest:
            latest = tf.train.latest_checkpoint(pretrain_dir)
            if not latest:
                raise RuntimeError(
                    f"Cannot quantize model, no pretrained weights found in: {pretrain_dir}")
            # Loads pretrained weights
            model.load_weights(latest)
            print(
                f"Quantization will initialize from finetuned checkpoint in: {latest}")
        else:
            # If previous ckpt from quantization exists - continue from last quantized checkpoint.
            initial_epoch = int(latest.replace(".bcq.npy", "")[-4:])
            load_quantized_weights(model, latest)
            print(
                f"Continuing quantization from latest checkpoint in: {latest} @ epoch {initial_epoch}")

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
            validation_data=val_batches,
            callbacks=[ckpt_callback],
        )

    bleu_score = test_model(test_examples, tokenizers, model, ckpt_dir, quantize=quantize)
    print(f"Test BLEU: {bleu_score}")
