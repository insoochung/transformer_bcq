"""Based off: https://www.tensorflow.org/text/tutorials/nmt_with_attention"""

import tensorflow_datasets as tfds
import tensorflow as tf

import tensorflow_text


def load_pten_data(data_name="ted_hrlr_translate/pt_to_en"):
    examples, metadata = tfds.load(data_name, with_info=True, as_supervised=True)

    return examples["train"], examples["validation"]


def load_pten_tokenizers(model_name="ted_hrlr_translate_pt_en_converter"):
    tf.keras.utils.get_file(
        f"{model_name}.zip",
        f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
        cache_dir=".",
        cache_subdir="",
        extract=True,
    )
    tokenizers = tf.saved_model.load(model_name)
    return tokenizers


def prepare_batch(pt, en, tokenizers, max_tokens=128):
    pt = tokenizers.pt.tokenize(pt)  # Output is ragged.
    pt = pt[:, :max_tokens]  # Trim to max_tokens.
    pt = pt.to_tensor()  # Convert to 0-padded dense Tensor

    en = tokenizers.en.tokenize(en)
    en = en[:, : (max_tokens + 1)]
    en_inputs = en[:, :-1].to_tensor()  # Drop the [END] tokens
    en_labels = en[:, 1:].to_tensor()  # Drop the [START] tokens

    return (pt, en_inputs), en_labels


def make_batches(ds, tokenizers, buffer_size=20000, batch_size=64, max_tokens=128):
    return (
        ds.shuffle(buffer_size)
        .batch(batch_size)
        .map(
            lambda pt, en: prepare_batch(pt, en, tokenizers, max_tokens),
            tf.data.AUTOTUNE,
        )
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
