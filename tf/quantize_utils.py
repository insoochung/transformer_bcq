import copy

import tensorflow as tf
import numpy as np


def collect_transformer_weights(model):
    ws = {}
    encoder = model.encoder
    decoder = model.decoder

    ws["encoder/pos_embedding/embedding"] = encoder.pos_embedding.embedding.get_weights()
    for i, layer in enumerate(encoder.enc_layers):
        ws[f"encoder/layer_{i}/self_attention/mha/"] = layer.self_attention.mha.get_weights()
        ws[f"encoder/layer_{i}/self_attention/mha/layernorm/"] = layer.self_attention.layernorm.get_weights()
        ws[f"encoder/layer_{i}/ffn/seq"] = layer.ffn.seq.get_weights()
        ws[f"encoder/layer_{i}/ffn/layer_norm"] = layer.ffn.layer_norm.get_weights()

    ws["decoder/pos_embedding/embedding"] = decoder.pos_embedding.embedding.get_weights()
    for i, layer in enumerate(decoder.dec_layers):
        ws[f"decoder/layer_{i}/causal_self_attention/mha/"] = layer.causal_self_attention.mha.get_weights()
        ws[f"decoder/layer_{i}/causal_self_attention/mha/layernorm/"] = layer.causal_self_attention.layernorm.get_weights()
        ws[f"decoder/layer_{i}/cross_attention/mha/"] = layer.cross_attention.mha.get_weights()
        ws[f"decoder/layer_{i}/cross_attention/mha/layernorm/"] = layer.cross_attention.layernorm.get_weights()
        ws[f"decoder/layer_{i}/ffn/seq"] = layer.ffn.seq.get_weights()
        ws[f"decoder/layer_{i}/ffn/layer_norm"] = layer.ffn.layer_norm.get_weights()

    ws["final_layer"] = model.final_layer.get_weights()

    # Check if we've fetched all weights
    num_layers = 0
    for v in ws.values():
        num_layers += len(v)

    if len(model.get_weights()) != num_layers:
        raise RuntimeError(
            "Number of collected weights does not match the number of weight matrices in the model")

    return ws


def quantize_transformer_weights(ws):
    qs = {}
    for k, _ws in ws.items():
        transpose = False
        if "final_layer" in k:
            transpose = True # final_layer is transposed {before quantization, after dequantization}
        
        print(k)
        print(f"ws: {[_w.shape for _w in _ws]}")

        if "layer" in k and "norm" in k:
            qs[k] = _ws # layer norm's are not quantized
            continue

        _qs = []
        for i, _w in enumerate(_ws):
            if i % 2: # odd entries are biases
                _qs.append(_w) # biases are not quantized
                continue
            _qs.append(QuantizedWeights(_w, transpose=transpose))
        qs[k] = _qs
        print(f"qs: {[_q.shape for _q in _qs]}")
    assert 0

def apply_quantized_weights(model, ws, qs):
    assert 0


class QuantizedWeights():
    def __init__(self, weight, num_bits=3, transpose=False, test_integrity=True):
        self.test_integrity = test_integrity
        self.num_bits = num_bits
        self.transpose = transpose
        self.w_orig_shape = list(weight.shape)

        if tf.is_tensor(weight):
            weight = weight.numpy()
        _weight = weight.copy() # [R, C]
        if len(_weight.shape) < 2:
            raise RuntimeError("QuantizedWeight class cannot handle 1-D vectors")
        elif len(_weight.shape) > 2:
            _weight = tf.reshape(_weight, [-1, self.w_orig_shape[-1]])
        
        if self.transpose:
            _weight = tf.transpose(_weight)
        self.w_2d_shape = list(_weight.shape)
        self.quantize(_weight)

    def quantize(self, weight):
        # Quantize
        num_rows, num_cols = weight.shape
        num_bits = self.num_bits
        if tf.is_tensor(weight):
            weight = weight.numpy()
        res_mat = weight.copy() # [R, C]
        self.bits = np.zeros([num_rows, num_cols, num_bits]) # [R, C, Q]
        self.alphas = np.zeros([num_rows, num_bits]) # [R, Q], 1 scalar value per row.
        
        # k-bit quantization using greedy approximation as described in https://openreview.net/pdf?id=S19dR9x0b
        for r in range(num_rows):
            res_vec = res_mat[r] # get row vector, set as initial residual
            for q in range(num_bits):
                _bs = (res_vec >= 0).astype(np.float32) * 2. - 1. # similar to np.sign, but 0s are set as 1.
                _as = np.linalg.norm(res_vec, ord=1) / num_cols
                self.bits[r, :, q] = _bs # a row vector of length num_cols
                self.alphas[r, q] = _as # single scalar value
                res_vec -= _as * _bs

        self.bits = np.packbits(self.bits >= 0., axis=1) # [R, C // 8, Q] of uint8
        self.shape = {"bits": self.bits.shape, "alphas": self.alphas.shape, "orig": self.w_orig_shape}
        
        if not self.test_integrity:
            return
        deq_w = self.dequantize()
        deq_w = np.reshape(deq_w, [-1, deq_w.shape[-1]])
        if self.transpose:
            deq_w = deq_w.T
        if not np.allclose(weight, res_mat + deq_w, rtol=1e-4, atol=1e-7):
            raise RuntimeError(f"Sum of residual and quantized-dequantized does not add up to the original weight values - shape: {self.shape}")
        

    def dequantize(self):
        weight = np.zeros(self.w_2d_shape)
        
        # Dequantize
        bits_unpacked = np.unpackbits(self.bits, axis=1) * 2. - 1.
        for q in range(self.num_bits):
            weight += bits_unpacked[:,:,q] * np.expand_dims(self.alphas[:, q], -1) # [R, C] * [R, 1] = [R, C]
        if self.transpose:
            weight = weight.T
        weight = np.reshape(weight, self.w_orig_shape)

        return weight


class CheckpointQuantizer(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.save_freq != "epoch":
            raise RuntimeError(
                "CheckpointQuantizer only supports epoch frequency")

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        ws = collect_transformer_weights(self.model)
        qs = quantize_transformer_weights(ws)
        apply_quantized_weights(self.model, ws, qs)
        assert 0
        self._save_model(epoch=epoch, batch=None, logs=logs)

    def _save_model(self):
        pass
