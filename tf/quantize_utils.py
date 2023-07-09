import tensorflow as tf
import numpy as np

from transformer.model import Encoder, Decoder


def collect_transformer_weights(model):
    """Ad hoc function for retrieving layer weights from a Transformer object"""
    ws = {}
    encoder = model.encoder
    decoder = model.decoder

    ws["encoder/pos_embedding/embedding"] = encoder.pos_embedding.embedding.get_weights()
    for i, layer in enumerate(encoder.enc_layers):
        ws[f"encoder/layer_{i}/self_attention/mha"] = layer.self_attention.mha.get_weights()
        ws[f"encoder/layer_{i}/self_attention/layernorm"] = layer.self_attention.layernorm.get_weights()
        ws[f"encoder/layer_{i}/ffn/seq"] = layer.ffn.seq.get_weights()
        ws[f"encoder/layer_{i}/ffn/layer_norm"] = layer.ffn.layer_norm.get_weights()

    ws["decoder/pos_embedding/embedding"] = decoder.pos_embedding.embedding.get_weights()
    for i, layer in enumerate(decoder.dec_layers):
        ws[f"decoder/layer_{i}/causal_self_attention/mha"] = layer.causal_self_attention.mha.get_weights()
        ws[f"decoder/layer_{i}/causal_self_attention/layernorm"] = layer.causal_self_attention.layernorm.get_weights()
        ws[f"decoder/layer_{i}/cross_attention/mha"] = layer.cross_attention.mha.get_weights()
        ws[f"decoder/layer_{i}/cross_attention/layernorm"] = layer.cross_attention.layernorm.get_weights()
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
    """Quantize FP transformer weights (except bias and layer norm parameters)"""
    qs = {}
    for k, _ws in ws.items():
        transpose = False
        if "final_layer" in k:
            transpose = True # final_layer is transposed {before quantization, after dequantization}
        
        print(f"Weight: {k}")
        print(f"- weights shape: {[_w.shape for _w in _ws]}")

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
        print(f"- quantized weights shape: {[_q.shape for _q in _qs]}")
    if set(ws.keys()) != set(qs.keys()):
        raise RuntimeError("Layer weight names and quantized weight names do not match.")

    return qs

def get_nested_layer(layer, layer_name):
    """Retrieve bottom-most layer per given name"""
    if len(layer_name) == 0:
        return layer
    pieces = layer_name.split("/")
    next_layer_name, remaining_layers_names = pieces[0], "/".join(pieces[1:])
    if next_layer_name.startswith("layer_") and next_layer_name != "layer_norm":
        layer_num = int(next_layer_name[6:])
        if isinstance(layer, Encoder):
            next_layer = layer.enc_layers[layer_num]
        elif isinstance(layer, Decoder):
            next_layer = layer.dec_layers[layer_num]
        else:
            raise RuntimeError(f"Unexpected layer type '{type(layer)}', code was expecting Encoder or Decoder.")
    else:
        next_layer = getattr(layer, next_layer_name)
    return get_nested_layer(next_layer, remaining_layers_names)

def sync_fp_weights_with_quantized_values(model, qs):
    """Model weights are set as dequantized alphas and bits"""
    for key in qs.keys():
        layer = get_nested_layer(model, key)
        deq_weights = [q.dequantize() if isinstance(q, QuantizedWeights) else q for q in qs[key]]
        if len(layer.get_weights()) != len(deq_weights):
            raise RuntimeError(f"Layer '{key}' show incorrect number of weights: {len(layer.get_weights())} != {len(deq_weights)}")
        layer.set_weights(deq_weights)
        print(f"Weights for '{key}' set to quantized-dequantized equivalent.")


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
        """Quantize FP weights to bits and alphas"""
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
        """Dequantized quantized weights and return FP weights"""
        weight = np.zeros(self.w_2d_shape)
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
        sync_fp_weights_with_quantized_values(self.model, qs)
        self._save_model(epoch=epoch, batch=None, logs=logs)

    def _save_model(self, epoch, batch, logs):
        assert 0
