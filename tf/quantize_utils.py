import tensorflow as tf

def collect_weights(transformer):
    ws = {}
    encoder = transformer.encoder
    decoder = transformer.decoder

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
    
    ws["final_layer"]  = transformer.final_layer.get_weights()

    return ws

def quantize_transformer(transformer):
    # quantize encoder

    qs = {}
    ws = collect_weights(transformer)

    num_layers = 0
    for k, v in ws.items():
        print(k, [_v.shape for _v in v])
        num_layers += len(v)
    
    if len(transformer.get_weights()) != num_layers:
        raise RuntimeError("Number of collected weights does not match the number of weight matrices in the model")



class CheckpointQuantizer(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.save_freq != "epoch":
            raise RuntimeError(
                "CheckpointQuantizer only supports epoch frequency")
    

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        quantize_transformer(self.model)
        assert 0
        self._save_model(epoch=epoch, batch=None, logs=logs)
    


    def _save_model(self):
        pass
