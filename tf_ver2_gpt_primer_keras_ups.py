import tensorflow as tf
from tensorflow.keras.layers import (
    Embedding, LayerNormalization, UpSampling2D, DepthwiseConv2D)

# Multi-Head Attention Layer. #
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(
        self, d_model, n_heads, ker_sz, 
        depth_ker=3, name="multi_head_attn_layer"):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        d_depth = int(d_model / n_heads)
        self.ker_sz  = ker_sz
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_depth = d_depth
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.wc = tf.keras.layers.Dense(d_model)
        
        self.wup = tf.keras.layers.Dense(d_depth)
        self.ups = UpSampling2D(size=(1, self.ker_sz[1]))
        
        self.depth_ker = (1, depth_ker)
        self.depth_cnn_q = DepthwiseConv2D(
            self.depth_ker, strides=(1, 1), 
            padding="VALID", depth_multiplier=1)
        self.depth_cnn_k = DepthwiseConv2D(
            self.depth_ker, strides=(1, 1), 
            padding="VALID", depth_multiplier=1)
        self.depth_cnn_v = DepthwiseConv2D(
            self.depth_ker, strides=(1, 1), 
            padding="VALID", depth_multiplier=1)
    
    def split_heads(self, x):
        # Input is (batch_size, seq_len, d_model). #
        # Output is (batch_size, num_heads, seq_len, depth). #
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[1]
        output_shp = (batch_size, seq_length, 
                      self.n_heads, self.d_depth)
        
        x = tf.reshape(x, output_shp)
        return tf.transpose(x, [0, 2, 1, 3])
    
    def combine_heads(self, x):
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[2]
        output_shp = (
            batch_size, seq_length, self.d_model)
        
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, output_shp)
    
    def call(self, v, k, q):
        neg_infty  = -1.0e9
        batch_size = tf.shape(q)[0]
        seq_length = tf.shape(q)[1]
        
        norm_dim = tf.math.rsqrt(
            tf.math.sqrt(float(self.d_depth)))
        
        # For average pooling. #
        zero_shp = [
            batch_size, self.n_heads, 
            self.ker_sz[1]-1, self.d_depth]
        zero_pad = tf.zeros(
            zero_shp, dtype=tf.float32, name="zero_pad")
        
        # For Depthwise CNN. #
        zero_cnn = [
            batch_size, self.depth_ker[1]-1, self.d_model]
        cnn_pad = tf.zeros(
            zero_cnn, dtype=tf.float32, name="cnn_pad")
        
        # Depthwise CNN. #
        v = self.wv(v)
        q = self.wq(q) * norm_dim
        k = self.wk(k) * norm_dim
        
        q_cnn = tf.expand_dims(tf.concat(
            [cnn_pad, q], axis=1), axis=1)
        k_cnn = tf.expand_dims(tf.concat(
            [cnn_pad, k], axis=1), axis=1)
        v_cnn = tf.expand_dims(tf.concat(
            [cnn_pad, v], axis=1), axis=1)
        
        q_in = self.split_heads(
            tf.squeeze(self.depth_cnn_q(q_cnn), axis=1))
        k_in = self.split_heads(
            tf.squeeze(self.depth_cnn_k(k_cnn), axis=1))
        v_in = self.split_heads(
            tf.squeeze(self.depth_cnn_v(v_cnn), axis=1))
        
        # Average Pooling. #
        q_pad = tf.concat([zero_pad, q_in], axis=2)
        k_pad = tf.concat([zero_pad, k_in], axis=2)
        v_pad = tf.concat([zero_pad, v_in], axis=2)
        
        q_prime = tf.nn.avg_pool2d(
            q_pad, self.ker_sz, 
            self.ker_sz, padding="VALID")
        k_prime = tf.nn.avg_pool2d(
            k_pad, self.ker_sz, 
            self.ker_sz, padding="VALID")
        v_prime = tf.nn.avg_pool2d(
            v_pad, self.ker_sz, 
            self.ker_sz, padding="VALID")
        
        # Generate the attention mechanism. #
        attn_len  = tf.shape(q_prime)[2]
        attn_mask = tf.linalg.band_part(
            tf.ones([attn_len, attn_len]), -1, 0)
        attn_mask = neg_infty * (1.0 - attn_mask)
        
        attn_logits  = tf.matmul(
            q_prime, k_prime, transpose_b=True)
        attn_weights = tf.nn.softmax(
            tf.add(attn_mask, attn_logits))
        attn_outputs = tf.matmul(
            attn_weights, v_prime)
        
        # Resample back to original length. #
        attn_outputs = self.wup(self.ups(
            attn_outputs))[:, :, :seq_length, :]
        attn_outputs = self.wc(
            self.combine_heads(attn_outputs))
        return attn_outputs
        
class FFWNetwork(tf.keras.layers.Layer):
    def __init__(self, d_ffwd, d_model):
        super(FFWNetwork, self).__init__()
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        
        self.ffwd_1 = tf.keras.layers.Dense(
            d_ffwd, activation="relu")
        self.ffwd_2 = tf.keras.layers.Dense(d_model)
    
    def call(self, x):
        # Use Square ReLU activation function. #
        return self.ffwd_2(tf.square(self.ffwd_1(x)))

# GPT Decoder Layer. #
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self, d_model, n_heads, d_ffwd, 
        ker_sz, depth_ker=3, rate1=0.1, 
        rate2=0.1, name="decoder_layer"):
        super(DecoderLayer, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.d_model = d_model
        self.depth_ker = (1, depth_ker)
        self.ffwd_self = FFWNetwork(d_ffwd, d_model)
        self.attn_self = MultiHeadAttention(
            d_model, n_heads, ker_sz, depth_ker=depth_ker, name=name)
        
        self.lnorm_1 = LayerNormalization(epsilon=1e-6)
        self.lnorm_2 = LayerNormalization(epsilon=1e-6)
        self.dropout_1 = tf.keras.layers.Dropout(rate1)
        self.dropout_2 = tf.keras.layers.Dropout(rate2)
    
    def call(
        self, x_enc, x_pos, training=True):
        batch_size = tf.shape(x_enc)[0]
        
        x_embed = x_enc + x_pos
        attn_self_output = self.attn_self(
            x_embed, x_embed, x_embed)
        
        # Apply Normalisation followed by adding. #
        attn_self_output = self.dropout_1(
            attn_self_output, training=training)
        attn_self_output = tf.add(
            x_embed, self.lnorm_1(attn_self_output))
        
        # Feed-Forward Network. #
        ffwd_self_output = self.lnorm_2(
            self.ffwd_self(attn_self_output))
        ffwd_self_output = tf.add(
            attn_self_output, ffwd_self_output)
        ffwd_self_output = self.dropout_2(
            ffwd_self_output, training=training)
        return ffwd_self_output

class Decoder(tf.keras.layers.Layer):
    def __init__(
        self, n_layers, d_model, n_heads, 
        d_ffwd, vocab_size, max_seq_length, 
        ker_sz, depth_ker=3, rate1=0.1, rate2=0.1):
        super(Decoder, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.n_heads  = n_heads
        self.n_layers = n_layers
        
        self.ker_sz  = (1, ker_sz)
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        self.seq_len = max_seq_length
        self.d_rsqrt = tf.math.sqrt(
            tf.cast(d_model, tf.float32))
        self.depth_ker  = depth_ker
        self.vocab_size = vocab_size
        
        # Embedding layers. #
        tmp_pos_embed = []
        for n_layer in range(n_layers):
            tmp_pos_embed.append(
                Embedding(max_seq_length, d_model))
        
        self.pos_embed = tmp_pos_embed
        self.dec_embed = Embedding(vocab_size, d_model)
        del tmp_pos_embed
        
        # Decoder Layers. #
        tmp_dec_layers = []
        for n_layer in range(n_layers):
            dec_layer_name = "decoder_layer_" + str(n_layer+1)
            tmp_dec_layers.append(DecoderLayer(
                d_model, n_heads, d_ffwd, 
                self.ker_sz, depth_ker=depth_ker, 
                rate1=rate1, rate2=rate2, name=dec_layer_name))
        
        self.dec_layers = tmp_dec_layers
        self.emb_dropout = tf.keras.layers.Dropout(rate1)
        del tmp_dec_layers
    
    def call(self, x, training=True):
        seq_length = tf.shape(x)[1]
        
        x_pos_index = tf.expand_dims(
            tf.range(seq_length), axis=0)
        x_tok_embed = self.dec_embed(x)
        x_tok_embed = self.emb_dropout(
            x_tok_embed * self.d_rsqrt, training=training)
        
        layer_input = x_tok_embed
        for m in range(self.n_layers):
            x_pos_embed = self.pos_embed[m](x_pos_index)
            x_pos_embed = self.emb_dropout(
                x_pos_embed * self.d_rsqrt, training=training)
            
            layer_output = self.dec_layers[m](
                layer_input, x_pos_embed, training=training)
            layer_input  = layer_output
        return layer_output

class GPTUpsample(tf.keras.Model):
    def __init__(
        self, n_layers, n_heads, d_model, 
        d_ffwd, vocab_size, max_seq_length, 
        ker_sz, depth_ker=3, rate1=0.1, rate2=0.1):
        super(GPTUpsample, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.n_heads  = n_heads
        self.n_layers = n_layers
        
        self.ker_sz  = ker_sz
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        self.seq_len = max_seq_length
        self.depth_ker  = depth_ker
        self.vocab_size = vocab_size
        
        # Output projection. #
        self.gpt_model = Decoder(
            n_layers, d_model, n_heads, d_ffwd, 
            vocab_size, max_seq_length, ker_sz, 
            depth_ker=depth_ker, rate1=rate1, rate2=rate2)
        self.p_decoder = tf.keras.layers.Dense(vocab_size)
    
    def call(self, x, training=True):
        dec_outputs = self.gpt_model(
            x, training=training)
        dec_logits  = self.p_decoder(dec_outputs)
        return dec_logits
    
    def infer(self, x):
        input_len = tf.shape(x)[1]
        infer_ids = [tf.expand_dims(x[:, 0], axis=1)]
        
        for step in range(self.seq_len):
            tmp_inputs = tf.concat(infer_ids, axis=1)
            tmp_logits = self.call(tmp_inputs, training=False)
            
            tmp_logit = tmp_logits[:, -1, :]
            tmp_index = tf.cond(
                step < (input_len-1), 
                lambda: x[:, step+1], 
                lambda: tf.argmax(
                    tmp_logit, axis=-1, output_type=tf.int32))
            infer_ids.append(tf.expand_dims(tmp_index, axis=1))
        return tf.concat(infer_ids, axis=1)
        

