import time
import numpy as np
import pandas as pd
import pickle as pkl
import byte_pair_encoding as bpe

import tensorflow as tf
import tensorflow_addons as tfa
import tf_ver2_gpt_keras_ups as gpt

# Model Parameters. #
p_keep = 0.9
p_drop = 1.0 - p_keep

num_heads  = 4
num_layers = 3
seq_length = 50
kernel_sz  = 5
hidden_size = 256
ffwd_size   = 4*hidden_size

model_ckpt_dir  = "TF_Models/dialogue_sw_gpt_keras_ups"
train_loss_file = "train_loss_dialogue_sw_gpt_keras_ups.csv"

# Load the data. #
tmp_pkl_file = "/home/Data/movie_dialogs/"
tmp_pkl_file += "movie_dialogues_subword.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    data_tuple = pkl.load(tmp_load_file)
    subword_vocab = pkl.load(tmp_load_file)
    idx_2_subword = pkl.load(tmp_load_file)
    subword_2_idx = pkl.load(tmp_load_file)
del data_tuple

vocab_size = len(subword_vocab)
print("Vocabulary Size:", str(vocab_size))

SOS_token = subword_2_idx["<SOS>"]
EOS_token = subword_2_idx["<EOS>"]
PAD_token = subword_2_idx["<PAD>"]
UNK_token = subword_2_idx["<UNK>"]

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Build the GPT. #
print("Building the GPT Performer Model.")
start_time = time.time()

gpt_model = gpt.GPTUpsample(
    num_layers, num_heads, hidden_size, 
    ffwd_size, vocab_size, seq_length, 
    kernel_sz, rate1=0.0, rate2=p_drop)
gpt_optimizer = tfa.optimizers.AdamW(
    weight_decay=1.0e-4)

elapsed_time = (time.time()-start_time) / 60
print("GPT Model Built", 
      "(" + str(elapsed_time) + " mins).")

# Print the model summary. #
tmp_zero = np.zeros(
    [1, seq_length], dtype=np.int32)
tmp_pred = gpt_model(tmp_zero, training=True)

print(gpt_model.summary())
print("-" * 50)
del tmp_zero, tmp_pred

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    gpt_model=gpt_model, 
    gpt_optimizer=gpt_optimizer)

manager = tf.train.CheckpointManager(
    ckpt, model_ckpt_dir, max_to_keep=1)

ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Model restored from {}".format(
        manager.latest_checkpoint))
else:
    print("Error: No latest checkpoint found.")

# Test the Performer model. #
n_iter = ckpt.step.numpy().astype(np.int32)

print("-" * 50)
print("Testing the GPT Network", 
      "(" + str(n_iter) + " iterations).")
print("-" * 50)

while True:
    tmp_phrase = input("Enter input phrase: ")
    tmp_phrase = tmp_phrase.lower().strip()
    
    if tmp_phrase == "":
        break
    else:
        tmp_i_idx = bpe.bp_encode(
            tmp_phrase, subword_vocab, subword_2_idx)
        tmp_i_tok = bpe.bp_decode(tmp_i_idx, idx_2_subword)
        n_sw_toks = len(tmp_i_idx) + 1
        
        tmp_in_phrase  = " ".join(
            tmp_i_tok).replace("<", "").replace(">", "")
        
        tmp_test = np.array(
            tmp_i_idx + [SOS_token], dtype=np.int32)
        tmp_test = tmp_test.reshape(1, -1)
        
        gen_tokens = gpt_model.infer(
            tmp_test).numpy()[0]
        gen_phrase = bpe.bp_decode(
            gen_tokens, idx_2_subword)
        gen_phrase = " ".join(
            gen_phrase).replace("<", "").replace(">", "")
        
        gen_output = bpe.bp_decode(
            gen_tokens[(n_sw_toks-1):], idx_2_subword)
        gen_output = " ".join(
            gen_output).replace("<", "").replace(">", "")
        
        print("Input Phrase:")
        print(tmp_in_phrase)
        print("Generated Phrase:")
        print(gen_phrase)
        print("Generated Response:")
        print(gen_output)
        print("-" * 50)

