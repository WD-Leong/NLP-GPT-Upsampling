import time
import numpy as np
import pandas as pd
import pickle as pkl
import byte_pair_encoding as bpe

import tensorflow as tf
import tensorflow_addons as tfa
import tf_ver2_gpt_keras_ups as gpt

# Model Parameters. #
p_keep  = 0.9
p_drop = 1.0 - p_keep

seq_length = 30
kernel_sz  = 5
num_heads  = 4
num_layers = 3
hidden_size = 256
ffwd_size   = 4*hidden_size

model_ckpt_dir  = "TF_Models/gpt_ups_sw_reddit"
train_loss_file = "train_loss_gpt_ups_sw_reddit.csv"

# Load the data. #
tmp_pkl_file = "/home/Data/reddit_jokes/"
tmp_pkl_file += "reddit_jokes_subword_v1.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    full_data = pkl.load(tmp_load_file)
    subword_vocab = pkl.load(tmp_load_file)
    idx_2_subword = pkl.load(tmp_load_file)
    subword_2_idx = pkl.load(tmp_load_file)

vocab_size = len(subword_vocab)
print("Vocabulary Size:", str(vocab_size) + ".")
del full_data

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

SOS_token = subword_2_idx["<SOS>"]
EOS_token = subword_2_idx["<EOS>"]
PAD_token = subword_2_idx["<PAD>"]
UNK_token = subword_2_idx["<UNK>"]

# Build the GPT. #
print("Building the GPT Performer Model.")
start_time = time.time()

gpt_model = gpt.GPTUpsample(
    num_layers, num_heads, hidden_size, 
    ffwd_size, vocab_size, seq_length, 
    kernel_sz, rate1=0.0, rate2=p_drop)
gpt_optimizer = tfa.optimizers.AdamW(
    weight_decay=1.0e-4)

elapsed_time = (time.time() - start_time) / 60
print("GPT Keras Model Built", 
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
n_iter = ckpt.step.numpy().astype(np.int32)

print("-" * 50)
print("Testing the GPT Performer Network", 
      "(" + str(n_iter) + " iterations).")
print("-" * 50)

# Testing the Performer. #
while True:
    tmp_phrase = input("Enter input seed: ")
    tmp_phrase = tmp_phrase.lower().strip()
    
    if tmp_phrase == "":
        break
    else:
        tmp_p_index = bpe.bp_encode(
            tmp_phrase, subword_vocab, subword_2_idx)
        
        in_phrase = bpe.bp_decode(
            tmp_p_index, idx_2_subword)
        in_phrase = " ".join(in_phrase).replace(
            "<", "").replace(">", "")
        
        n_tokens = len(tmp_p_index)
        tmp_test = np.array(tmp_p_index).reshape(1, -1)
        tmp_test = tmp_test.astype(np.int32)
        tmp_infer = gpt_model.infer(tmp_test)
        
        gen_phrase = bpe.bp_decode(
            tmp_infer[0].numpy(), idx_2_subword)
        gen_phrase = " ".join(gen_phrase).replace(
            "<", "").replace(">", "")
        del tmp_p_index
        
        print("Input Phrase:")
        print(in_phrase)
        print("Generated Phrase:")
        print(gen_phrase)
        print("-" * 50)

