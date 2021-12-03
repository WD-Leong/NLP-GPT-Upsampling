
import time
import json
import pickle as pkl
from collections import Counter
import byte_pair_encoding as bpe
from nltk.tokenize import wordpunct_tokenize

print("Loading the data.")
start_tm = time.time()

tmp_file = "/home/Data/reddit_jokes/reddit_jokes.json"
tmp_data = json.loads(open(tmp_file).read())
max_len  = 50

# Extract the data. #
tmp_jokes_tuple = []
for tmp_row in tmp_data:
    if tmp_row["body"].find(tmp_row["title"]) != -1:
        tmp_joke = tmp_row["body"]
    elif tmp_row["title"].find(tmp_row["body"]) != -1:
        tmp_joke = tmp_row["title"]
    else:
        tmp_joke = tmp_row["title"] + " " + tmp_row["body"]
    tmp_jokes_tuple.append(tmp_joke)
del tmp_row, tmp_joke
print("Total of", len(tmp_jokes_tuple), "jokes loaded.")

# Process the data. #
tmp_jokes_filtered = []

w_counter = Counter()
for tmp_joke in tmp_jokes_tuple:
    tmp_joke = tmp_joke.replace(
        "\n", " \n ").replace("\'", " ")
    tmp_tokens = [
        x for x in wordpunct_tokenize(
            tmp_joke.lower()) if x != ""]
    
    if len(tmp_tokens) <= max_len:
        w_counter.update(tmp_tokens)
        tmp_jokes_filtered.append(tmp_joke)
    del tmp_tokens

print("Total of", len(tmp_jokes_filtered), "jokes filtered.")
del tmp_jokes_tuple

word_counts = []
for word, count in w_counter.items():
    tmp_word = "<" + word + ">"
    tmp_word = "".join([
        x + " " for x in tmp_word]).strip()
    word_counts.append((tmp_word, count))
word_counts = dict(word_counts)

elapsed_tm = (time.time() - start_tm) / 60.0
print("Total of", len(word_counts), "words.")
print("Elapsed Time:", str(elapsed_tm), "mins.")

# Fit the subword vocabulary. #
print("Fitting subword vocabulary.")
start_tm = time.time()

n_iters = 750
vocab_size = 10000
tuple_out  = bpe.learn_subword_vocab(
    word_counts, n_iters, vocab_size=vocab_size)

subword_vocab = tuple_out[0]
idx_2_subword = tuple_out[1]
subword_2_idx = tuple_out[2]

elapsed_tm = (time.time() - start_tm) / 60
print("Total Sub-word Vocabulary size:", 
      len(subword_vocab), "sub-word tokens")
print("Elapsed Time:", str(elapsed_tm), "mins.")

# Encode the corpus to subword tokens. #
print("Encoding the corpus to subwords.")
start_tm = time.time()
num_data = len(tmp_jokes_filtered)

new_jokes_tuple = []
for n_joke in range(num_data):
    tmp_joke = tmp_jokes_filtered[n_joke]
    tmp_joke_sw = bpe.bp_encode(
        tmp_joke, subword_vocab, subword_2_idx)
    new_jokes_tuple.append(tmp_joke_sw)
    
    if (n_joke + 1) % 10000 == 0:
        print(n_joke+1, "out of", 
              num_data, "jokes processed.")

elapsed_tm = (time.time() - start_tm) / 60.0
print("Elapsed Time:", str(elapsed_tm), "mins.")

# Save the data. #
print("Saving the file.")

tmp_pkl_file = "/home/Data/reddit_jokes/"
tmp_pkl_file += "reddit_jokes_subword_v1.pkl"
with open(tmp_pkl_file, "wb") as tmp_file_save:
    pkl.dump(new_jokes_tuple, tmp_file_save)
    pkl.dump(subword_vocab, tmp_file_save)
    pkl.dump(idx_2_subword, tmp_file_save)
    pkl.dump(subword_2_idx, tmp_file_save)
