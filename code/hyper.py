dataset_slice = slice(100_000)  # in sentences
ru_word_count_minimum = 2
en_word_count_minimum = 2
infrequent_words_percent = 0.01
batch_size = 8

embed_dim = 256
hidden_dim = 256

learning_rate = 0.001
norm_clip = 1.

epochs = 20
log_interval = 10  # in batches

teaching_percent = 0.5
