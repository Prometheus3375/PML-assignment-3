dataset_slice = slice(1_000)  # in sentences
infrequent_words_percent = 0.01
batch_size = 100

embed_dim = 256
hidden_dim = 256

learning_rate = 0.001
norm_clip = 1.

epochs = 10
log_interval = 10  # in batches

teaching_percent = 0.5
