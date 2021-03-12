dataset_slice = slice(10_000)  # in sentences
ru_word_count_minimum = 2
en_word_count_minimum = 2
infrequent_words_percent = 0.01
batch_size = 30

encoder_bi = True
encoder_hidden_dim = 256
encoder_hd = (encoder_bi + 1) * encoder_hidden_dim
encoder_embed_dim = 256

decoder_hidden_dim = 256
decoder_hd = decoder_hidden_dim
decoder_embed_dim = 256

learning_rate = 0.001
gradient_norm_clip = 1.

epochs = 20

teaching_percent = 0.5
