from gensim.scripts.glove2word2vec import glove2word2vec

glove_input_file = 'D:\\glove\\glove.840B.300d.txt'
word2vec_output_file = 'D:\\glove\\glove.840B.300d.gensim.txt'
glove2word2vec(glove_input_file, word2vec_output_file)