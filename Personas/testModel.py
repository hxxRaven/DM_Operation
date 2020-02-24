from gensim.models import word2vec
import time

model  = word2vec.Word2Vec.load(r'E:\DM_Operation\Personas\data\word2vec_300-1w.model')

print(model.wv.most_similar("大哥"))

print(model.wv.most_similar("清华"))