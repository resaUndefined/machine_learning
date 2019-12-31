from keras.models import load_model
import pickle

model = load_model('model_spam_v1.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

s = ["Selamat ini 50 juta untuk anda",
     "Beli paket Flash mulai 1GB", "Nanti ketemuan dimana ?"]
# s = ["selamat ini 50 juta untuk anda"]
seq_str = tokenizer.texts_to_sequences(s)
enc_str = tokenizer.sequences_to_matrix(seq_str, mode='tfidf')
enc_str.shape
pred = model.predict_classes(enc_str)
print("Prediksi kelas string ' {} ' adalah {}".format(s, pred))
