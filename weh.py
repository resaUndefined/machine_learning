from keras.preprocessing.text import Tokenizer
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import models, layers
import pickle

tokenizer = Tokenizer()
nama_file = "D:\\resa\\D\\KULIAH\\S2\\Semester 1\\python\\mlNN_1\\datasetSMS\\dataset.csv"
df = pd.read_csv(nama_file).values
data = df[:, 0]
label = df[:, 1]

label = to_categorical(label)
print(label)
print(label.shape)
X_train, X_test, y_train, y_test = train_test_split(
    data,
    label,
    test_size=0.2,
    random_state=123)

# fit hanya berdasarkan data train
tokenizer.fit_on_texts(X_train)
# konversi train
seq_x_train = tokenizer.texts_to_sequences(X_train)
X_enc_train = tokenizer.sequences_to_matrix(seq_x_train, mode="tfidf")
# konversi test
seq_x_test = tokenizer.texts_to_sequences(X_test)
X_enc_test = tokenizer.sequences_to_matrix(seq_x_test, mode="tfidf")

# print(X_enc_train.shape)
# print(X_enc_test.shape)
# print(X_enc_train)

_, jum_fitur = X_enc_train.shape
model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(jum_fitur,)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))  # karena kelasnya ada 3
model.compile(optimizer="adam",
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_enc_train, y_train, epochs=3,
                    batch_size=2, validation_split=0.2)
results = model.evaluate(X_enc_test, y_test)
print("Hasil [loss,acc] untuk data test : ")
print(results)

model.save('model_spam_v1.h5')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
