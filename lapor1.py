from keras.preprocessing.text import Tokenizer
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import models, layers, metrics, optimizers
import pickle
import csv
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from keras import backend as K


tokenizer = Tokenizer()
nama_file = "D:\\resa\\D\\KULIAH\\S2\\Semester 1\\python\\mlNN_1\\datasetSMS\\dataset_lapor.csv"
# df = pd.read_csv(nama_file, encoding = "ISO-8859-1").values
# data = df[:, 0]
# label = df[:, 1]

data = []
label = []
with open(nama_file, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(reader)  # skip header
    for row in reader:
        data.append(row[0])
        label.append(row[1])
# test lihat dua data pertama
# print(data[:2])
# print(label[:2])

label = to_categorical(label)
# print(label)
# print(label.shape)
X_train, X_test, y_train, y_test = train_test_split(
    data,
    label,
    test_size=0.2,
    random_state=1)


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# fit hanya berdasarkan data train
tokenizer.fit_on_texts(X_train)
# konversi train
seq_x_train = tokenizer.texts_to_sequences(X_train)
X_enc_train = tokenizer.sequences_to_matrix(seq_x_train, mode="tfidf")
# # konversi teks
seq_x_test = tokenizer.texts_to_sequences(X_test)
X_enc_test = tokenizer.sequences_to_matrix(seq_x_test, mode="tfidf")
# print(seq_x_train)
# print(X_enc_train.shape)
# # print(X_enc_test.shape)
# print(X_enc_train)

_, jum_fitur = X_enc_train.shape
model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(jum_fitur,)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(9, activation='softmax'))  # karena kelasnya ada 3
sgd = optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy',f1_m,precision_m, recall_m])

history = model.fit(X_enc_train, y_train, epochs=200,
                    batch_size=256, validation_split=0.2, verbose=0)
loss, accuracy, f1_score, precision, recall = model.evaluate(X_enc_test, y_test, verbose=0)
results = model.evaluate(X_enc_test, y_test)
print("Hasil [loss,acc] untuk data test : ")
print(results)
print('\n')
print('Loss : ' + str(loss))
print('Accuray : ' + str(accuracy))
print('F1 Score : ' + str(f1_score))
print('Recall : ' + str(recall))
# print(metrics.accuracy(y_true, y_pred))

model.save('model_lapor_noprep_v1.h5')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# lb = LabelBinarizer()
# # trainY = lb.fit_transform(trainY)
# # testY = lb.fit_transform(testY)

# # Evaluate TEST model class prediction accuracy
# print("[INFO] Evaluating network...")
# predictions = model.predict(X_enc_test, batch_size=256)
# target_names = [str(x) for x in lb.classes_]
# print(classification_report(y_test.argmax(axis=1),
#                             predictions.argmax(axis=1),
#                             target_names=target_names))
# # Evaluate TRAIN model class prediction accuracy
# print("[INFO] Evaluating network...")
# trainPreds = model.predict(X_enc_train, batch_size=256)
# target_names = [str(x) for x in lb.classes_]
# print(classification_report(y_train.argmax(axis=1),
#                             trainPreds.argmax(axis=1),
#                             target_names=target_names))

# y_pred = model.predict(X_enc_test)
# print(classification_report(y_true, y_pred))
