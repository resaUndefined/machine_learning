from keras.preprocessing.text import Tokenizer
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import models, layers
import pickle
import csv
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string
from keras import optimizers
from keras import backend as K


factory2 = StemmerFactory()
stemmer = factory2.create_stemmer()

stop_factory = StopWordRemoverFactory().get_stop_words()
more_stopword = [
                    'dengan', 'ia','bahwa','oleh', 'kepada', 'kpd', 'yth',
                    'pemerintah','provinsi', 'pt', 'mohon', 'terima', 'kasih',
                    'trmksh', 'saya', 'selalu', 'kok', 'belum', 'tidak', 'biasa',
                    'ada', 'bantuan', 'bantuannya', 'sy', 'bahwa', 'juga',
                    'atas', 'nama', 'mau', 'hari', 'bulan', 'tahun', 'tetapi', 
                    'ini', 'itu', 'tapi', 'ya', 'tp', 'ngga', 'nggak', 'enggak',
                    'akan', 'kata', 'sendiri', 'mengatakan', 'dimana', 'untuk',
                    'tersebut', 'sangat', 'sulit', 'hingga', 'di', 'saat', 'dari',
                    'kita', 'pada', 'sampai', 'telah', 'pasti', 'saja', 'nah',
                    'disaat', 'apabila', 'maka', 'masing', 'masing-masing',
                    'namun', 'dr', 'dri', 'karena', 'kan', 'sllu', 'setiap',
                    'kemarin', 'kemaren', 'walaupun', 'rencana', 'rokok', 'rekan',
                    'ramp', 'rampung', 'ramah', 'tanpa', 'terlalu', 'udah',
                    'tiap', 'tiga', 'tgl', 'tolong', 'tentang', 'ternyata',
                    'tdk', 'tanyakan', 'tanya', 'tsb', 'trus', 'toyib', 'yang',
                    'yg', 'yaitu', 'yuli', 'ya', 'yah', 'yuk', 'your', 'yaa',
                     'yaaa','yaaah', 'yaaaaa', 'yaaaa', 'yacc', 'y', 'univeritas',
                     'ujang', 'utk', 'ingin', 'indonesia', 'iva', 'iurang', 'isteri',
                     'ingatkan', 'orang', 'padahal', 'pak', 'para', 'pagi', 
                     'pernah', 'perlu', 'pukul', 'pun', 'punya', 'apakah', 'agar',
                     'atau', 'adanya', 'apa', 'adalah', 'ada', 'ataupun', 'akhir',
                     'artinya', 'akhirnya', 'ayang', 'arjuna', 'antara', 'angke', 
                     'aladin', 'sudah', 'sebagai', 'seperti', 'segera', 'sedang',
                     'setelah', 'sekarang', 'sebelumnya', 'secara', 'sering',
                     'sdh', 'seharusnya', 'sama', 'suka', 'solusinya', 'syahputra',
                     'supaya', 'sruh', 'sit', 'sinta', 'dan', 'dapat', 'dalam',
                     'duren', 'disini', 'ditambah', 'disana', 'diri', 'disalah',
                     'dikarenakan', 'dia', 'dgn', 'dwi', 'drg', 'dong', 'doen',
                     'dll', 'fajar', 'green', 'gong', 'gebang', 'harus', 'hal',
                     'hendak', 'hormat', 'harinya', 'jika', 'jadi', 'jangan',
                     'jelas', 'kami', 'kurang', 'kapan', 'kenapa', 'kec', 'kecamatan',
                     'katanya', 'kakak', 'krn', 'kosambi', 'kopi', 'kiri', 'kini',
                     'kerata', 'ketika', 'kesana', 'lebih', 'lagi', 'lalu',
                     'lanjutnya', 'lain', 'lodan', 'lift?', 'laporan', 'lapor',
                     'lainnya', 'lubuk', 'cengkareng', 'cempaka', 'cukup',
                     'cirimekar', 'ceger', 'cara', 'cira', 'cina', 'cenderung',
                     'cari', 'bisa', 'banyak', 'bagaimana', 'beberapa', 'bagi',
                     'bertanya', 'baru', 'berapa', 'baik', 'bagian', 'banyaknya',
                     'bukan', 'buka', 'buaya', 'boleh', 'bernama', 'berapakah', 
                     'berada', 'bilang', 'begitu', 'bahkan', 'nya', 'nyata', 
                     'nikita', 'noka', 'nanya', 'ngurah', 'ngasih', 'namum',
                     'namanya', 'masih', 'melakukan', 'mereka', 'merokok',
                     'memang', 'mana', 'malam', 'misal', 'melporkan', 'mdr',
                     'mas', 'mangga', 'mungkin', 'mundu', 'menurut',
                    ]
data = stop_factory + more_stopword
dictionary = ArrayDictionary(data)

# factory = StopWordRemoverFactory()
# data = factory.get_stop_words()+more_stopword
# stopword = factory.create_stop_word_remover()
stopword = StopWordRemover(dictionary)

tokenizer = Tokenizer()
nama_file = "D:\\resa\\D\\KULIAH\\S2\\Semester 1\\python\\mlNN_1\\datasetSMS\\data-set-50-persen.csv"
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

    , f1_m, precision_m, recall_m
label = to_categorical(label)
print(label)
print(label.shape)
data2 = []
for x in data:
    # casefolding
    x2 = x.lower()
    # punctuation proccess
    bersih = x2.translate(str.maketrans("","",string.punctuation))
    # stoppword use PYsastrawi
    stop = stopword.remove(bersih)
    # steeming use PYsastrawi
    katadasar = stemmer.stem(stop)
    # insert to another or new list with name data2
    data2.append(katadasar)
    # ready to proccess

X_train, X_test, y_train, y_test = train_test_split(
    data2,
    label,
    test_size=0.2,
    random_state=1)

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
model.add(layers.Dense(9, activation='softmax'))  # karena kelasnya ada 3
sgd = optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy', f1_m, precision_m, recall_m])

history = model.fit(X_enc_train, y_train, epochs=200,
                    batch_size=2, validation_split=0.2, verbose=0)
# results = model.evaluate(X_enc_test, y_test)
# print("Hasil [loss,acc] untuk data test : ")
# print(results)
loss, accuracy, f1_score, precision, recall = model.evaluate(
    X_enc_test, y_test, verbose=0)
print('\n')
print('Loss : ' + str(loss))
print('Accuray : ' + str(accuracy))
print('F1 Score : ' + str(f1_score))
print('Recall : ' + str(recall))

model.save('model_lapor_prep_50_persen_v1.h5')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
