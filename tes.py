from keras.preprocessing.text import Tokenizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string


factory2 = StemmerFactory()
stemmer = factory2.create_stemmer()

stop_factory = StopWordRemoverFactory().get_stop_words()
more_stopword = ['dengan', 'ia', 'bahwa', 'oleh', 'kepada', 'kpd', 'yth',
                 'pemerintah', 'provinsi', 'pt', 'mohon', 'terima', 'kasih',
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
                 'yaaa', 'yaaah', 'yaaaaa', 'yaaaa', 'yacc', 'y', 'univeritas',
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
                 'mas', 'mangga', 'mungkin', 'mundu', 'menurut', ]
data = stop_factory + more_stopword
# print(data)
dictionary = ArrayDictionary(data)

# factory = StopWordRemoverFactory()
# data = factory.get_stop_words()+more_stopword
# stopword = factory.create_stop_word_remover()
stopword = StopWordRemover(dictionary)

tokenizer = Tokenizer()
texts = [
    "Kepada, Yth. Pemerintah Provinsi DKI Jakarta, PT.Metaliska yang telah mendirikan bangunan di atas tanah bantaran kali (melanggar GSK) 300 m dari terminal bus Pulogadung arah ke Timur, mohon ditindak Terima kasih"]
# kalimat = "Kepada, Yth. Pemerintah Provinsi DKI Jakarta, PT. Metaliska yang telah mendirikan bangunan di atas tanah bantaran kali (melanggar GSK) 300 m dari terminal bus Pulogadung arah ke Timur, mohon ditindak Terima kasih"
kalimat = input("Masukkan Kalimat : ")
print('==================================')
print('Kalimat awal/asli : ')
print(str(kalimat) + '\n')
print("1. hasil proses case folding :")
print(kalimat.lower())
print('\n')
print('2. hasil proses menghilangkan tanda baca/filtering :')
hasil = kalimat.lower().translate(str.maketrans("", "", string.punctuation))
print(hasil)
print('\n')
print("3. hasil proses stopping word : ")
stop = stopword.remove(hasil)
print(stop)
print('\n')
print('4. hasil proses steeming : ')
katadasar = stemmer.stem(stop)
print(katadasar)
print('\n')
print('5. hasil proses tokenisasi : ')
token = [katadasar]
tokenizer.fit_on_texts(token)
seq = tokenizer.texts_to_sequences(token)

print(tokenizer.word_index)
print(seq)

# print(kalimat.lower())
# kalimat2 = "mendirikan"

# print(hasil)
# hasil2 = hasil.lower()
# # #kalimat baru
# seq1 = tokenizer.texts_to_sequences(["nasi panas sekali"])
# # print("\n"+"Kalimat :\n" + str(kalimat)+"\n")

# print(stop)
# print(stop)
# print("\n")
# print(katadasar)
# seq2 = tokenizer.texts_to_sequences(stop)
# print(seq)
# print(seq2)
# print(seq)
# print("Kalimat :\n"+str(seq1))
# for s in seq:

# katadasar = stemmer.stem(kalimat)
# print("setelah steeming : ")
# print(katadasar)
# print("Hasil Lowercase : " + str(kalimat.lower()))
# print(type(tokenizer.word_index))
# print("Before Steeming : "+str(tokenizer.word_index))
# kataDasar = {}
# angka = 1
# for k in tokenizer.word_index:
# 	kd = stemmer.stem(k)
# 	kataDasar[kd] = angka
# 	angka+=1

# print("\nsetelah Steeming : "+str(kataDasar))
# print("seq. corpus : "+str(seq))
# print("seq. untuk 'nasi panas sekali' : "+str(seq1))

# # ubah list sequence menjadi vector matriks numpy (pilihan : tf-idf)
# encoded_tfidf = tokenizer.sequences_to_matrix(seq, mode="tfidf")
# print("tfidf : ")
# print(encoded_tfidf)
# encoded_binary = tokenizer.sequences_to_matrix(seq, mode="binary")
# print("binary : ")
# print(encoded_binary)
# encoded_count = tokenizer.sequences_to_matrix(seq, mode="count")
# print("count : ")
# print(encoded_count)
# encoded_freq = tokenizer.sequences_to_matrix(seq, mode="freq")
# print("freq")
# print(encoded_freq)

# # contoh padding secara manual dengan fungsi yg disediakan keras
# # yaitu menggunakan fungsi pad_sequences
# print("\nContoh padding manual dengan pad_sequences")
# from keras.preprocessing.sequence import pad_sequences
# print("sebelum padding:")
# print(seq)
# X = pad_sequences(seq)
# print("sesudah padding:")
# print(X)
# print(X.shape)

# import csv
