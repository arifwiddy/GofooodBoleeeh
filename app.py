import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import squarify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec


st.title(" :bar_chart: Analisis Gofood")
st.image("gofood.jpg")
st.write("oleh Kelompok GofooodBoleeeh")
st.write("""Anggota kelompok:
1. Arif Widiatmoko (021002314005)
2. Mukhlisha Hayuningtyas (021002314003)
3. Safira Izzah (021002314006)

Grup Kelas Khusus S1 Ekonomi Pembangunan""")


##mba lisha
st.write("## Pendahuluan")
st.markdown("""
<div style="text-align: justify;">
Gofood merupakan layanan Gojek yang melayani food delivery service di Indonesia. Sebagai bagian dari Grup Gojek, GoFood mengalami peningkatan pesanan yang tinggi sepanjang tahun 2019 di mana meningkat hingga 133% (Januari hingga Agustus) dibandingkan periode yang sama tahun lalu. Jika dilihat dari pangsa pasarnya, Go-Food telah menguasai 47% pangsa pasar di Indonesia atau senilai dengan 1,74 US Dollar (Setyowati, 2021). Keberadaan layanan GoFood telah memberikan dampak signifikan dalam mendorong pertumbuhan Usaha Mikro, Kecil, dan Menengah (UMKM) di sektor kuliner lokal. Melalui platform ini, para pengusaha kuliner dapat memperluas jangkauan pasar mereka secara digital, meningkatkan visibilitas produk, dan mengakses pelanggan yang lebih luas. Menurut survei yang dilakukan oleh Alvara Research Center, GoFood menjadi platform paling populer di kalangan UMKM kuliner, dengan 86,6% responden menggunakan layanan ini untuk menjual produk mereka. Selain itu, data dari Gojek menunjukkan bahwa 99% dari lebih dari 1,4 juta mitra usaha GoFood merupakan UMKM kuliner. Hal ini menegaskan peran penting GoFood dalam ekosistem UMKM di Indonesia.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: justify;">
Dalam hal ini lokus yang menjadi sorotan adalah DKI Jakarta, Surabaya, dan Medan. Pasalnya, Melihat dari data survei Alvara Research Center, mayoritas UMKM kuliner di kota-kota besar Indonesia, termasuk Jakarta, Surabaya, dan Medan, memilih GoFood sebagai platform utama untuk menjual produk mereka, dengan persentase mencapai 86,6%. Selain itu, data internal Gojek menunjukkan bahwa dari sekitar 1 juta mitra GoFood di seluruh Indonesia, 99% di antaranya merupakan UMKM lokal. Kehadiran GoFood di kota-kota besar ini tidak hanya memperluas akses pasar bagi UMKM kuliner, tetapi juga mendorong pertumbuhan ekonomi lokal melalui peningkatan penjualan dan pembukaan cabang baru.
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div style="text-align: justify;">
Dengan berkembangnya layanan berbasis digital, seperti aplikasi Gojek dengan fitur GoFood, perilaku konsumen di sektor kuliner di kota-kota besar Indonesia, seperti Jakarta, Surabaya, dan Medan, mengalami perubahan yang signifikan. Dalam tulisan ini, akan dibahas mengenai analisis keputusan pembelian pada aplikasi GoFood di ketiga kota tersebut pada tahun 2021. Analisis ini didasarkan pada data mentah yang diambil dari Kaggle, sebanyak 45.195 entri, yang mencakup berbagai informasi penting terkait perilaku konsumen, pola pemesanan, informasi diskon, serta kategori makanan dan minuman yang paling diminati.
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div style="text-align: justify;">
Studi ini akan menggunakan pendekatan deskriptif, seperti analisis statistik, wordcloud untuk memvisualisasikan kategori populer, serta klasterisasi untuk mengidentifikasi pola pembelian berdasarkan data GoFood. Dengan pendekatan ini, penelitian diharapkan dapat memberikan wawasan yang bermanfaat bagi UMKM kuliner dalam meningkatkan keterlibatan dengan pelanggan. Selain itu, temuan ini juga berpotensi mendukung UMKM dalam merancang strategi pemasaran yang lebih efektif, seperti pemanfaatan diskon, penyesuaian menu, dan optimalisasi layanan berbasis lokasi.
</div>
""", unsafe_allow_html=True)


##mba lisha
st.write("## Deskripsi Data")
st.write("Dalam menganalisis data pemesanan Gofood, data yang dibutuhkan di antaranya sebagai berikut:")

st.write("1.	Merchant Name : Variabel ini mencantumkan nama mitra usaha (merchant) yang terdaftar di platform GoFood. Data ini penting untuk mengidentifikasi mitra usaha yang berkontribusi dalam layanan GoFood, serta untuk analisis distribusi aktivitas merchant di berbagai kategori dan area.")
st.write("2.	Merchant Area : Area geografis tempat merchant beroperasi. Dalam konteks kota besar seperti Jakarta, Surabaya, dan Medan, variabel ini dapat digunakan untuk mengelompokkan data berdasarkan wilayah, sehingga memudahkan analisis pola geografis, preferensi konsumen di masing-masing kota, dan pengaruh lokasi terhadap keputusan pembelian.")
st.write("3.	Category : Kategori produk atau jenis makanan yang ditawarkan, misalnya makanan roti, makanan barat, minuman, atau aneka nasi.")
st.write("4.	Display : Deskripsi visual atau tampilan produk di platform yang lebih  mengarah pada pengelompokkan jenis pemesanan yang diinginkan konsumen.")
st.write("5.	Product : Nama produk atau menu yang ditawarkan. Variabel ini memungkinkan analisis produk terlaris, mengidentifikasi menu favorit konsumen, serta melihat tren kuliner berdasarkan data penjualan.")
st.write("6.	Price/harga produk : Analisis variabel ini penting untuk memahami strategi penentuan harga oleh merchant dan bagaimana harga memengaruhi keputusan pembelian konsumen. Data ini juga bisa digunakan untuk analisis statistik deskriptif.")
st.write("7.	Discount : Diskon yang ditawarkan oleh merchant atau platform. Variabel ini sangat relevan untuk menganalisis peran promosi dalam menarik konsumen, mengidentifikasi efektivitas diskon, serta kaitannya dengan volume transaksi.")



data= pd.read_csv('gofood_dataset.csv')
pilihan_tampil_tabel = st.checkbox('Menampilkan lima data awal')
st.write(pilihan_tampil_tabel)
if pilihan_tampil_tabel == True:
    st.write("#### Lima Data Awal")
    st.write(data.head() )


##Ar
st.write("## Visualisasi")

st.write("#### A. Statistik Dasar Data Gofood di Kota Besar Indonesia")
st.write("Berikut merupakan statistik dasar dari Data Gofood di Kota Besar Indonesia:")
st.write( data.describe() )
st.write("Statistik dasar yang ditampilkan seperti nilai rata-rata, standar deviasi, nilai minimum, nilai maksimum, dan nilai kuartil dari masing-masing variable seperti price dan discount")


st.write("#### B. Informasi Diskon")
# Grouping data dan menghitung persentase
df = data.groupby(['merchant_area', 'isDiscount']).agg({'isDiscount': 'count'})
percen = data.groupby(['merchant_area']).agg({'isDiscount': 'count'})
df = df.div(percen, level='merchant_area') * 100
# Filter data yang memiliki discount_price
data['price'] = pd.to_numeric(data['price'], errors='coerce')
data['discount_price'] = pd.to_numeric(data['discount_price'], errors='coerce')
discprice = data[data['discount_price'].notnull()]
# Hitung persentase diskon
discprice['perc_disc'] = 100 * (discprice['price'] - discprice['discount_price']) / discprice['price']
# Kelompokkan data berdasarkan merchant_area dan hitung rata-rata
mean_disc = discprice.groupby('merchant_area', as_index=False)['perc_disc'].mean()
mean_disc.set_index('merchant_area', inplace=True)
# Atur layout Streamlit dengan dua kolom
col1, col2 = st.columns(2)
# Plot pertama (Jumlah Merchant)
with col1:
    colors = {
        ('jakarta', 0): 'yellow', ('jakarta', 1): 'purple',
        ('surabaya', 0): 'yellow', ('surabaya', 1): 'purple',
        ('medan', 0): 'yellow', ('medan', 1): 'purple'
    }
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    for index, value in df.iterrows():
        ax1.barh(str(index), value['isDiscount'], color=colors.get(index, 'gray'))
    ax1.set_title("Jumlah Merchant yang Memberikan Diskon Setiap Kota", fontsize=12)
    ax1.tick_params(axis='x', labelsize=10)
    ax1.tick_params(axis='y', labelsize=8)
    # Tambahkan legend
    purple_patch = mpatches.Patch(color='purple', label='Diskon')
    yellow_patch = mpatches.Patch(color='yellow', label='Tidak Diskon')
    ax1.legend(handles=[purple_patch, yellow_patch], loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)
    st.pyplot(fig1)
# Plot kedua (Rata-rata Persentase Diskon)
with col2:
    colors = {'jakarta': 'red', 'surabaya': 'blue', 'medan': 'green'}
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    mean_disc['perc_disc'].plot(kind='bar', color=[colors.get(area, 'gray') for area in mean_disc.index], ax=ax2)
    ax2.set_title("Rata-Rata Persentase Diskon Setiap Kota", fontsize=12)
    ax2.set_xlabel("")
    ax2.set_xticklabels(mean_disc.index, fontsize=10, rotation=0)
    ax2.set_ylabel("Persentase Diskon (%)")
    ax2.tick_params(axis='y', labelsize=10)
    st.pyplot(fig2)



st.write("#### C. WordCloud untuk Setiap Kategori")
# Daftar kategori yang ada pada data
most_cat = data['category'].unique().tolist()
# Filter data yang memiliki deskripsi
data_with_desc = data[data['description'].notnull()]
# Fungsi menggabungkan deskripsi menjadi satu string
def listToString(s): 
    return " ".join(s)
# Membuat dropdown untuk memilih kategori
selected_category = st.selectbox("Pilih Kategori yang akan ditampilkan", most_cat)
# Stopwords untuk WordCloud
stop_words = ["dengan", "mohon", "maaf", "tidak", "yang", "untuk", "agar", "nya", "ini", 
              "pcs", "ala", "and", "dan", "di", "the", "with"]
# Filter data berdasarkan kategori yang dipilih
filtered_data = data_with_desc[data_with_desc['category'].str.startswith(selected_category)]
combined_text = listToString(filtered_data['description'].values)
# Membuat dan menampilkan WordCloud
if combined_text:  # Pastikan deskripsi tidak kosong
    st.write(f"###### WordCloud untuk Kategori **{selected_category.capitalize()}**")
    wordcloud = WordCloud(
        background_color='white',
        max_words=500,
        max_font_size=100,
        random_state=1,
        stopwords=stop_words
    ).generate(combined_text)
    
    # Plot WordCloud
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)
else:
    st.warning(f"Tidak ada deskripsi untuk kategori **{selected_category.capitalize()}**.")


st.write("#### D. Customer Behavior Segmentation")
import streamlit as st
import pandas as pd
import plotly.express as px

# Data preprocessing
data[['Brand', 'Lokasi']] = data['merchant_name'].str.split(',', expand=True, n=2)
data.drop(['discount_price', 'description'], axis=1, inplace=True)
data = data.apply(lambda x: x.astype(str).str.lower())
data['Brand'] = data['Brand'].str.replace(r'[^\w\s]+', '', regex=True)
data['display'] = data['display'].str.replace(r'[^\w\s]+', '', regex=True)
data = data.apply(lambda x: x.astype(str).str.strip())
data_category = data.copy()

# Mengelompokkan data berdasarkan lokasi dan brand
df = pd.DataFrame(data.groupby('merchant_area')['Brand'].value_counts().unstack(fill_value=0).T)

# Mengelompokkan data berdasarkan kategori dan lokasi
df_category = data.groupby(['merchant_area', 'category']).size().reset_index(name='jumlah_merchant')

# Dropdown untuk memilih jenis visualisasi
visualization_option = st.selectbox('Pilih Visualisasi', ['Most Repeat Order Brand', 'Most Brand Merchant', 'Most Category Merchant'])

# Opsi multiselect untuk memilih kota
cities = ['jakarta', 'medan', 'surabaya']
selected_cities = st.multiselect("Pilih Kota", cities, default=cities)

if selected_cities:
    if visualization_option == 'Most Repeat Order Brand':
        for city in selected_cities:
            top_brands = df.sort_values([city], ascending=False).head(5)
            st.write(f"##### {city.capitalize()} - Most Repeat Order Brand")
            fig = px.line(top_brands, x=top_brands.index, y=city, title=f"Top Brands in {city.capitalize()}")
            st.plotly_chart(fig)

    elif visualization_option == 'Most Brand Merchant':
        df_brand = pd.crosstab(data['Brand'], data['merchant_area'])
        for city in selected_cities:
            top_brands = df_brand.sort_values([city], ascending=False).head(5)
            st.write(f"##### {city.capitalize()} - Most Brand Merchant")
            fig = px.line(top_brands, x=top_brands.index, y=city, title=f"Top Brand Merchant in {city.capitalize()}")
            st.plotly_chart(fig)

    elif visualization_option == 'Most Category Merchant':
        for city in selected_cities:
            top_categories = df_category[df_category['merchant_area'] == city].sort_values('jumlah_merchant', ascending=False).head(5)
            st.write(f"##### {city.capitalize()} - Most Category Merchant")
            fig = px.line(top_categories, x='category', y='jumlah_merchant', title=f"Top Categories in {city.capitalize()}")
            st.plotly_chart(fig)
else:
    st.write("Silakan pilih kota untuk menampilkan visualisasi.")




st.write("#### E. Cluster Data")
import plotly.graph_objects as go
from io import BytesIO
# Load data
data = pd.read_csv('gofood_dataset.csv')
# Pilih kolom yang relevan
if 'product' not in data.columns or 'merchant_area' not in data.columns:
    st.error("Data harus memiliki kolom 'product' dan 'merchant_area'!")
else:
    # Preprocessing kolom 'product'
    data['product'] = data['product'].str.lower()
    replacements = {
        'ayam': 'chicken', 'es': 'ice', 'dingin': 'ice', 'iced': 'ice', 'susu': 'milk', 'teh': 'tea',
        'choco': 'chocolate', 'chocolat': 'chocolate', 'chocolatee': 'chocolate', 'cokelat': 'chocolate',
        'coklat': 'chocolate', 'pisang': 'banana', 'keju': 'cheese', 'cheicee': 'cheese', 'kopi': 'coffee',
        'caffe': 'coffee', 'nasi': 'rice', 'telur': 'egg', 'telor': 'egg', 'beef': 'sapi', 'big': 'jumbo',
        'pcs': '', 'cm': ''
    }
    for key, value in replacements.items():
        data['product'] = data['product'].str.replace(key, value)

    # Cluster secara keseluruhan
    documents = data['product'].values.tolist()
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)

    k = 2
    model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
    model.fit(X)
    data['cluster'] = model.labels_


    # Dropdown untuk visualisasi cluster global
    st.write("###### Visualisasi Cluster Produk Secara Keseluruhan")
    cluster_option = st.selectbox("Pilih Klaster untuk Dilihat:", [f"Klaster {i}" for i in range(k)])
    selected_cluster = int(cluster_option.split()[-1])
    cluster_data = data[data['cluster'] == selected_cluster]

    # Graph chart visualisasi
    fig = px.bar(cluster_data['product'].value_counts().head(10), 
                 x=cluster_data['product'].value_counts().head(10).index, 
                 y=cluster_data['product'].value_counts().head(10).values, 
                 title=f"Top 10 Produk di {cluster_option}",
                 labels={"x": "Produk", "y": "Frekuensi"})
    st.plotly_chart(fig)

    # Cluster berdasarkan kota
    st.subheader("Clustering Produk Berdasarkan Kota")
    for city in data['merchant_area'].unique():
        with st.expander(f"Lihat Klaster Produk di Kota: {city}"):
            city_data = data[data['merchant_area'] == city]
            city_documents = city_data['product'].values.tolist()

            vectorizer = TfidfVectorizer(stop_words='english')
            X = vectorizer.fit_transform(city_documents)

            model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
            model.fit(X)
            city_data['cluster'] = model.labels_

            order_centroids = model.cluster_centers_.argsort()[:, ::-1]
            terms = vectorizer.get_feature_names_out()

            # Dropdown untuk klaster di kota tertentu
            cluster_option_city = st.selectbox(f"Pilih Klaster di {city}:", [f"Klaster {i}" for i in range(k)], key=city)
            selected_cluster_city = int(cluster_option_city.split()[-1])
            cluster_city_data = city_data[city_data['cluster'] == selected_cluster_city]

            # Graph chart visualisasi untuk klaster kota
            fig_city = px.bar(cluster_city_data['product'].value_counts().head(10), 
                              x=cluster_city_data['product'].value_counts().head(10).index, 
                              y=cluster_city_data['product'].value_counts().head(10).values, 
                              title=f"Top 10 Produk di {cluster_option_city} - {city}",
                              labels={"x": "Produk", "y": "Frekuensi"})
            st.plotly_chart(fig_city)
    # Tombol download hasil clustering
    st.subheader("Download Data Hasil Cluster")
    output_data = data.copy()
    output_csv = output_data.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Hasil Clustering", data=output_csv, file_name='clustered_gofood_data.csv',
                       mime='text/csv')








#Mba Izzah
st.write("## Analisis")
st.write("Buat analisis sederhana dari visualisasi data yang muncul di bagian sebelumnya.")

st.write("## Kesimpulan")
st.write("Tuliskan butir-butir kesimpulan dari analisis.")

st.write("## Referensi / Daftar Pustaka")
st.write("1. Ariq Syahalam, R. (2021). Indonesia food delivery Gofood product list. Kaggle. https://www.kaggle.com/datasets/ariqsyahalam/indonesia-food-delivery-gofood-product-list/code")
st.write(" 2.	‘Punya 1 Juta Mitra, GoFood Siap Bersaing-Halaman 1’, teknologi.bisnis.com, 2024 < https://teknologi.bisnis.com/read/20220131/266/1495410/punya-1-juta-mitra-gofood-siap-bersaing > [accessed 18 December 2024].")
st.write("3.	Larkhin Destamar, D., & Aryani, L. (n.d.). ANALISIS KEPUTUSAN PEMBELIAN PADA PENGGUNA APLIKASI GOJEK FITUR GOFOOD (Vol. 2).")
st.write("3.	Strategi Pemasaran Online dengan Aplikasi Gojek Menggunakan Fitur Gofood. (2024).")
st.write("4.	Widyaswara, I. W. E. (2018). Begini Potret Perpustakaan Daerah Bali di Era Milenial, Masih Pakai Cara Jadul di Zaman Digital - Halaman 2 - Tribun-bali.com. Bali.Tribunnews.Com.")
st.write("5.	 ‘Gofood Jadi Platform Paling Laku untuk Jual Makanan UMKM pada 2022-Halaman 1’, databoks.katadata.co.id, 2024 < https://databoks.katadata.co.id/produk-konsumen/statistik/a892971e0af01b9/gofood-jadi-platform-paling-laku-untuk-jual-makanan-umkm-pada-2022?utm_ > [accessed 18 December 2024].")
st.write("6.	 ‘Punya 1 Juta Mitra, GoFood Siap Bersaing-Halaman 1’, teknologi.bisnis.com, 2024 < https://teknologi.bisnis.com/read/20220131/266/1495410/punya-1-juta-mitra-gofood-siap-bersaing > [accessed 18 December 2024].")




st.title(" Terima Kasih! 😊🙏")
         
video_url = "https://www.youtube.com/watch?v=DB1UvpfPUVs"
# Tampilkan video
st.video(video_url)


