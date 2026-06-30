# Klasifikasi Tutupan Lahan Kabupaten Purwakarta Menggunakan Citra Landsat 8 dan Random Forest

Proyek ini bertujuan untuk melakukan klasifikasi tutupan lahan di wilayah Kabupaten Purwakarta, Jawa Barat, menggunakan citra satelit Landsat 8 Collection 2 Level-2 (Surface Reflectance) dan algoritma machine learning Random Forest. Klasifikasi dilakukan untuk mengidentifikasi lima kelas tutupan lahan: vegetasi, lahan terbuka, area permukiman, kawasan industri, dan badan air.

## Daftar Isi

- [Latar Belakang](#latar-belakang)
- [Data](#data)
- [Metodologi](#metodologi)
- [Struktur Proyek](#struktur-proyek)
- [Instalasi](#instalasi)
- [Alur Kerja](#alur-kerja)
- [Hasil](#hasil)
- [Referensi](#referensi)

## Latar Belakang

Pemantauan perubahan tutupan lahan merupakan aspek penting dalam perencanaan wilayah, terutama di daerah yang mengalami pertumbuhan urbanisasi dan industrialisasi seperti Kabupaten Purwakarta. Penginderaan jauh dengan citra Landsat 8 memungkinkan analisis tutupan lahan secara efisien dan dapat diulang secara berkala, dikombinasikan dengan algoritma klasifikasi berbasis machine learning seperti Random Forest yang dikenal robust terhadap data multidimensi dan noise.

## Data

| Komponen | Keterangan |
|---|---|
| Sumber citra | Google Earth Engine - koleksi `LANDSAT/LC08/C02/T1_L2` |
| Wilayah kajian | Kabupaten Purwakarta, Jawa Barat |
| Band yang digunakan | SR_B1 - SR_B7 (Coastal Aerosol hingga SWIR 2) |
| Resolusi spasial | 30 meter |
| Format file | GeoTIFF (.tif), satu file per band |
| Kelas tutupan lahan | Vegetation, Bare land, Residence area, Industry, Water bodies |

Setiap band diekspor secara terpisah dari Google Earth Engine menjadi file individual (`Purwakarta_SR_B1.tif` sampai `Purwakarta_SR_B7.tif`) dan kemudian digabungkan kembali menjadi array multiband pada tahap pra-pemrosesan di Python.

## Metodologi

Alur kerja proyek ini terdiri dari lima tahapan utama:

1. **Akuisisi dan pra-pemrosesan citra** — pembacaan band Landsat 8 SR menggunakan `rasterio`, penggabungan band menjadi array 3D `(bands, rows, cols)`, dan verifikasi rentang nilai reflectance.
2. **Komposit citra (visualisasi RGB)** — penyusunan komposit warna asli (true color) menggunakan kombinasi band Red (B4), Green (B3), dan Blue (B2) dengan teknik *contrast stretching* (`rescale_intensity`).
3. **Ekstraksi nilai piksel pada titik sampel** — pengambilan nilai reflectance pada koordinat titik latih dan uji menggunakan `rasterio.transform.rowcol`.
4. **Pelatihan model klasifikasi** — pembagian data menjadi data latih (70%) dan data uji (30%), pelatihan model `RandomForestClassifier` dari scikit-learn menggunakan tujuh band sebagai prediktor.
5. **Evaluasi dan prediksi spasial penuh** — evaluasi akurasi model menggunakan confusion matrix dan classification report, kemudian penerapan model pada seluruh piksel citra untuk menghasilkan peta klasifikasi tutupan lahan.

### Catatan Teknis Penting

- File Surface Reflectance yang diunduh dari Google Earth Engine pada proyek ini sudah berada dalam skala reflectance (bukan digital number mentah), sehingga **tidak diperlukan** penerapan ulang faktor skala `0.0000275` dan offset `-0.2` di sisi Python. Faktor skala tersebut hanya relevan apabila band diunduh dalam bentuk DN mentah (rentang nilai ribuan).
- Pada tahap rekonstruksi citra hasil prediksi dari array satu dimensi kembali ke dua dimensi, urutan `reshape` harus konsisten dengan urutan saat melakukan flatten pada tahap ekstraksi piksel (`reshape(bands, -1).T`), menggunakan dimensi baris dan kolom asli dari array citra — bukan dari hasil transpose atau concatenate yang dapat mengubah urutan dimensi secara tidak sengaja.

## Struktur Proyek

```
.
├── landsat_dir/
│   ├── Purwakarta_SR_B1.tif
│   ├── Purwakarta_SR_B2.tif
│   ├── ...
│   └── Purwakarta_SR_B7.tif
├── sample_extract.csv          # titik sampel beserta label kelas
├── klasifikasi_purwakarta.ipynb
└── README.md
```

## Instalasi

Proyek ini menggunakan Python dengan beberapa pustaka geospasial dan machine learning berikut:

```bash
pip install numpy pandas rasterio scikit-learn scikit-image matplotlib geopandas
```

## Alur Kerja

### 1. Membaca dan menggabungkan band

```python
stacked = []
for f in tif_files:
    with rio.open(f) as src:
        band = src.read(1).astype(float)
        stacked.append(band)
        landsat_transform = src.transform
landsat_image = np.stack(stacked, axis=0)  # shape: (7, rows, cols)
```

### 2. Membuat komposit warna asli (true color composite)

```python
from skimage.exposure import rescale_intensity

def stretch(band, p_low=2, p_high=98):
    lo, hi = np.percentile(band, (p_low, p_high))
    return rescale_intensity(band, in_range=(lo, hi), out_range=(0, 1))

red   = stretch(landsat_image[3])  # B4
green = stretch(landsat_image[2])  # B3
blue  = stretch(landsat_image[1])  # B2

arr_image = np.stack([red, green, blue], axis=-1)
plt.imshow(arr_image)
```

### 3. Pelatihan model Random Forest

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

prediksi = ["Purwakarta_SR_B1", "Purwakarta_SR_B2", "Purwakarta_SR_B3",
            "Purwakarta_SR_B4", "Purwakarta_SR_B5", "Purwakarta_SR_B6", "Purwakarta_SR_B7"]

train, test = train_test_split(sample_extract, train_size=0.7, random_state=2)

model = RandomForestClassifier()
model.fit(train[prediksi], train["value"])
```

### 4. Evaluasi model

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

test_apply = model.predict(test[prediksi])
mask = test["value"].notna()
y_true = test["value"][mask].astype(int)
y_pred = test_apply[mask.values].astype(int)

cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm).plot()

print(classification_report(y_true, y_pred, target_names=labels))
```

### 5. Prediksi spasial penuh

```python
bands, rows, cols = landsat_image.shape

table_image = pd.DataFrame(
    landsat_image.reshape(bands, -1).T,
    columns=band_names,
)

prediction = model.predict(table_image[prediksi])
prediction_image = prediction.reshape(rows, cols)

plt.figure(figsize=(10, 10))
plt.imshow(prediction_image, cmap=cmap, interpolation="nearest")
plt.legend(**legend)
```

## Hasil

Model menghasilkan peta klasifikasi tutupan lahan untuk Kabupaten Purwakarta dengan lima kelas:

- 🟩 **Vegetation** — area bervegetasi (sawah, kebun, hutan)
- 🟫 **Bare land** — lahan terbuka tanpa tutupan vegetasi
- 🟥 **Residence area** — kawasan permukiman
- 🟪 **Industry** — kawasan industri
- 🟦 **Water bodies** — badan air (sungai, waduk)

<img width="527" height="436" alt="image" src="https://github.com/user-attachments/assets/dd879c30-0148-4434-9d36-540ad4c913fc" />



## Referensi

- USGS, *Landsat 8-9 Collection 2 (C2) Level 2 Science Product Guide*.
- Google Earth Engine Data Catalog, *USGS Landsat 8 Level 2, Collection 2, Tier 1*.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
- Dokumentasi Kode dari GitHub Ramadhan (ramiqcom) *Land Cover Classification with Python* GitHub Profile : https://github.com/ramiqcom

---

**Adnan Yusuf Hartawan** Mahasiswa Teknik Geodesi, Universitas Diponegoro
