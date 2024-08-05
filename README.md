# Deep Learning ile Askeri Uçak Tiplerinin sınıflandırılması
- Projenin temel amacı uydu fotoğrafları üzerinden askeri uçak tiplerini tespit edebilen bir sınıflandırma modeli geliştirmektir. Çalışmalar MATLAB dilinde MATLAB R2022b ortamında yapılmıştır.
- Proje kapsamında Deep Learning ile CNN (Evrişimli sinir ağı) kullanılarak askeri uçak modellerinin sınıflandırılması sağlanmıştır.
- Projede önceden eğitilmiş VGG-16 ağ modeli kullanılmıştır.
- %95.7 doğruluk oranına sahip olan model, Cosine KNN sınıflandırma modeli ile elde edilmiştir.

## :rocket: Kullanım

Projeyi kullanmak için aşağıdaki adımları izleyebilirsiniz:

1. GitHub üzerinden veya yerel bilgisayarınıza klonlayarak proje dosyalarını indirin.
2. MATLAB ortamınızı açın.
3. MATLAB'da `aircraftClassification.m` dosyasını açın.
4. Dosyayı düzenleyerek veya doğrudan çalıştırarak modeli başlatın.
5. Modelin başarımlarını ve sonuçlarını MATLAB komut penceresinde gözlemleyin.
6. Classification Learner uygulaması kullanılarak farklı sınıflandırma algoritmaları için model değerlendirmeleri yapılmıştır.
7. Kodlar 4 bloğa ayrılmıştır.
   1. **Özellik Çıkarma:** VGG-16 modelini kullanarak uçak görüntülerinden özellik çıkarımı yapar. Görüntüler önce tam boyutta alınır ve ardından her biri dört eşit büyüklükte parçaya bölünür.
   2. **Özellik seçme:** özellik çıkarımının ardından özellik matrisi üzerinde PCA (Principal Component Analysis) analizi uygular. 
   3. **Tekli test:** test görüntüsü (test.jpg) üzerinde sınıflandırma testi gerçekleşir.
   4. **Toplu test:** TestAirCraft klasöründeki tüm JPG formatındaki test görüntülerini işler.

Detaylı yönergeler için dosya içindeki yorum satırlarını inceleyebilirsiniz.


## 📊 Veri Seti

Bu proje için kullanılan veri seti, Wu (2019) tarafından sağlanan "Muti-type Aircraft of Remote Sensing Images: MTARSI" veri setidir. Veri setine [Zenodo üzerinden buradan](https://doi.org/10.5281/zenodo.3464319) erişilebilir.
MTARSI, Google Earth uydu görüntülerinden elde edilen ve manuel olarak genişletilen, 36 havaalanını kapsayan 20 farklı uçak tipini içeren toplam 9'385 uzaktan algılama görüntüsüne sahiptir.

Projede kullanılmış olan Version v3, 20 uçak tipinden oluşmaktadır: 
B-1, B-2, B-29, B-52, Boeing, C-130, C-135, C-17, C-5, E-3, F-16, F-22, KC-10, C-21, U-2, A-10, A-26, P-63, T-6, T-43.

Tüm örnek görüntüler, uzaktan algılama görüntülerinin yorumlanması alanında uzman yedi kişi tarafından dikkatlice etiketlenmiştir.
Her görüntü bir ve sadece bir tam uçak içermektedir.

<img src="https://github.com/AbdulhalikAkdeniz/Aircraft-Classification-DeepLearning/assets/139945380/93f640fb-8ba7-42cb-a779-f323de47035b" width="600">

### Alıntı:
Wu, Z. (2019). Muti-type Aircraft of Remote Sensing Images: MTARSI [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3464319

## 🔍 Önerilen Yöntem
<img src="https://github.com/AbdulhalikAkdeniz/Aircraft-Classification-DeepLearning/assets/139945380/dd32a6bd-2d3f-44b9-aa46-813f95e9cb55" width="900">

Uçak türlerinin birbirinden ayırt edilmesi için önerilen bu derin öğrenme modelinde planlanan adımlar;
- **Veri hazırlığı** : Görüntü dosyaları her uçak türü için 1 den 20’ye kadar etiketlenmiştir.
- **Veri işleme ve yama bölme** : Görüntüler VGG-16 ağının giriş boyutuna uygun şekilde yeniden boyutlandırılarak özellik çıkarımı yapılmaktadır, ek olarak yamalara bölme işlemi uygulanmaktadır. Görüntüler 4 parçaya bölünür. Görüntünün tamamı için özellik çıkarımının yanında, her parçası için de ağ’dan özellik çıkarımı gerçekleşir.
- **Özellik çıkarma** : Tam parça görüntü ve 4 yamaya ait her parça için VGG-16 ağından özellik vektörleri dönmektedir. Özelliklerin çıkarılması ‘fc8’ katmanında gerçekleşmekte olup bu katman ağ’a ait tam bağlı katmandır ve 1000 özellik içeren vektör çıktısı vermektedir.
- **Birleştirme** : Tam görüntü ve 4 yama için özellik vektörleri, genel özellik matrisinde birleştirilmektedir. Yamalar için özellik çıkarma işlemleri bittiğinde genel özellik matrisi hazır olmalıdır.
- **Min-Max normalizasyonu** : Genel özellik matrisindeki değerlere min-max normalizasyonu uygulanmaktadır. Model performansının iyileştirilmesi amaçlanmıştır.
- **Özellik seçme** : Özellik seçimi ve boyut azaltma için PCA algoritması kullanılmıştır.
  PCA algoritması, içerisinde önemli özellikleri barındıracak şekilde matrisin boyutunu azaltır. Bu modelde ilk 1000 önemli özellik seçilmesi sağlanmıştır.
- **Sınıflandırma** : Genel özellik matrisi kullanılarak, sınıflandırıcı modeller için eğitim yapılır, en yüksek doğruluk oranı gösteren model KNN (K-Nearest Neighbors) algoritması, peşine LDA ve SVM algoritmaları gelmektedir.

## 📈 Deneysel Sonuçlar

<div style="text-align: center;">
    <img src="https://github.com/AbdulhalikAkdeniz/Aircraft-Classification-DeepLearning/assets/139945380/435849d4-0c76-4b6f-86d3-8f9352bac5c8" width="480">
    <p><strong>Cosine KNN</strong> sınıflandırma modeline ait Confusion Matrix</p>
</div>

- Bu sonuçlar, veri setinin %20'sinin doğrulama seti olarak kullanılmasıyla elde edilmiştir.

### Doğruluk oranları
- **Cosine KNN** sınıflandırıcısı için %95.7
- **Linear Discriminant Analysis** için %94.6
- **Neural Network - Wide Neural Network** için %94.2
- **Quadratic SVM** için %93.6
- **Cubic SVM** için %93.2


## 📧 İletişim
abdulhalikakdeniz08@gmail.com
