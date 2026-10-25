# 🫒 Argon-Bilig: Gemlik Zeytini Yapay Zekâ Teşhis ve Karar Portalı

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://argon-bilig-app.streamlit.app)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/JoQert/argon-bilig-olive-models)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/JoQert/argon-bilig-olive-dataset)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

> **TÜBİTAK 2204-A Lise Öğrencileri Araştırma Projeleri Yarışması** ve **TEKNOFEST Tarımsal Teknolojiler** kapsamında geliştirilmiş uçtan uca hibrit yapay zekâ karar destek sistemi.

---

## 🌐 Canlı Web Uygulaması ve Bağlantılar
* 🚀 **Canlı Test Platformu:** [argon-bilig-app.streamlit.app](https://argon-bilig-app.streamlit.app)
* 🧠 **ONNX Model Deposu:** [JoQert/argon-bilig-olive-models](https://huggingface.co/JoQert/argon-bilig-olive-models)
* 📦 **Referans Veri Seti:** [JoQert/argon-bilig-olive-dataset](https://huggingface.co/datasets/JoQert/argon-bilig-olive-dataset)

---

## 👥 Proje Geliştiricileri ve Künye
* **Muhammed Emin DUYAR**
* **Ahmet Buğra GİRGİN**
* **Yıl:** 2026

---

## 📌 Projenin Amacı ve Özgün Değeri
Gemlik (*Olea europaea* cv. 'Gemlik') zeytini üretiminde rekolte kaybına yol açan en kritik iki etmen **Halkalı Leke (*Venturia oleaginea*)** mantarı ve **Zeytin Pas Akarı (*Aculus olearius*)** zararlısıdır.

Mevcut tarımsal sınıflandırma modelleri arka plandaki toprak, ahşap masa veya el gibi gürültüleri (noise) yaprak zannederek yanlış çıkarımlar yapmaktadır. **Argon-Bilig**, iki aşamalı hibrit mimarisiyle:
1. Yaprağı ve lezyonları **YOLOv8s-seg** ile piksel hassasiyetinde segmente eder,
2. Arka planı gürültüsüz siyah maskeyle izole eder,
3. Morfolojik deformasyonları **EfficientNet-B0** ile sınıflandırarak fenolojik döneme uygun **Zirai Mücadele Reçetesi** üretir.

---

## 🧠 Sistem Mimarisi ve Akış Şeması

```
       [Ham Yaprak Fotoğrafı (Kamera / Dosya)]
                         │
                         ▼
             [Model 1: YOLOv8s-seg ONNX]
             (Piksel Seviyesinde Segmentasyon)
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
 [Leke Alanı > %0.5]             [Belirgin Leke Yok]
         │                               │
         ▼                               ▼
[Halkalı Leke Teşhisi]            [Arka Plan İzolasyonu]
  + Şiddet Analizi (%)                   │
  + Bakırlı Reçete                       ▼
                            [Model 2: EfficientNet-B0 ONNX]
                            (Akar ve Sağlık Sınıflandırması)
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
                 [Zeytin Pas Akarı]             [Sağlıklı Yaprak]
                 + Kükürtlü Reçete              + Koruyucu Bakım
```

---

## 🎯 Hedef Sınıflar ve Patojenler

| Sınıf Etiketi | Bilimsel Adı | Türkçe Karşılığı | Mücadele Reçetesi |
| :--- | :--- | :--- | :--- |
| `olive_peacock_spot` | *Venturia oleaginea* | Halkalı Leke | 
| `acerculus_olearius` | *Aculus olearius* | Zeytin Pas Akarı | 
| `Healthy` | *Olea europaea* | Sağlıklı Yaprak |

---

## ⚡ Teknik Üstünlükler ve Donanım Performansı
* **Ultra Hafiflik:** Toplam hibrit sistem boyutu **~39 MB**'tır.
* **Düşük Gecikme:** Standart CPU üzerinde çıkarım süresi **<40 ms**'dir.
* **Kenar Cihaz (Edge AI) Uyumu:** İnternet erişimi olmayan zeytinliklerde mobil veya gömülü cihazlarda (Raspberry Pi/Jetson) çevrimdışı (offline) çalışabilir.

---

## 💻 Yerel Kurulum ve Çalıştırma (Local Run)

```bash
# 1. Depoyu klonlayın
git clone https://github.com/Joker-qp/argon-bilig-app.git
cd argon-bilig-app

# 2. Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt

# 3. Streamlit uygulamasını başlatın
streamlit run app.py
```

---

## 📜 Lisans ve Akademik Atıf (Citation)

Bu proje ve türetilmiş veri seti **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** lisansı altındadır.

Projeyi çalışmalarınızda referans göstermek için:

```bibtex
@misc{argonbilig2026,
  title={Argon-Bilig: Gemlik Zeytini Hastalık ve Zararlıları İçin Hibrit Yapay Zekâ Teşhis Motoru},
  author={Duyar, Muhammed Emin and Girgin, Ahmet Bu{\u{g}}ra},
  year={2026},
  howpublished={GitHub: \url{https://github.com/Joker-qp/argon-bilig-app}},
  note={Streamlit Karar Portalı ve ONNX Çıkarım Motoru}
}
```

---

## 🙏 Veri Kaynakları ve Teşekkür (Acknowledgements)
Öncül ham verilerin sağlanmasındaki katkıları için teşekkür ederiz:
* **Olive Peacock Eye Dataset:** Diker, A., Elen, A., Közkurt, C. et al. (2024). *Eur Food Res Technol*, 250, 287–299. [DOI: 10.1007/s00217-023-04386-8](https://doi.org/10.1007/s00217-023-04386-8)
* **Olive Leaf Disease Unified Dataset:** Abeer Saleh (CC0: Public Domain)
* **Olive Leaf Image Dataset:** Habibul Basher (CC0: Public Domain)
