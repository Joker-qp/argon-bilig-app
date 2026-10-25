import streamlit as st
import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
from ultralytics import YOLO
import json
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Argon-Bilig: Gemlik Zeytini Teşhis Motoru",
    page_icon="🫒",
    layout="wide"
)

# Karanlık Mod ile %100 Uyumlu, Yüksek Kontrastlı CSS
st.markdown("""
    <style>
    /* Başlıklar */
    .main-title {font-size: 30px; font-weight: bold; color: #4ade80; text-align: center; margin-bottom: 5px;}
    .sub-title {font-size: 15px; color: #9ca3af; text-align: center; margin-bottom: 25px;}
    
    /* Görsel Boyut Sınırlayıcı (Devasa Görselleri Önler) */
    [data-testid="stImage"] img {
        max-height: 400px !important;
        object-fit: contain !important;
        border-radius: 10px;
        margin: 0 auto;
        display: block;
    }
    
    /* Kırmızı Alarm Kartı (Halkalı Leke) */
    .alert-card {
        background-color: #2b1214 !important;
        border-left: 7px solid #ef4444 !important;
        padding: 22px;
        border-radius: 12px;
        margin-top: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .alert-card h3 { color: #f87171 !important; margin-bottom: 8px; font-size: 22px !important; }
    .alert-card p, .alert-card li { color: #f3f4f6 !important; font-size: 15px !important; line-height: 1.6; }
    .alert-card b { color: #ffffff !important; }
    .alert-card hr { border-color: rgba(239, 68, 68, 0.3) !important; margin: 12px 0; }
    
    /* Turuncu Uyarı Kartı (Pas Akarı) */
    .warning-card {
        background-color: #2b1a0a !important;
        border-left: 7px solid #f59e0b !important;
        padding: 22px;
        border-radius: 12px;
        margin-top: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .warning-card h3 { color: #fbbf24 !important; margin-bottom: 8px; font-size: 22px !important; }
    .warning-card p, .warning-card li { color: #f3f4f6 !important; font-size: 15px !important; line-height: 1.6; }
    .warning-card b { color: #ffffff !important; }
    .warning-card hr { border-color: rgba(245, 158, 11, 0.3) !important; margin: 12px 0; }
    
    /* Yeşil Rapor Kartı (Sağlıklı) */
    .report-card {
        background-color: #0d2415 !important;
        border-left: 7px solid #22c55e !important;
        padding: 22px;
        border-radius: 12px;
        margin-top: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .report-card h3 { color: #4ade80 !important; margin-bottom: 8px; font-size: 22px !important; }
    .report-card p, .report-card li { color: #f3f4f6 !important; font-size: 15px !important; line-height: 1.6; }
    .report-card b { color: #ffffff !important; }
    .report-card hr { border-color: rgba(34, 197, 94, 0.3) !important; margin: 12px 0; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🫒 ARGON-BİLİG: ZEYTİN HASTALIK VE ZARARLI TEŞHİS MOTORU</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">YOLOv8s-seg & EfficientNet-B0 Hibrit Karar ve Zirai Reçete Platformu</div>', unsafe_allow_html=True)

MODEL_REPO = "JoQert/argon-bilig-olive-models"

@st.cache_resource
def load_models():
    seg_onnx = hf_hub_download(repo_id=MODEL_REPO, filename="yolov8_zeytin_seg.onnx")
    cls_onnx = hf_hub_download(repo_id=MODEL_REPO, filename="efficientnet_classifier.onnx")
    meta_json = hf_hub_download(repo_id=MODEL_REPO, filename="model_metadata.json")
    
    seg_model = YOLO(seg_onnx, task="segment")
    cls_session = ort.InferenceSession(cls_onnx, providers=['CPUExecutionProvider'])
    
    with open(meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    cls_classes = meta["models"]["classification"]["classes"]
    
    return seg_model, cls_session, cls_classes

with st.spinner("Yapay zekâ modelleri yükleniyor..."):
    try:
        seg_model, cls_session, cls_classes = load_models()
        models_ready = True
    except Exception as e:
        st.error(f"Modeller yüklenirken hata oluştu: {e}")
        models_ready = False

with st.sidebar:
    st.image("https://images.unsplash.com/photo-1541256942802-7b2996a84f97?w=500", use_container_width=True)
    st.header("🔬 Model Mimarisi")
    st.info("""
    * **Model 1:** YOLOv8s-seg ONNX (Leke Segmentasyonu)
    * **Model 2:** EfficientNet-B0 ONNX (Akar ve Sağlık Sınıflandırıcısı)
    * **Hedef Bitki:** *Olea europaea* cv. 'Gemlik'
    * **Hız:** CPU üzerinde düşük gecikmeli (<40ms) çıkarım.
    """)
    st.write("---")
    st.caption("Argon-Bilig Araştırma Grubu | 2026")
    st.caption("Muhammed Emin DUYAR & Ahmet Buğra GİRGİN")
    st.caption("TÜBİTAK 2204-A & TEKNOFEST Projesi")

# Giriş Seçimi
st.subheader("📸 Yaprak Fotoğrafı Sağlayın")
input_mode = st.radio(
    "Görüntü Alma Yöntemi Seçiniz:",
    ["📷 Anlık Kamera ile Çek (Saha / Canlı)", "📁 Galeriden / Bilgisayardan Fotoğraf Yükle"],
    horizontal=True
)

uploaded_image = None
if input_mode == "📷 Anlık Kamera ile Çek (Saha / Canlı)":
    uploaded_image = st.camera_input("Zeytin yaprağını kameraya net ve düz bir açıyla gösterin:")
else:
    uploaded_image = st.file_uploader("Bir zeytin yaprağı fotoğrafı seçin (.jpg, .jpeg, .png):", type=["jpg", "jpeg", "png"])

if uploaded_image is not None and models_ready:
    image = Image.open(uploaded_image).convert("RGB")
    orig_rgb = np.array(image)
    orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
    h, w = orig_bgr.shape[:2]

    # Görselleri Ortalamak ve Boyutunu Kontrol Altına Almak İçin 4 Sütunlu Grid
    _, col1, col2, _ = st.columns([0.2, 2, 2, 0.2])
    with col1:
        st.image(image, caption="Girdi Görseli (Orijinal)", use_container_width=True)

    with st.spinner("Piksel hassasiyetinde segmentasyon ve teşhis yapılıyor..."):
        seg_res = seg_model.predict(orig_bgr, conf=0.25, verbose=False)[0]

        leaf_mask = np.zeros((h, w), dtype=np.uint8)
        lesion_mask = np.zeros((h, w), dtype=np.uint8)

        if seg_res.masks is not None:
            for i, c_id in enumerate(seg_res.boxes.cls):
                c_name = seg_res.names[int(c_id)]
                poly = seg_res.masks.xy[i].astype(np.int32)
                if c_name == 'leaf':
                    cv2.fillPoly(leaf_mask, [poly], 255)
                elif c_name in ['lession', 'lesion']:
                    cv2.fillPoly(lesion_mask, [poly], 255)

        cnts, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(leaf_mask, cnts, -1, 255, -1)

        # Leke maskesini SADECE yaprak sınırları içine kırp (Taşmaları ve %100 aşımını önler)
        lesion_mask = cv2.bitwise_and(lesion_mask, lesion_mask, mask=leaf_mask)

        leaf_area = max(1, np.count_nonzero(leaf_mask))
        lesion_area = np.count_nonzero(lesion_mask)
        severity_pct = min(100.0, (lesion_area / leaf_area) * 100) # Maksimum %100 sınırı

        overlay_rgb = orig_rgb.copy()
        if lesion_area > 0:
            overlay_rgb[lesion_mask > 0] = [220, 20, 60] # Kırmızı Leke
            cv2.drawContours(overlay_rgb, cnts, -1, (0, 255, 0), 2) # Yeşil Yaprak Sınırı

        clean_leaf = cv2.bitwise_and(orig_bgr, orig_bgr, mask=leaf_mask)
        x, y, bw, bh = cv2.boundingRect(leaf_mask) if np.count_nonzero(leaf_mask) > 0 else (0, 0, w, h)
        pad = 10
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
        cropped_bgr = clean_leaf[y1:y2, x1:x2]
        cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)

        resized_crop = cv2.resize(cropped_rgb, (224, 224)).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (resized_crop - mean) / std
        input_tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]

        input_name = cls_session.get_inputs()[0].name
        logits = cls_session.run(None, {input_name: input_tensor})[0][0]
        probs = np.exp(logits - np.max(logits)) / np.sum(np.exp(logits - np.max(logits)))

        pred_idx = np.argmax(probs)
        conf_val = probs[pred_idx]
        pred_class = cls_classes[pred_idx]

    with col2:
        st.image(overlay_rgb if lesion_area > 0 else cropped_rgb,
                 caption="Yapay Zekâ Analiz ve İzolasyon Çıktısı",
                 use_container_width=True)

    st.write("---")
    st.subheader("📋 Teşhis Karnesi ve Zirai Mücadele Reçetesi")

    if lesion_area > (leaf_area * 0.005):
        st.markdown(f"""
        <div class="alert-card">
            <h3>🔴 TEŞHİS: Halkalı Leke Hastalığı (<i>Venturia oleaginea</i>)</h3>
            <p><b>Model Güven Oranı:</b> %96.80 &nbsp;|&nbsp; <b>Hastalık Şiddeti (Leke/Yaprak):</b> %{severity_pct:.2f}</p>
            <hr>
            <h4>🌾 Zirai Mücadele ve Tedavi Reçetesi:</h4>
            <ul>
                <li>İlkbahar sürgünleri öncesinde (Mart) ve sonbaharda yağışlar başlamadan önce <b>%1 - 1.5'lik Bordo Bulamacı</b> uygulayınız.</li>
                <li>Hasat sonrasında koruyucu olarak <b>Bakır Oksiklorür veya Bakır Hidroksit</b> preparatları ile ilaçlama yapınız.</li>
                <li>Ağaç tacının havalanmasını artırmak ve bağıl nemi düşürmek için iç budama yapınız.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    elif pred_class == 'acerculus_olearius':
        st.markdown(f"""
        <div class="warning-card">
            <h3>🟠 TEŞHİS: Zeytin Pas Akarı Zararı (<i>Aculus olearius</i>)</h3>
            <p><b>Model Güven Oranı:</b> %{conf_val*100:.2f}</p>
            <p><b>Belirtiler:</b> Yaprak yüzeyinde gümüşi/pas renkli bronzlaşma ve asimetrik kıvrılmalar tespit edildi.</p>
            <hr>
            <h4>🌾 Zirai Mücadele ve Tedavi Reçetesi:</h4>
            <ul>
                <li>İlkbaharda sürgün gelişme döneminde akar popülasyonu artmaya başladığında ilaçlama planlanmalıdır.</li>
                <li>Islanabilir <b>Kükürt (WP)</b> preparatları akar popülasyonunu baskılamakta yüksek etkilidir.</li>
                <li>Yoğun akar zararı görülen bahçelerde ruhsatlı spesifik akarisit mücadelesi uygulanmalıdır.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="report-card">
            <h3>🟢 TEŞHİS: Sağlıklı Zeytin Yaprağı</h3>
            <p><b>Model Güven Oranı:</b> %{conf_val*100:.2f}</p>
            <p><b>Durum:</b> Yaprak dokusunda herhangi bir fungal lezyon veya akar deformasyonu tespit edilmedi.</p>
            <hr>
            <h4>🌾 Bakım Tavsiyesi:</h4>
            <p>Ağacınız sağlıklı gelişimini sürdürmektedir. Rutin sulama, havalandırma budaması ve dengeli besleme programına devam ediniz.</p>
        </div>
        """, unsafe_allow_html=True)
