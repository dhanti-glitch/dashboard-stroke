# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency
import numpy as np
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Stroke Risk Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    df = pd.read_csv("stroke_dataset_cleaned_final.csv")
    return df

df = load_data()

st.sidebar.image("https://img.icons8.com/color/96/brain.png", width=80)
st.sidebar.title("🧠 Stroke Risk Dashboard")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigasi",
    [
        "🤖 Prediksi Stroke",
        "📊 Overview",
        "🔍 EDA & Distribusi",
        "⚠️ Faktor Risiko",
        "🧪 A/B Testing",
        "📋 Kesimpulan"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Total Data:** {len(df):,} baris")
st.sidebar.markdown(f"**Fitur:** {df.shape[1]} kolom")
stroke_pct = df['stroke'].mean() * 100
st.sidebar.markdown(f"**Prevalensi Stroke:** {stroke_pct:.1f}%")

# Warna lebih cerah
STROKE_COLOR_MAP = {"Stroke": "#FF4B4B", "Tidak Stroke": "#00BFFF"}

def stroke_label(x):
    return "Stroke" if x == 1 else "Tidak Stroke"

df["stroke_label"] = df["stroke"].apply(stroke_label)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0: PREDIKSI STROKE (dipindah ke atas)
# ══════════════════════════════════════════════════════════════════════════════
if menu == "🤖 Prediksi Stroke":
    st.title("🤖 Prediksi Risiko Stroke")
    st.markdown("Masukkan data pasien di bawah ini untuk memprediksi risiko stroke.")
    st.markdown("---")

    col_input1, col_input2 = st.columns(2)

    with col_input1:
        age = st.slider("Usia (tahun)", 0, 100, 45)
        hypertension = st.selectbox("Hipertensi", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        heart_disease = st.selectbox("Penyakit Jantung", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        glucose = st.number_input("Kadar Glukosa (mg/dL)", 50.0, 300.0, 100.0)

    with col_input2:
        st.markdown("#### 📏 Hitung BMI Otomatis")
        berat = st.number_input("Berat Badan (kg)", 20.0, 200.0, 65.0)
        tinggi = st.number_input("Tinggi Badan (cm)", 100.0, 250.0, 165.0)
        tinggi_m = tinggi / 100
        bmi_calc = berat / (tinggi_m ** 2)
        st.metric("BMI Terhitung", f"{bmi_calc:.1f}")

        if bmi_calc < 18.5:
            st.info("Kategori: Underweight")
        elif bmi_calc < 25:
            st.success("Kategori: Normal")
        elif bmi_calc < 30:
            st.warning("Kategori: Overweight")
        else:
            st.error("Kategori: Obesitas")

    bmi = bmi_calc

    st.markdown("---")

    if st.button("🔍 Prediksi Sekarang", use_container_width=True):
        risk_score = (
            age * 0.03
            + glucose * 0.01
            + bmi * 0.01
            + hypertension * 2
            + heart_disease * 2
        )
        probability = min(risk_score / 10, 1)

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Probabilitas Stroke", f"{probability*100:.1f}%")
        col_r2.metric("BMI", f"{bmi:.1f}")
        col_r3.metric("Usia", f"{age} tahun")

        st.markdown("---")
        if probability > 0.5:
            st.error(f"⚠️ **Risiko Stroke TINGGI** ({probability*100:.1f}%)")
            st.markdown("Segera konsultasikan ke dokter dan lakukan pemeriksaan lebih lanjut.")
        else:
            st.success(f"✅ **Risiko Stroke RENDAH** ({probability*100:.1f}%)")
            st.markdown("Tetap jaga pola hidup sehat dan lakukan skrining rutin.")

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={"text": "Risiko Stroke (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#FF4B4B" if probability > 0.5 else "#00C853"},
                "steps": [
                    {"range": [0, 30], "color": "#C8F7C5"},
                    {"range": [30, 60], "color": "#FFF3CD"},
                    {"range": [60, 100], "color": "#FFCCCC"},
                ],
                "threshold": {"line": {"color": "black", "width": 4}, "thickness": 0.75, "value": 50}
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "📊 Overview":
    st.title("📊 Overview Dataset Stroke Prediction")
    st.markdown("Dataset: **Stroke Prediction Dataset** (Kaggle - fedesoriano) | Sudah melalui proses Data Wrangling")
    st.markdown("---")

    # KPI Cards — tanpa delta/panah pada Kasus Stroke
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Pasien", f"{len(df):,}")
    col2.metric("Kasus Stroke", f"{df['stroke'].sum():,}")   # ← panah dihapus
    col3.metric("Rata-rata Usia", f"{df['age'].mean():.0f} tahun")
    col4.metric("Rata-rata BMI", f"{df['bmi'].mean():.1f}")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Distribusi Kasus Stroke")
        stroke_counts = df['stroke_label'].value_counts().reset_index()
        stroke_counts.columns = ['Status', 'Jumlah']
        fig = px.pie(stroke_counts, values='Jumlah', names='Status',
                     color='Status',
                     color_discrete_map=STROKE_COLOR_MAP,
                     hole=0.4)
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Informasi Dataset")
        st.dataframe(
            pd.DataFrame({
                "Kolom": df.columns,
                "Tipe Data": df.dtypes.values,
                "Null": df.isnull().sum().values,
                "Unik": df.nunique().values
            }),
            use_container_width=True,
            height=350
        )

    st.markdown("---")
    st.subheader("Preview Data (5 baris pertama)")
    st.dataframe(df.drop(columns=['stroke_label']).head(), use_container_width=True)

    st.markdown("---")
    st.subheader("📖 Data Dictionary")
    dict_data = {
        'Fitur': ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                  'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
                  'smoking_status', 'stroke'],
        'Tipe': ['Kategorikal', 'Numerik', 'Biner', 'Biner', 'Kategorikal',
                 'Kategorikal', 'Kategorikal', 'Numerik', 'Numerik',
                 'Kategorikal', 'Biner (Target)'],
        'Deskripsi': [
            'Jenis kelamin pasien',
            'Usia pasien dalam tahun',
            '1 = memiliki hipertensi, 0 = tidak',
            '1 = memiliki penyakit jantung, 0 = tidak',
            'Status pernikahan',
            'Jenis pekerjaan pasien',
            'Tipe tempat tinggal pasien',
            'Rata-rata kadar glukosa dalam darah',
            'Body Mass Index — indeks massa tubuh',
            'Status merokok pasien',
            '1 = pernah stroke, 0 = tidak (variabel target)'
        ],
        'Nilai / Rentang': [
            'Male, Female',
            '0 – 82 tahun',
            '0, 1',
            '0, 1',
            'Yes, No',
            'Private, Self-employed, Govt_job, children, Never_worked',
            'Urban, Rural',
            '55.12 – 271.74 mg/dL',
            '10.3 – 97.6',
            'formerly smoked, never smoked, smokes, Unknown',
            '0, 1'
        ]
    }
    st.dataframe(pd.DataFrame(dict_data), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: EDA & DISTRIBUSI
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "🔍 EDA & Distribusi":
    st.title("🔍 Exploratory Data Analysis")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📈 Variabel Numerik", "🏷️ Variabel Kategorikal", "🔥 Korelasi"])

    # ── TAB 1: NUMERIK ──────────────────────────────────────────────────────
    with tab1:
        st.subheader("Distribusi Variabel Numerik")

        # Filter stroke/tidak stroke
        filter_stroke_num = st.radio(
            "Filter Status Stroke:",
            ["Semua", "Stroke", "Tidak Stroke"],
            horizontal=True,
            key="filter_num"
        )

        if filter_stroke_num == "Semua":
            df_num = df.copy()
        else:
            df_num = df[df["stroke_label"] == filter_stroke_num]

        num_cols = ['age', 'avg_glucose_level', 'bmi']
        labels_num = {'age': 'Usia', 'avg_glucose_level': 'Rata-rata Kadar Glukosa', 'bmi': 'BMI'}

        for col in num_cols:
            # Hitung bin 10-10 sesuai range data
            col_min = int(df[col].min())
            col_max = int(df[col].max()) + 10
            bin_edges = list(range(col_min - (col_min % 10), col_max, 10))

            fig = px.histogram(
                df_num, x=col,
                color='stroke_label' if filter_stroke_num == "Semua" else None,
                barmode='overlay',
                nbins=len(bin_edges),
                color_discrete_map=STROKE_COLOR_MAP,
                labels={col: labels_num[col], 'stroke_label': 'Status'},
                title=f"Distribusi {labels_num[col]}",
                text_auto=True   # ← angka tepat di atas bar
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(height=350, bargap=0.05,
                              xaxis=dict(dtick=10))   # ← interval 10
            st.plotly_chart(fig, use_container_width=True)

        # Boxplot DIHAPUS sesuai saran advisor

    # ── TAB 2: KATEGORIKAL ───────────────────────────────────────────────────
    with tab2:
        st.subheader("Distribusi Variabel Kategorikal")

        # Filter stroke/tidak stroke
        filter_stroke_cat = st.radio(
            "Filter Status Stroke:",
            ["Semua", "Stroke", "Tidak Stroke"],
            horizontal=True,
            key="filter_cat"
        )

        if filter_stroke_cat == "Semua":
            df_cat = df.copy()
        else:
            df_cat = df[df["stroke_label"] == filter_stroke_cat]

        cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        labels_cat = {
            'gender': 'Jenis Kelamin', 'ever_married': 'Status Menikah',
            'work_type': 'Jenis Pekerjaan', 'Residence_type': 'Tipe Tempat Tinggal',
            'smoking_status': 'Status Merokok'
        }

        for col in cat_cols:
            if filter_stroke_cat == "Semua":
                grp = df_cat.groupby([col, 'stroke_label']).size().reset_index(name='count')
                fig = px.bar(
                    grp, x=col, y='count', color='stroke_label', barmode='group',
                    color_discrete_map=STROKE_COLOR_MAP,
                    labels={'count': 'Jumlah', col: labels_cat[col], 'stroke_label': 'Status'},
                    title=f"Distribusi {labels_cat[col]}",
                    text_auto=True
                )
            else:
                grp = df_cat.groupby(col).size().reset_index(name='count')
                fig = px.bar(
                    grp, x=col, y='count', barmode='group',
                    color_discrete_sequence=["#FF4B4B" if filter_stroke_cat == "Stroke" else "#00BFFF"],
                    labels={'count': 'Jumlah', col: labels_cat[col]},
                    title=f"Distribusi {labels_cat[col]} — {filter_stroke_cat}",
                    text_auto=True
                )
            fig.update_traces(textposition='outside')
            fig.update_layout(height=340)
            st.plotly_chart(fig, use_container_width=True)

    # ── TAB 3: KORELASI ──────────────────────────────────────────────────────
    with tab3:
        st.subheader("Heatmap Korelasi Antar Variabel Numerik")
        num_df = df[['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease', 'stroke']]
        corr = num_df.corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    linewidths=0.5, ax=ax, vmin=-1, vmax=1)
        ax.set_title("Heatmap Korelasi", fontsize=14, fontweight='bold')
        st.pyplot(fig)

        st.markdown("**Insight:** Variabel `age` memiliki korelasi tertinggi dengan `stroke`, diikuti `hypertension` dan `avg_glucose_level`.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: FAKTOR RISIKO
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "⚠️ Faktor Risiko":
    st.title("⚠️ Analisis Faktor Risiko Stroke")
    st.markdown("---")

    # Segmentasi Usia
    bins = [0, 18, 35, 50, 65, 100]
    labels_age = ['0-18', '19-35', '36-50', '51-65', '65+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels_age, right=False)

    # Feature Engineering — ditampilkan sebagai chart, bukan tabel
    df["health_risk_score"] = df["hypertension"] + df["heart_disease"]
    df["glucose_bmi_risk"] = df["avg_glucose_level"] * df["bmi"]

    st.subheader("🔬 Feature Engineering")
    st.markdown("Berikut adalah fitur turunan yang dibuat untuk analisis risiko:")

    col_fe1, col_fe2 = st.columns(2)

    with col_fe1:
        # Health Risk Score distribution
        hrs = df.groupby(['health_risk_score', 'stroke_label']).size().reset_index(name='count')
        hrs['health_risk_score'] = hrs['health_risk_score'].astype(str)
        fig_fe1 = px.bar(
            hrs, x='health_risk_score', y='count', color='stroke_label', barmode='group',
            color_discrete_map=STROKE_COLOR_MAP,
            labels={'health_risk_score': 'Health Risk Score (0–2)', 'count': 'Jumlah', 'stroke_label': 'Status'},
            title="Health Risk Score (Hipertensi + Penyakit Jantung)",
            text_auto=True
        )
        fig_fe1.update_traces(textposition='outside')
        fig_fe1.update_layout(height=350)
        st.plotly_chart(fig_fe1, use_container_width=True)
        st.caption("Skor 0 = tidak ada, 1 = salah satu, 2 = keduanya")

    with col_fe2:
        # Glucose x BMI risk by stroke
        fig_fe2 = px.box(
            df, x='stroke_label', y='glucose_bmi_risk', color='stroke_label',
            color_discrete_map=STROKE_COLOR_MAP,
            labels={'glucose_bmi_risk': 'Glukosa × BMI', 'stroke_label': 'Status'},
            title="Distribusi Glucose × BMI Risk per Status Stroke"
        )
        fig_fe2.update_layout(height=350)
        st.plotly_chart(fig_fe2, use_container_width=True)
        st.caption("Semakin tinggi nilai, semakin besar kombinasi risiko glukosa dan BMI")

    st.markdown("---")

    # Segmentasi Usia
    st.subheader("1. Segmentasi Kelompok Usia terhadap Risiko Stroke")
    age_stroke = df.groupby(['age_group', 'stroke_label']).size().reset_index(name='count')
    fig = px.bar(
        age_stroke, x='age_group', y='count', color='stroke_label', barmode='group',
        color_discrete_map=STROKE_COLOR_MAP,
        labels={'count': 'Jumlah', 'age_group': 'Kelompok Usia', 'stroke_label': 'Status'},
        title="Distribusi Stroke per Kelompok Usia",
        text_auto=True
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis=dict(dtick=1))
    st.plotly_chart(fig, use_container_width=True)

    # Rata-rata Glukosa per Kategori
    st.subheader("2. Rata-rata Kadar Glukosa per Faktor Risiko")
    cat_sel = st.selectbox("Pilih faktor:", ['work_type', 'smoking_status', 'ever_married', 'gender'],
                           format_func=lambda x: {
                               'work_type': 'Jenis Pekerjaan', 'smoking_status': 'Status Merokok',
                               'ever_married': 'Status Menikah', 'gender': 'Jenis Kelamin'
                           }[x])
    glc = df.groupby([cat_sel, 'stroke_label'])['avg_glucose_level'].mean().reset_index()
    glc['avg_glucose_level'] = glc['avg_glucose_level'].round(1)
    fig = px.bar(
        glc, x=cat_sel, y='avg_glucose_level', color='stroke_label', barmode='group',
        color_discrete_map=STROKE_COLOR_MAP,
        labels={'avg_glucose_level': 'Rata-rata Kadar Glukosa', 'stroke_label': 'Status'},
        title=f"Rata-rata Kadar Glukosa berdasarkan {cat_sel}",
        text='avg_glucose_level'
    )
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

    # Hipertensi & Penyakit Jantung
    st.subheader("3. Pengaruh Hipertensi & Penyakit Jantung")
    col1, col2 = st.columns(2)

    with col1:
        ht = df.groupby(['hypertension', 'stroke_label']).size().reset_index(name='count')
        ht['hypertension'] = ht['hypertension'].map({0: 'Tidak', 1: 'Ya'})
        fig = px.bar(ht, x='hypertension', y='count', color='stroke_label', barmode='group',
                     color_discrete_map=STROKE_COLOR_MAP,
                     title="Hipertensi vs Stroke",
                     labels={'hypertension': 'Hipertensi', 'count': 'Jumlah', 'stroke_label': 'Status'},
                     text_auto=True)
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        hd = df.groupby(['heart_disease', 'stroke_label']).size().reset_index(name='count')
        hd['heart_disease'] = hd['heart_disease'].map({0: 'Tidak', 1: 'Ya'})
        fig = px.bar(hd, x='heart_disease', y='count', color='stroke_label', barmode='group',
                     color_discrete_map=STROKE_COLOR_MAP,
                     title="Penyakit Jantung vs Stroke",
                     labels={'heart_disease': 'Penyakit Jantung', 'count': 'Jumlah', 'stroke_label': 'Status'},
                     text_auto=True)
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    # Scatter Plot
    st.subheader("4. Scatter Plot: Usia vs Kadar Glukosa")
    fig = px.scatter(df, x='age', y='avg_glucose_level', color='stroke_label',
                     color_discrete_map=STROKE_COLOR_MAP,
                     opacity=0.6, size_max=6,
                     labels={'age': 'Usia', 'avg_glucose_level': 'Kadar Glukosa', 'stroke_label': 'Status'},
                     title="Hubungan Usia dan Kadar Glukosa terhadap Risiko Stroke")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: A/B TESTING
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "🧪 A/B Testing":
    st.title("🧪 Implementasi A/B Testing & Confusion Matrix")
    st.markdown("---")

    tab_ab, tab_cm = st.tabs(["📊 A/B Testing", "🔲 Confusion Matrix"])

    # ── TAB A/B TESTING ───────────────────────────────────────────────────────
    with tab_ab:
        st.markdown("""
        Pengujian untuk melihat apakah terdapat hubungan signifikan antara
        hipertensi dan kejadian stroke.
        """)

        contingency = pd.crosstab(df["hypertension"], df["stroke"])
        st.subheader("Tabel Kontingensi")
        st.dataframe(contingency)

        chi2, p, dof, expected = chi2_contingency(contingency)

        col1, col2 = st.columns(2)
        col1.metric("Chi-Square", round(chi2, 3))
        col2.metric("P-Value", round(p, 6))

        if p < 0.05:
            st.success("✅ Terdapat hubungan **signifikan** antara hipertensi dan stroke (tolak H₀).")
        else:
            st.warning("⚠️ Tidak terdapat hubungan signifikan (gagal tolak H₀).")

        st.markdown("""
        **Hipotesis**
        - **H₀**: Tidak ada hubungan antara hipertensi dengan stroke
        - **H₁**: Ada hubungan antara hipertensi dengan stroke
        """)

    # ── TAB CONFUSION MATRIX ─────────────────────────────────────────────────
    with tab_cm:
        st.subheader("Confusion Matrix — Prediksi Model")
        st.markdown("Confusion matrix menunjukkan seberapa akurat prediksi model terhadap kondisi stroke pasien.")

        # Filter tampilan confusion matrix
        filter_cm = st.radio(
            "Tampilkan prediksi untuk:",
            ["Semua", "Prediksi: Stroke", "Prediksi: Tidak Stroke"],
            horizontal=True,
            key="filter_cm"
        )

        # Simulasi prediksi model (rule-based score seperti di halaman Prediksi)
        def predict_stroke(row):
            score = (
                row['age'] * 0.03
                + row['avg_glucose_level'] * 0.01
                + row['bmi'] * 0.01
                + row['hypertension'] * 2
                + row['heart_disease'] * 2
            )
            prob = min(score / 10, 1)
            return 1 if prob > 0.5 else 0

        df['y_pred'] = df.apply(predict_stroke, axis=1)
        df['y_pred_label'] = df['y_pred'].apply(lambda x: "Stroke" if x == 1 else "Tidak Stroke")

        # Confusion matrix values
        TP = len(df[(df['stroke'] == 1) & (df['y_pred'] == 1)])
        TN = len(df[(df['stroke'] == 0) & (df['y_pred'] == 0)])
        FP = len(df[(df['stroke'] == 0) & (df['y_pred'] == 1)])
        FN = len(df[(df['stroke'] == 1) & (df['y_pred'] == 0)])

        total = TP + TN + FP + FN
        accuracy = (TP + TN) / total if total > 0 else 0

        # Tampilkan subset sesuai filter
        if filter_cm == "Prediksi: Stroke":
            df_show = df[df['y_pred'] == 1][['age', 'hypertension', 'heart_disease',
                                             'avg_glucose_level', 'bmi', 'stroke_label', 'y_pred_label']]
            st.dataframe(df_show.head(20), use_container_width=True)
        elif filter_cm == "Prediksi: Tidak Stroke":
            df_show = df[df['y_pred'] == 0][['age', 'hypertension', 'heart_disease',
                                             'avg_glucose_level', 'bmi', 'stroke_label', 'y_pred_label']]
            st.dataframe(df_show.head(20), use_container_width=True)

        # Visualisasi Confusion Matrix
        cm_labels = [["True Positive\n(Stroke → Stroke)", "False Negative\n(Stroke → Tidak Stroke)"],
                     ["False Positive\n(Tidak Stroke → Stroke)", "True Negative\n(Tidak Stroke → Tidak Stroke)"]]
        cm_values = [[TP, FN], [FP, TN]]
        cm_text = [[f"{TP}\n(Benar: Stroke)", f"{FN}\n(Salah: Terlewat)"],
                   [f"{FP}\n(Salah: False Alarm)", f"{TN}\n(Benar: Tidak Stroke)"]]

        fig_cm = go.Figure(data=go.Heatmap(
            z=[[TP, FN], [FP, TN]],
            x=["Prediksi: Stroke", "Prediksi: Tidak Stroke"],
            y=["Aktual: Stroke", "Aktual: Tidak Stroke"],
            text=[[f"TP: {TP}", f"FN: {FN}"], [f"FP: {FP}", f"TN: {TN}"]],
            texttemplate="%{text}",
            textfont={"size": 16, "color": "white"},
            colorscale=[[0, "#00BFFF"], [0.5, "#FFA500"], [1, "#FF4B4B"]],
            showscale=False
        ))
        fig_cm.update_layout(
            title="Confusion Matrix",
            height=400,
            xaxis_title="Prediksi Model",
            yaxis_title="Aktual"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        # Metrik model
        st.markdown("---")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("✅ Akurasi", f"{accuracy*100:.1f}%")
        col_m2.metric("🔴 True Positive", TP)
        col_m3.metric("🟢 True Negative", TN)
        col_m4.metric("⚠️ False Negative", FN, help="Stroke yang tidak terdeteksi — perlu diminimalkan")

        st.markdown("""
        **Keterangan:**
        - **True Positive (TP)**: Model memprediksi Stroke, aktualnya memang Stroke ✅
        - **True Negative (TN)**: Model memprediksi Tidak Stroke, aktualnya memang Tidak Stroke ✅
        - **False Positive (FP)**: Model memprediksi Stroke, aktualnya Tidak Stroke ❌
        - **False Negative (FN)**: Model memprediksi Tidak Stroke, aktualnya Stroke ❌ *(lebih berbahaya)*
        """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: KESIMPULAN
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "📋 Kesimpulan":
    st.title("📋 Kesimpulan & Insight")
    st.markdown("---")

    insights = [
        ("🔴", "Usia adalah faktor risiko terkuat",
         "Pasien stroke rata-rata berusia lebih tua (~68 tahun) dibanding yang tidak stroke (~41 tahun). Risiko meningkat signifikan pada kelompok usia 51-65 dan 65+."),
        ("🟠", "Kadar glukosa tinggi meningkatkan risiko",
         "Pasien stroke memiliki rata-rata kadar glukosa lebih tinggi (~133 mg/dL) dibanding yang tidak stroke (~103 mg/dL). Kelompok dengan glukosa >125 mg/dL perlu perhatian khusus."),
        ("🟡", "Hipertensi berkontribusi signifikan",
         "Proporsi hipertensi pada pasien stroke (26.5%) jauh lebih tinggi dibanding yang tidak stroke (9.5%), menunjukkan korelasi yang kuat."),
        ("🟢", "Penyakit jantung memperparah risiko",
         "Pasien dengan riwayat penyakit jantung memiliki kemungkinan stroke yang lebih tinggi, meski prevalensinya relatif kecil dalam dataset."),
        ("🔵", "BMI berperan namun tidak dominan",
         "Rata-rata BMI pasien stroke sedikit lebih tinggi, namun perbedaannya tidak sebesar variabel usia atau glukosa."),
        ("🟣", "Status merokok dan pekerjaan berpengaruh",
         "Kelompok 'formerly smoked' dan 'Private worker' memiliki proporsi stroke lebih tinggi, kemungkinan terkait faktor usia yang lebih tua dalam kelompok tersebut."),
    ]

    for icon, title, desc in insights:
        with st.expander(f"{icon} {title}", expanded=True):
            st.write(desc)

    st.markdown("---")
    st.subheader("📌 Rekomendasi")
    st.markdown("""
    1. **Skrining rutin** untuk populasi usia >50 tahun, terutama yang memiliki hipertensi atau penyakit jantung.
    2. **Pengelolaan kadar glukosa** sebagai langkah pencegahan stroke, khususnya pada penderita diabetes.
    3. **Gaya hidup sehat** (berhenti merokok, menjaga BMI ideal) untuk mengurangi faktor risiko yang dapat dikontrol.
    4. **Model prediktif** menggunakan fitur: `age`, `avg_glucose_level`, `bmi`, `hypertension`, `heart_disease` sebagai fitur utama.
    """)

    st.markdown("---")
    st.caption("Dashboard dibuat untuk keperluan analisis data proyek Data Science | Stroke Prediction Dataset (Kaggle - fedesoriano)")
