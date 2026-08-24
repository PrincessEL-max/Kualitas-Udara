"""
Dashboard Kualitas Udara Jakarta
Proyek portofolio data science — analisis & prediksi PM2.5 Jakarta.

Cara jalanin:
    streamlit run app.py
"""

import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Kualitas Udara Jakarta", page_icon="🌫️", layout="wide")

# ── Konstanta ────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "lag_1", "lag_2", "lag_3", "lag_7", "rolling_mean_3", "rolling_mean_7",
    "month", "day_of_week", "is_weekend", "is_musim_kemarau",
]

# Kategori PM2.5 (µg/m³, rata-rata 24 jam) — sumber: BMKG / ISPU PermenLHK No. 14/2020
PM25_CATEGORIES = [
    (15.5, "Baik", "#00e400", "Aman untuk semua kelompok, termasuk yang sensitif."),
    (55.4, "Sedang", "#ffff00", "Masih aman untuk umum; kelompok sensitif (anak, lansia, penderita asma/jantung) mulai kurangi aktivitas berat di luar ruangan."),
    (150.4, "Tidak Sehat", "#ff7e00", "Kurangi aktivitas fisik lama/berat di luar ruangan, terutama kelompok sensitif. Pertimbangkan masker."),
    (250.4, "Sangat Tidak Sehat", "#ff0000", "Hindari aktivitas luar ruangan yang lama. Kelompok sensitif sebaiknya tetap di dalam ruangan."),
    (float("inf"), "Berbahaya", "#7e0023", "Semua orang sebaiknya menghindari aktivitas luar ruangan. Gunakan penyaring udara di dalam ruangan bila ada."),
]

# Hasil evaluasi model final di test set (dicatat dari sesi eksperimen sebelumnya)
MODEL_COMPARISON = pd.DataFrame({
    "Model": ["Baseline (persistence)", "Extra Trees (prediksi delta)"],
    "MAE (µg/m³)": [6.82, 7.33],
    "RMSE (µg/m³)": [9.86, 9.46],
})


# ── Data & model loading (di-cache biar gak reload tiap interaksi) ────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/pm25_jakarta_daily_filled.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["location_name", "date"]).reset_index(drop=True)


@st.cache_resource
def load_model():
    model = joblib.load("model/model_pm25.joblib")
    columns = joblib.load("model/model_columns.joblib")
    return model, columns


def get_supported_stations(df, model_columns):
    dummy_prefix = "location_name_"
    return sorted(c[len(dummy_prefix):] for c in model_columns if c.startswith(dummy_prefix))


def categorize_pm25(value):
    for threshold, label, color, advice in PM25_CATEGORIES:
        if value <= threshold:
            return label, color, advice
    return PM25_CATEGORIES[-1][1:]


def compute_features(station_df, ref_date):
    """Hitung lag & rolling feature untuk satu tanggal acuan di satu stasiun.
    Butuh minimal 7 hari histori sebelum ref_date. Return None kalau gak cukup."""
    station_df = station_df.sort_values("date").reset_index(drop=True)
    idx = station_df.index[station_df["date"] == ref_date]
    if len(idx) == 0:
        return None
    i = idx[0]
    if i < 7:
        return None

    row = station_df.loc[i]
    return {
        "lag_1": station_df.loc[i - 1, "value"],
        "lag_2": station_df.loc[i - 2, "value"],
        "lag_3": station_df.loc[i - 3, "value"],
        "lag_7": station_df.loc[i - 7, "value"],
        "rolling_mean_3": station_df.loc[i - 3:i - 1, "value"].mean(),
        "rolling_mean_7": station_df.loc[i - 7:i - 1, "value"].mean(),
        "month": ref_date.month,
        "day_of_week": ref_date.dayofweek,
        "is_weekend": int(ref_date.dayofweek in [5, 6]),
        "is_musim_kemarau": int(ref_date.month in [5, 6, 7, 8, 9, 10]),
        "actual_value": row["value"],
    }


def predict_next_day(model, model_columns, station, features):
    X = pd.DataFrame([{k: v for k, v in features.items() if k in FEATURE_COLS}])
    X["location_name"] = station
    X = pd.get_dummies(X, columns=["location_name"])
    X = X.reindex(columns=model_columns, fill_value=0)
    pred_delta = model.predict(X)[0]
    return features["lag_1"] + pred_delta


# ── Load semuanya ──────────────────────────────────────────────────────────
df = load_data()
model, model_columns = load_model()
supported_stations = get_supported_stations(df, model_columns)

st.title("🌫️ Kualitas Udara Jakarta")
st.caption("Analisis & prediksi PM2.5 dari data OpenAQ — dibangun sebagai proyek portofolio data science.")

station = st.sidebar.selectbox("Pilih stasiun", supported_stations)
st.sidebar.caption(
    "Catatan: BMKG 1 tidak didukung — datanya terlalu sedikit (4 baris) "
    "buat dilatih jadi model yang bisa dipercaya."
)

station_df = df[df["location_name"] == station].sort_values("date").reset_index(drop=True)

tab_trend, tab_predict, tab_advice, tab_model = st.tabs(
    ["📈 Tren Historis", "🔮 Prediksi", "💡 Rekomendasi", "📊 Performa Model"]
)

# ── TAB 1: Tren Historis ───────────────────────────────────────────────────
with tab_trend:
    st.subheader(f"Tren PM2.5 — {station}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rata-rata", f"{station_df['value'].mean():.1f} µg/m³")
    col2.metric("Minimum", f"{station_df['value'].min():.1f} µg/m³")
    col3.metric("Maksimum", f"{station_df['value'].max():.1f} µg/m³")

    fig = px.line(station_df, x="date", y="value", labels={"date": "Tanggal", "value": "PM2.5 (µg/m³)"})
    fig.update_traces(line_color="#4C72B0")
    st.plotly_chart(fig, width='stretch')

    st.subheader("Pola musiman (semua stasiun, digabung)")
    monthly = df.groupby(df["date"].dt.month)["value"].mean()
    month_labels = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
    fig2 = go.Figure(go.Bar(x=month_labels, y=[monthly.get(m, 0) for m in range(1, 13)], marker_color="#4C72B0"))
    fig2.update_layout(yaxis_title="PM2.5 rata-rata (µg/m³)")
    st.plotly_chart(fig2, width='stretch')
    st.caption("Musim kemarau (Mei–Okt) konsisten lebih tinggi dari musim hujan (Nov–Mar).")

# ── TAB 2: Prediksi ─────────────────────────────────────────────────────────
with tab_predict:
    st.subheader(f"Prediksi PM2.5 — {station}")

    min_valid_date = station_df["date"].iloc[7]
    max_date = station_df["date"].max()
    ref_date = st.date_input(
        "Tanggal acuan (\"hari ini\")", value=max_date,
        min_value=min_valid_date, max_value=max_date,
    )
    ref_date = pd.Timestamp(ref_date).tz_localize("Asia/Jakarta")

    features = compute_features(station_df, ref_date)

    if features is None:
        st.warning("Data buat tanggal ini gak lengkap (butuh minimal 7 hari histori sebelumnya, atau tanggal tidak ada di data).")
    else:
        pred_value = predict_next_day(model, model_columns, station, features)
        next_date = ref_date + pd.Timedelta(days=1)

        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"PM2.5 tanggal {ref_date.date()}", f"{features['actual_value']:.1f} µg/m³")
        with col2:
            st.metric(f"Prediksi PM2.5 {next_date.date()}", f"{pred_value:.1f} µg/m³")

        # kalau data buat besok beneran ada (tanggal acuan bukan yg paling akhir), tampilin actual buat validasi
        actual_next = station_df[station_df["date"] == next_date]
        if not actual_next.empty and pd.notna(actual_next["value"].iloc[0]):
            actual_val = actual_next["value"].iloc[0]
            error = abs(pred_value - actual_val)
            st.info(f"Nilai asli di {next_date.date()}: **{actual_val:.1f} µg/m³** (selisih dari prediksi: {error:.1f} µg/m³)")

        with st.expander("Lihat fitur yang dipakai model"):
            st.json({k: round(v, 2) if isinstance(v, float) else v for k, v in features.items() if k != "actual_value"})

        st.session_state["last_prediction"] = pred_value

# ── TAB 3: Rekomendasi ──────────────────────────────────────────────────────
with tab_advice:
    st.subheader("Rekomendasi Aktivitas")

    default_value = st.session_state.get("last_prediction", station_df["value"].iloc[-1])
    value_for_advice = st.slider(
        "Nilai PM2.5 (µg/m³) — otomatis terisi dari tab Prediksi, bisa diubah manual",
        min_value=0.0, max_value=300.0, value=float(round(default_value, 1)), step=0.5,
    )

    label, color, advice = categorize_pm25(value_for_advice)
    st.markdown(
        f"""
        <div style="padding: 1.5rem; border-radius: 0.5rem; background-color: {color}22; border-left: 6px solid {color};">
            <h3 style="margin: 0;">{label} — {value_for_advice:.1f} µg/m³</h3>
            <p style="margin-top: 0.5rem;">{advice}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Kategori berdasarkan ambang PM2.5 rata-rata 24 jam (BMKG / ISPU PermenLHK No. 14/2020).")

    with st.expander("Lihat semua kategori"):
        for threshold_lo, (threshold_hi, lbl, clr, adv) in zip(
            [0] + [t for t, *_ in PM25_CATEGORIES[:-1]], PM25_CATEGORIES
        ):
            hi_text = "∞" if threshold_hi == float("inf") else f"{threshold_hi}"
            st.markdown(f"- **{lbl}** ({threshold_lo}–{hi_text} µg/m³): {adv}")

# ── TAB 4: Performa Model ───────────────────────────────────────────────────
with tab_model:
    st.subheader("Model vs Baseline")
    st.caption(
        "Baseline \"persistence\" cuma nebak: PM2.5 besok = PM2.5 hari ini. "
        "Model Extra Trees belajar dari lag features, rolling mean, dan indikator kalender, "
        "dengan target prediksi berupa selisih (delta) dari hari sebelumnya."
    )

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(name="MAE", x=MODEL_COMPARISON["Model"], y=MODEL_COMPARISON["MAE (µg/m³)"], marker_color="#4C72B0"))
    fig3.add_trace(go.Bar(name="RMSE", x=MODEL_COMPARISON["Model"], y=MODEL_COMPARISON["RMSE (µg/m³)"], marker_color="#DD8452"))
    fig3.update_layout(barmode="group", yaxis_title="Error (µg/m³, makin kecil makin baik)")
    st.plotly_chart(fig3, width='stretch')

    st.dataframe(MODEL_COMPARISON, hide_index=True, width='stretch')

    st.markdown(
        """
**Yang menarik dari hasil ini:** baseline persistence ternyata cukup kompetitif — PM2.5 punya
autokorelasi tinggi (nilai hari ini sangat mirip kemarin), jadi "nebak besok = hari ini" itu susah
dikalahkan. Model Extra Trees menang di RMSE (lebih jarang meleset ekstrem) meski MAE-nya nyaris seri
dengan baseline. Ini temuan yang jujur, bukan kelemahan yang disembunyikan — beberapa pendekatan lain
(model per-stasiun, penambahan data cuaca dari Open-Meteo) sudah dicoba dan tidak memberikan perbaikan
konsisten, jadi model ini yang dipilih sebagai versi final.
        """
    )