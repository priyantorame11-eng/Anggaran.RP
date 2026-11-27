import pickle
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

try:
    import Orange
except ModuleNotFoundError as exc:
    if exc.name != "Orange":
        raise
    Orange = None

st.title("Dashboard Analitik DJPb")

# Sidebar Filter
st.sidebar.header("Menu Filter")
opsi = st.sidebar.selectbox("Pilih Model:", ["Klasifikasi", "Regresi"])

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "models" / "Best_model.pkcls"
    with model_path.open("rb") as model_file:
        return pickle.load(model_file)


model = load_model() if Orange is not None else None

data = pd.read_csv("data/02_realisasi_anggaran_klasifikasi.csv")
kategori_ikpa = [
    "Sangat Rendah (<70)",
    "Rendah (70–<80)",
    "Cukup (80–<85)",
    "Baik (85–<90)",
    "Sangat Baik (≥90)",
]
data["kategori_ikpa"] = pd.cut(
    data["skor_ikpa"],
    bins=[float("-inf"), 70, 80, 85, 90, float("inf")],
    labels=kategori_ikpa,
    right=False,
)

tab_prediksi, tab_visualisasi = st.tabs(["Menu Prediksi", "Visualisasi"])

with tab_prediksi:
    st.header("Prediksi Realisasi 95%")
    st.markdown("Masukkan data input berikut dan klik tombol untuk melihat hasil prediksi.")
    if Orange is None:
        st.error(
            "Orange3 belum tersedia pada interpreter aplikasi "
            f"({sys.executable}). Jalankan `uv sync` dari folder proyek, lalu "
            "mulai ulang dengan `uv run streamlit run streamlit_app.py`."
        )

    col1, col2 = st.columns(2)
    with col1:
        jumlah_spm = st.number_input(
            "Jumlah SPM",
            min_value=1,
            max_value=200,
            value=10,
            step=1,
            format="%d",
        )
        revisi_dipa = st.number_input(
            "Revisi DIPA",
            min_value=0,
            max_value=5,
            value=1,
            step=1,
            format="%d",
        )
        deviasi_rpd_persen = st.number_input(
            "Deviasi RPD (%)",
            min_value=0.00,
            max_value=30.00,
            value=5.50,
            format="%.2f",
        )
    with col2:
        skor_ikpa = st.slider(
            "Skor IKPA",
            min_value=70.00,
            max_value=100.00,
            value=85.00,
            step=0.01,
            format="%.2f",
        )
        tipe_satker = st.selectbox(
            "Tipe Satker",
            ["Dekonsentrasi", "Kantor Daerah", "Kantor Pusat", "Tugas Pembantuan"],
        )

    if st.button("Prediksi", type="primary", disabled=model is None):
        tipe_values = [
            "Dekonsentrasi",
            "Kantor Daerah",
            "Kantor Pusat",
            "Tugas Pembantuan",
        ]
        nilai_fitur = {
            "jumlah_spm": float(jumlah_spm),
            "revisi_dipa": float(revisi_dipa),
            "deviasi_rpd_persen": float(deviasi_rpd_persen),
            "skor_ikpa": float(skor_ikpa),
            "tipe_satker": tipe_satker,
            **{
                f"tipe_satker={tipe}": float(tipe_satker == tipe)
                for tipe in tipe_values
            },
        }

        try:
            fitur_tidak_dikenal = [
                variabel.name
                for variabel in model.domain.attributes
                if variabel.name not in nilai_fitur
            ]
            if fitur_tidak_dikenal:
                raise ValueError(
                    "Input belum tersedia untuk fitur model: "
                    + ", ".join(fitur_tidak_dikenal)
                )

            input_row = [
                nilai_fitur[variabel.name]
                for variabel in model.domain.attributes
            ]
            table = Orange.data.Table.from_list(model.domain, [input_row])
            prediction_index = int(model(table)[0])
            probabilities = model(table, model.Probs)[0]
            class_name = model.domain.class_var.values[prediction_index]
            confidence = float(probabilities[prediction_index])
            st.session_state["hasil_prediksi"] = (class_name, confidence)
        except Exception as exc:
            st.session_state.pop("hasil_prediksi", None)
            st.error(f"Prediksi gagal dijalankan: {exc}")

    if "hasil_prediksi" in st.session_state:
        class_name, confidence = st.session_state["hasil_prediksi"]
        st.subheader("Hasil Prediksi")
        if class_name == "Ya":
            st.success(f"Prediksi: {class_name} (keyakinan {confidence:.1%})")
        else:
            st.warning(f"Prediksi: {class_name} (keyakinan {confidence:.1%})")


with tab_visualisasi:
    st.header("Visualisasi Bubble Plot")
    st.markdown(
        "Grafik bubble menunjukkan skor IKPA terhadap deviasi RPD, dengan warna berdasarkan kolom `realisasi_tercapai_95persen`."
    )

    chart = (
        alt.Chart(data)
        .mark_circle(opacity=0.7, size=100)
        .encode(
            x=alt.X(
                "skor_ikpa",
                title="Skor IKPA",
                scale=alt.Scale(domain=[70, 100]),
            ),
            y=alt.Y("deviasi_rpd_persen", title="Deviasi RPD (%)"),
            color=alt.Color(
                "realisasi_tercapai_95persen:N",
                title="Realisasi Tercapai 95%",
                scale=alt.Scale(
                    domain=["Tidak", "Ya"],
                    range=["#ef4444", "#2563eb"],
                ),
            ),
            tooltip=[
                "kode_satker",
                "tipe_satker",
                "skor_ikpa",
                "kategori_ikpa",
                "deviasi_rpd_persen",
                "realisasi_tercapai_95persen",
            ],
        )
        .properties(width=900, height=550)
        .interactive()
    )
    st.altair_chart(chart, width=900)

    st.subheader("Distribusi Skor IKPA")
    histogram = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("skor_ikpa:Q", bin=alt.Bin(maxbins=20), title="Skor IKPA"),
            y=alt.Y("count():Q", title="Jumlah Satker"),
            tooltip=[
                alt.Tooltip("skor_ikpa:Q", bin=alt.Bin(maxbins=20), title="Rentang Skor"),
                alt.Tooltip("count():Q", title="Jumlah Satker"),
            ],
        )
        .properties(width=900, height=300)
    )
    st.altair_chart(histogram, width=900)

    st.subheader("Jumlah Satker per Kategori IKPA")
    histogram_kategori = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(
                "kategori_ikpa:N",
                title="Kategori Skor IKPA",
                sort=kategori_ikpa,
                axis=alt.Axis(labelAngle=-25),
            ),
            y=alt.Y("count():Q", title="Jumlah Satker"),
            color=alt.Color(
                "kategori_ikpa:N",
                title="Kategori IKPA",
                sort=kategori_ikpa,
                scale=alt.Scale(
                    domain=kategori_ikpa,
                    range=["#b91c1c", "#f97316", "#eab308", "#22c55e", "#2563eb"],
                ),
            ),
            tooltip=[
                alt.Tooltip("kategori_ikpa:N", title="Kategori IKPA"),
                alt.Tooltip("count():Q", title="Jumlah Satker"),
            ],
        )
        .properties(width=900, height=350)
    )
    st.altair_chart(histogram_kategori, width=900)

    st.subheader("Peringkat IKPA Satker")
    kolom_peringkat = [
        "kode_satker",
        "skor_ikpa",
        "kategori_ikpa",
        "nama_kementerian",
        "provinsi",
        "tipe_satker",
    ]
    tertinggi, terendah = st.columns(2)
    with tertinggi:
        st.markdown("**10 Satker dengan Skor IKPA Tertinggi**")
        st.dataframe(
            data.nlargest(10, "skor_ikpa")[kolom_peringkat]
            .rename(
                columns={
                    "skor_ikpa": "Skor IKPA",
                    "kategori_ikpa": "Kategori IKPA",
                }
            )
            .reset_index(drop=True),
            hide_index=True,
            use_container_width=True,
        )
    with terendah:
        st.markdown("**10 Satker dengan Skor IKPA Terendah**")
        st.dataframe(
            data.nsmallest(10, "skor_ikpa")[kolom_peringkat]
            .rename(
                columns={
                    "skor_ikpa": "Skor IKPA",
                    "kategori_ikpa": "Kategori IKPA",
                }
            )
            .reset_index(drop=True),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Peta Persebaran Satker dan Skor IKPA")
    koordinat_provinsi = {
        "Aceh": (5.55, 95.32),
        "Bali": (-8.65, 115.22),
        "DKI Jakarta": (-6.21, 106.85),
        "Jawa Barat": (-6.91, 107.61),
        "Jawa Tengah": (-6.99, 110.42),
        "Jawa Timur": (-7.25, 112.75),
        "Kalimantan Timur": (-0.50, 117.15),
        "Lampung": (-5.43, 105.26),
        "NTB": (-8.58, 116.10),
        "NTT": (-10.18, 123.61),
        "Papua": (-2.53, 140.72),
        "Riau": (0.51, 101.45),
        "Sulawesi Selatan": (-5.14, 119.41),
        "Sumatera Barat": (-0.95, 100.35),
        "Sumatera Utara": (3.59, 98.67),
    }
    peta = (
        data.groupby("provinsi", as_index=False)
        .agg(
            jumlah_satker=("kode_satker", "count"),
            skor_ikpa_rata_rata=("skor_ikpa", "mean"),
        )
    )
    peta["latitude"] = peta["provinsi"].map(
        {provinsi: lokasi[0] for provinsi, lokasi in koordinat_provinsi.items()}
    )
    peta["longitude"] = peta["provinsi"].map(
        {provinsi: lokasi[1] for provinsi, lokasi in koordinat_provinsi.items()}
    )

    provinsi_tanpa_koordinat = peta.loc[
        peta[["latitude", "longitude"]].isna().any(axis=1), "provinsi"
    ].tolist()
    if provinsi_tanpa_koordinat:
        st.error(
            "Koordinat peta belum tersedia untuk provinsi: "
            + ", ".join(provinsi_tanpa_koordinat)
        )
    else:
        peta["status_ikpa"] = peta["skor_ikpa_rata_rata"].lt(85).map(
            {True: "Perlu perhatian", False: "Di atas batas perhatian"}
        )
        peta["warna_skor"] = peta["status_ikpa"].map(
            {"Perlu perhatian": "#ef4444", "Di atas batas perhatian": "#2563eb"}
        )
        peta["ukuran_marker"] = peta["jumlah_satker"] * 2000
        st.caption(
            "Titik merah menandai rata-rata skor IKPA di bawah 85 (perlu perhatian); "
            "titik biru menandai skor 85 ke atas. Ukuran titik menunjukkan jumlah satker."
        )
        st.map(
            peta,
            latitude="latitude",
            longitude="longitude",
            color="warna_skor",
            size="ukuran_marker",
            zoom=4,
            height=600,
        )
        peta["skor_ikpa_rata_rata"] = peta["skor_ikpa_rata_rata"].round(2)
        st.dataframe(
            peta[
                [
                    "provinsi",
                    "jumlah_satker",
                    "skor_ikpa_rata_rata",
                    "status_ikpa",
                ]
            ]
            .rename(
                columns={
                    "provinsi": "Provinsi",
                    "jumlah_satker": "Jumlah Satker",
                    "skor_ikpa_rata_rata": "Rata-rata Skor IKPA",
                    "status_ikpa": "Status",
                }
            )
            .sort_values("Jumlah Satker", ascending=False)
            .reset_index(drop=True),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Rekomendasi Akhir untuk Menteri")
    ringkasan_provinsi = (
        data.groupby("provinsi", as_index=False)
        .agg(rata_rata_ikpa=("skor_ikpa", "mean"))
        .sort_values("rata_rata_ikpa")
    )
    provinsi_prioritas = ringkasan_provinsi.loc[
        ringkasan_provinsi["rata_rata_ikpa"] < 85
    ]
    tiga_provinsi_terendah = ", ".join(
        f"{baris.provinsi} ({baris.rata_rata_ikpa:.2f})"
        for baris in ringkasan_provinsi.head(3).itertuples()
    )
    satker_capai_target = data["realisasi_tercapai_95persen"].eq("Ya").sum()
    persentase_capai_target = satker_capai_target / len(data) * 100

    st.markdown(
        f"Berdasarkan **{len(data)} satker** dalam dataset, rata-rata skor IKPA "
        f"adalah **{data['skor_ikpa'].mean():.2f}**. Sebanyak "
        f"**{len(provinsi_prioritas)} dari {len(ringkasan_provinsi)} provinsi** "
        "memiliki rata-rata skor di bawah 85. Provinsi dengan rata-rata terendah: "
        f"**{tiga_provinsi_terendah}**. Indikator realisasi mencapai 95% berstatus "
        f"Ya pada **{satker_capai_target} satker ({persentase_capai_target:.1f}%)**."
    )
    st.markdown(
        f"""
1. **Prioritaskan pendampingan terarah** pada satker di provinsi dengan rata-rata IKPA di bawah 85. Mulai dari {tiga_provinsi_terendah}; telusuri skor dan indikator setiap satker untuk menentukan tindakan perbaikan yang sesuai.
2. **Perkuat pemantauan realisasi anggaran dan deviasi RPD.** Karena baru {persentase_capai_target:.1f}% satker berstatus mencapai realisasi 95%, minta unit kerja meninjau rencana penarikan dan progres realisasi secara berkala, lalu tindak lanjuti penyimpangan.
3. **Tetapkan pemilik aksi dan ukur kemajuan.** Pantau skor IKPA, pencapaian realisasi 95%, serta deviasi RPD per satker dalam evaluasi berkala; gunakan hasilnya untuk menyesuaikan pendampingan dan eskalasi.
4. **Validasi penyebab sebelum menetapkan kebijakan.** Data ini menunjukkan capaian dan perbedaan antarprovinsi, tetapi tidak membuktikan penyebab skor rendah. Verifikasi komponen penilaian dan konteks tiap satker sebelum memutuskan intervensi.
"""
    )