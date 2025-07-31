
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from infer import load_model_and_tokenizer, predict_properties

st.set_page_config(page_title="Polymer Property Predictor Using Regression Method", layout="wide")
st.title("🔬 TransPolymer - Property Prediction from SMILES")

@st.cache_resource
def get_model_and_tokenizer():
    return load_model_and_tokenizer()

model, tokenizer = get_model_and_tokenizer()
TARGETS = ['Eea', 'Egb', 'Egc', 'Ei', 'EPS', 'Nc']

tab1, tab2 = st.tabs(["🔹 Ramal Satu SMILES", "📂 Muat Naik CSV"])

# Tab 1: Single SMILES prediction
with tab1:
    smiles_input = st.text_input("🧪 Masukkan struktur SMILES:", placeholder="Contoh: CC(C)C(=O)O")

    if st.button("🔍 Ramal Sifat", key="predict_single") and smiles_input.strip() != "":
        with st.spinner("Sedang meramal..."):
            try:
                prediction = predict_properties(smiles_input, model, tokenizer)
                st.success("✅ Ramalan berjaya!")
                st.subheader("📈 Nilai Sifat Diramal:")
                st.bar_chart(pd.DataFrame(prediction, index=["Prediksi"]).T)
                st.dataframe(pd.DataFrame(prediction, index=["Prediksi"]).T)
            except Exception as e:
                st.error(f"Ralat: {e}")

# Tab 2: CSV upload
with tab2:
    st.write("Muat naik fail CSV dengan **lajur 'smiles'**.")
    uploaded_file = st.file_uploader("📁 Pilih fail CSV", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if 'smiles' not in df.columns:
                st.warning("❌ Fail tidak mengandungi lajur 'smiles'.")
            else:
                st.write("📄 Contoh data dimuat naik:")
                st.dataframe(df.head())

                results = []
                for smi in df['smiles']:
                    preds = predict_properties(smi, model, tokenizer)
                    preds['smiles'] = smi
                    results.append(preds)

                result_df = pd.DataFrame(results)
                st.success("✅ Ramalan berjaya untuk semua baris!")
                st.dataframe(result_df)

                st.download_button("💾 Muat Turun Keputusan", result_df.to_csv(index=False), file_name="ramalan_sifat.csv")
        except Exception as e:
            st.error(f"Ralat membaca fail: {e}")
