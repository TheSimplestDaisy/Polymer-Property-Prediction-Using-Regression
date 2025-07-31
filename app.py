import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from infer import load_model_and_tokenizer, predict_properties

# ✅ Cache model dan tokenizer agar tidak reload berulang kali
@st.cache_resource
def get_model_and_tokenizer():
    return load_model_and_tokenizer(
        model_path="ckpt/poc_fast_model.pt",
        tokenizer_path="tokenizer.pt"
    )

# ✅ Load model dan tokenizer
model, tokenizer = get_model_and_tokenizer()

# ✅ Setup UI
st.set_page_config(page_title="Polymer Property Predictor Using Regression")
st.title("🔬 TransPolymer - Property Prediction from SMILES")

# ✅ Input SMILES
smiles_input = st.text_input("Enter SMILES string for polymer:")

if smiles_input:
    try:
        result = predict_properties(model, tokenizer, smiles_input)
        st.success("Prediction complete!")

        # ✅ Display results
        st.subheader("📊 Predicted Properties:")
        df = pd.DataFrame(result.items(), columns=["Property", "Value"])
        st.dataframe(df)

        # ✅ Plot bar chart
        st.subheader("📈 Visualization:")
        st.bar_chart(df.set_index("Property"))

    except Exception as e:
        st.error(f"Prediction failed: {e}")
