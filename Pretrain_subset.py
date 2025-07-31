import pandas as pd

# Baca data asal
df = pd.read_csv("data/pretrain_1M.csv")  # Ganti jika nama fail asal lain

# Ambil 10,000 sampel rawak
df_subset = df.sample(n=10000, random_state=42)

# Simpan sebagai fail baharu
df_subset.to_csv("data/pretrain_subset_10k.csv", index=False)

print("✅ Fail pretrain_subset_10k.csv berjaya disimpan.")
