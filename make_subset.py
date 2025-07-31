# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 17:41:48 2025

@author: zzulk
"""

import pandas as pd

# Baca fail asal
df = pd.read_csv('data/pretrain_500k.csv')

# Ambil subset 50,000 baris secara rawak (tukar ikut keperluan)
df_subset = df.sample(n=50000, random_state=42)

# Simpan sebagai fail baru
df_subset.to_csv('data/pretrain_50kv1.csv', index=False)

print("Subset pretrain.csv saved as pretrain_subset.csv")
