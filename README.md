# Netflix Genre Analysis using Replicate + IBM Granite (LLaMA-3)

## 🎯 Project Overview
Analisis konten Netflix berdasarkan genre, tahun rilis, dan asal negara. Tujuannya adalah mengungkap pola dominasi genre, tren waktu, dan peluang strategi kurasi konten.

## 📂 Dataset
[Netflix Movies and TV Shows – Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows)

## 🔍 Insight & Findings
- “Drama” adalah genre paling umum secara global
- Lonjakan besar konten terjadi pada tahun 2019
- AS dan India jadi negara dominan, tapi konten non-Inggris meningkat sejak 2017

## 🤖 AI Support
Menggunakan model LLaMA-3-70B di Replicate untuk:
- Menyimpulkan insight berdasarkan hasil analisis EDA
- Memberikan rekomendasi strategi untuk Netflix
- Menjawab prompt tentang klasifikasi genre dan tren rilis

Prompt:
## 1. Install & Setup
!pip install pandas matplotlib seaborn
!pip install replicate

# 2. Import Library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import replicate
import os

# 3. Load Dataset
url = "https://raw.githubusercontent.com/syncthing-project/netflix-dataset/main/netflix_titles.csv"
df = pd.read_csv(url)
df.head()

# 4. Clean Dataset
df.dropna(subset=["type", "title", "country", "release_year", "listed_in"], inplace=True)
df["country"] = df["country"].apply(lambda x: x.split(",")[0])
df["genre"] = df["listed_in"].apply(lambda x: x.split(",")[0])
df_clean = df[["title", "type", "country", "release_year", "genre"]]

# 5. EDA: Genre Populer
genre_count = df_clean["genre"].value_counts().head(10)
plt.figure(figsize=(10,6))
sns.barplot(x=genre_count.values, y=genre_count.index)
plt.title("Top 10 Genre di Netflix")
plt.xlabel("Jumlah")
plt.ylabel("Genre")
plt.show()

# 6. EDA: Konten per Tahun
plt.figure(figsize=(12,6))
df_clean["release_year"].value_counts().sort_index().plot()
plt.title("Tren Rilis Konten Netflix per Tahun")
plt.xlabel("Tahun")
plt.ylabel("Jumlah Konten")
plt.grid(True)
plt.show()

# 7. EDA: Negara Teratas
country_count = df_clean["country"].value_counts().head(10)
plt.figure(figsize=(10,6))
sns.barplot(x=country_count.values, y=country_count.index)
plt.title("Top 10 Negara Produsen Konten Netflix")
plt.xlabel("Jumlah")
plt.ylabel("Negara")
plt.show()

# 8. Generate Insight with IBM Granite via Replicate
# Set your Replicate API token
os.environ["REPLICATE_API_TOKEN"] = "your_replicate_api_token"

from replicate.client import Client
replicate_client = Client(api_token=os.environ["REPLICATE_API_TOKEN"])

prompt = f"""
You are a data analyst. Based on the following findings:
1. Top genres: {', '.join(genre_count.index)}
2. Content spike year: 2019
3. Top countries: {', '.join(country_count.index)}

Generate a business insight summary and 3 recommendations for Netflix.
"""

output = replicate_client.run(
    "meta/meta-llama-3-70b-instruct",
    input={"prompt": prompt, "temperature": 0.7, "top_p": 0.9, "max_tokens": 500}
)

print("\n".join(output))

# 🔚 END
