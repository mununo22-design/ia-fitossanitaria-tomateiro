# ============================================================

# Inteligência Artificial Aplicada à Vigilância Fitossanitária do Tomateiro

# EPPO + FAOSTAT | Risco Fitossanitário Global | Índice de Exposição Fitossanitária Potencial-Total | Machine Learning

# ============================================================

from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import scipy.stats as stats

# ============================================================

# CONFIGURAÇÃO DA PÁGINA

# ============================================================

st.set_page_config(
page_title="IA Aplicada à Vigilância Fitossanitária do Tomateiro",
page_icon="🍅",
layout="wide"
)

# ============================================================

# SIDEBAR - AMBIENTE

# ============================================================

def verificar_bibliotecas():
libs = ["streamlit","pandas","numpy","plotly","matplotlib","seaborn","sklearn","scipy"]
status = []
for lib in libs:
if importlib.util.find_spec(lib):
status.append(f"OK - {lib}")
else:
status.append(f"FALTANDO - {lib}")
return status

with st.sidebar.expander("✅ Ambiente Python"):
for s in verificar_bibliotecas():
st.write(s)

st.sidebar.title("Filtros")
pais_selecionado = st.sidebar.text_input("Filtrar país")

# ============================================================

# CARREGAMENTO DOS DADOS

# ============================================================

DATA_PATH = Path("data")

@st.cache_data
def carregar_dados():
df = pd.read_csv(DATA_PATH / "base_analitica_paises_tomateiro.csv")
return df

df = carregar_dados()

if pais_selecionado:
df = df[df["pais"].str.contains(pais_selecionado, case=False)]

# ============================================================

# NORMALIZAÇÃO CIENTÍFICA

# ============================================================

scaler = MinMaxScaler()
df[["producao_ton", "num_patogenos"]] = scaler.fit_transform(
df[["producao_ton", "num_patogenos"]]
)

df["iefp_normalizado"] = df["producao_ton"] * df["num_patogenos"]

# ============================================================

# CABEÇALHO

# ============================================================

st.title("🍅 Inteligência Artificial Aplicada à Vigilância Fitossanitária do Tomateiro")

st.markdown("""
Integração das bases **EPPO Global Database** e **FAOSTAT** para avaliação do risco fitossanitário global.
""")

# ============================================================

# METODOLOGIA (PRISMA)

# ============================================================

st.header("📊 Metodologia Científica")

st.markdown("""

* Dados extraídos da EPPO e FAOSTAT (06/05/2026)
* Padronização via ISO3
* Filtragem de presença confirmada
* Cálculo do IEFP-T

**Fórmula:**
IEFP-T = presença × regulação × pressão × importância produtiva
""")

# ============================================================

# MÉTRICAS

# ============================================================

col1, col2, col3 = st.columns(3)

col1.metric("Países", df["pais"].nunique())
col2.metric("Patógenos", int(df["num_patogenos"].sum()))
col3.metric("IEFP médio", round(df["iefp"].mean(), 3))

# ============================================================

# INTERVALO DE CONFIANÇA

# ============================================================

mean_iefp = df["iefp"].mean()
conf = stats.t.interval(
0.95,
len(df["iefp"]) - 1,
loc=mean_iefp,
scale=stats.sem(df["iefp"])
)

st.write(f"Intervalo de Confiança (95%): {conf}")

# ============================================================

# GRÁFICO SCATTER

# ============================================================

fig = px.scatter(
df,
x="producao_ton",
y="num_patogenos",
size="iefp",
color="classe_exposicao",
hover_name="pais"
)

st.plotly_chart(fig, use_container_width=True)

st.info("Maiores produtores apresentam maior risco fitossanitário.")

# ============================================================

# RANKING

# ============================================================

top10 = df.sort_values("iefp", ascending=False).head(10)

fig2 = px.bar(top10, x="pais", y="iefp", color="classe_exposicao")
st.plotly_chart(fig2, use_container_width=True)

# ============================================================

# MAPA GLOBAL

# ============================================================

if "iso3" in df.columns:
fig_map = px.choropleth(
df,
locations="iso3",
color="iefp",
hover_name="pais",
color_continuous_scale="Reds"
)
st.plotly_chart(fig_map, use_container_width=True)

# ============================================================

# HEATMAP

# ============================================================

fig, ax = plt.subplots()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# ============================================================

# MACHINE LEARNING

# ============================================================

st.header("🤖 Modelos de Classificação")

if "classe_exposicao" in df.columns:

```
X = df[["producao_ton", "num_patogenos", "iefp"]]
y = LabelEncoder().fit_transform(df["classe_exposicao"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# SVM
svm = SVC()
svm.fit(X_train, y_train)
pred_svm = svm.predict(X_test)
acc_svm = accuracy_score(y_test, pred_svm)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test, pred_knn)

col1, col2 = st.columns(2)
col1.metric("Acurácia SVM", round(acc_svm, 2))
col2.metric("Acurácia KNN", round(acc_knn, 2))

# MATRIZ DE CONFUSÃO
cm = confusion_matrix(y_test, pred_svm)

fig_cm, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig_cm)
```

# ============================================================

# DISCUSSÃO

# ============================================================

st.header("📚 Discussão Científica")

st.markdown("""
A integração EPPO + FAOSTAT permite análise global do risco fitossanitário.

Países com maior produção tendem a maior exposição.

A ausência de dados indica fragilidade nos sistemas de vigilância.
""")

# ============================================================

# RODAPÉ

# ============================================================

st.markdown("---")

st.caption("""
Paulo Mununu João Pedro
Engenheiro Agrónomo | Mestrando em Agroquímica
Instituto Federal Goiano — Campus Rio Verde
Lattes: http://lattes.cnpq.br/0856915480190039
LinkedIn: [www.linkedin.com/in/paulopedro2](http://www.linkedin.com/in/paulopedro2)
E-mail: [mununo22@live.com.pt](mailto:mununo22@live.com.pt)
""")
