from pathlib import Path
from html import escape

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# CONFIGURAÇÃO GERAL DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="IA Aplicada à Vigilância Fitossanitária do Tomateiro",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONSTANTES DO PROJETO
# ============================================================

DATA_DIR = Path("data")

AUTHOR_NAME = "Paulo Mununu João Pedro"
AUTHOR_TITLE = "Engenheiro Agrônomo | Mestrando em Agroquímica"
AUTHOR_INSTITUTION = "Instituto Federal Goiano — Campus Rio Verde"
AUTHOR_LATTES = "http://lattes.cnpq.br/0856915480190039"
AUTHOR_LINKEDIN = "https://www.linkedin.com/in/paulopedro2"
AUTHOR_EMAIL = "mununo22@live.com.pt"

FILES = {
    "base_paises": "base_analitica_paises_tomateiro.csv",
    "eppo_pathogens": "eppo_pathogens_tomato_clean.csv",
    "eppo_hosts": "eppo_hosts_tomato_clean.csv",
    "eppo_distribution": "eppo_distribution_tomato_clean.csv",
    "eppo_categorization": "eppo_categorization_tomato_clean.csv",
    "faostat_clean": "faostat_tomato_clean.csv",
    "faostat_summary": "faostat_tomato_summary.csv",
    "faostat_recent": "faostat_tomato_recent_summary.csv",
    "faostat_importance": "faostat_tomato_importance_scores.csv",
    "faostat_country_year": "faostat_tomatoes_production_by_country_year.csv",
    "faostat_world": "faostat_tomatoes_world_production.csv",
    "iefp_index": "iefp_tomato_exposure_index.csv",
    "iefp_classificado": "indice_exposicao_fitossanitaria_tomateiro_classificado.csv",
    "ranking_paises": "ranking_paises_exposicao_fitossanitaria_tomateiro.csv",
    "ranking_patogeno_pais": "ranking_final_iefp_tomateiro_patogeno_pais.csv",
    "manifesto": "manifesto_reprodutibilidade_pipeline.csv",
}


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

@st.cache_data(show_spinner=False)
def carregar_csv(nome_arquivo: str) -> pd.DataFrame:
    """
    Lê arquivos CSV de forma robusta.
    Tenta diferentes codificações e separadores para evitar erro de leitura.
    """
    caminho = DATA_DIR / nome_arquivo

    if not caminho.exists():
        return pd.DataFrame()

    tentativas = [
        {"encoding": "utf-8", "sep": ","},
        {"encoding": "utf-8", "sep": ";"},
        {"encoding": "utf-8-sig", "sep": ","},
        {"encoding": "utf-8-sig", "sep": ";"},
        {"encoding": "latin1", "sep": ","},
        {"encoding": "latin1", "sep": ";"},
    ]

    for params in tentativas:
        try:
            df = pd.read_csv(caminho, **params)
            df.columns = [str(col).strip() for col in df.columns]
            return df
        except Exception:
            continue

    try:
        df = pd.read_csv(caminho, sep=None, engine="python")
        df.columns = [str(col).strip() for col in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Encontra a primeira coluna existente entre possíveis nomes."""
    if df is None or df.empty:
        return None

    cols_lower = {col.lower(): col for col in df.columns}

    for cand in candidates:
        if cand in df.columns:
            return cand

        cand_lower = cand.lower()
        if cand_lower in cols_lower:
            return cols_lower[cand_lower]

    return None


def to_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    """Converte coluna para numérico com segurança."""
    if df is None or df.empty or col not in df.columns:
        return pd.Series(dtype=float)

    return pd.to_numeric(df[col], errors="coerce")


def format_number(value, decimals: int = 0) -> str:
    """Formata números em padrão visual brasileiro."""
    if pd.isna(value):
        return "0"

    try:
        value = float(value)
    except Exception:
        return str(value)

    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.{decimals}f} bi".replace(".", ",")

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.{decimals}f} mi".replace(".", ",")

    if abs(value) >= 1_000:
        return f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")

    return f"{value:.{decimals}f}".replace(".", ",")


def safe_dataframe(df: pd.DataFrame, height: int | None = None):
    """Mostra dataframe com compatibilidade entre versões do Streamlit."""
    if df is None or df.empty:
        st.info("Nenhum dado disponível para exibição nesta seção.")
        return

    try:
        st.dataframe(df, width="stretch", height=height)
    except TypeError:
        st.dataframe(df, use_container_width=True, height=height)


def safe_plotly_chart(fig):
    """Mostra gráfico Plotly com compatibilidade entre versões do Streamlit."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def metric_card(title: str, value: str, subtitle: str = ""):
    """Cria card visual para indicadores principais."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{escape(title)}</div>
            <div class="metric-value">{escape(str(value))}</div>
            <div class="metric-subtitle">{escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def filtrar_por_paises(df: pd.DataFrame, paises: list[str]) -> pd.DataFrame:
    """Filtra um dataframe por países quando houver coluna compatível."""
    if df is None or df.empty or not paises:
        return df

    country_col = find_col(
        df,
        [
            "country",
            "country_faostat",
            "country_eppo",
            "area",
            "area_name",
            "country_name",
        ]
    )

    if country_col is None:
        return df

    return df[df[country_col].astype(str).isin(paises)].copy()


def filtrar_por_classe_exposicao(df: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Filtra dataframe por classe de exposição quando houver coluna compatível."""
    if df is None or df.empty or not classes:
        return df

    class_col = find_col(
        df,
        [
            "country_exposure_class",
            "exposure_class_integrated",
            "exposure_class_integrated_with_zero",
            "exposure_class_log_sensitivity",
        ]
    )

    if class_col is None:
        return df

    return df[df[class_col].astype(str).isin(classes)].copy()


def filtrar_por_classe_biologica(df: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Filtra dataframe por classe biológica quando houver coluna compatível."""
    if df is None or df.empty or not classes:
        return df

    bio_col = find_col(df, ["biological_class", "taxonomic_group", "group"])

    if bio_col is None:
        return df

    return df[df[bio_col].astype(str).isin(classes)].copy()


def top_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    orientation: str = "v",
    top_n: int = 20,
    color_col: str | None = None
):
    """Cria gráfico de barras para rankings."""
    if df is None or df.empty:
        st.info("Base vazia para gerar gráfico.")
        return

    if x_col not in df.columns or y_col not in df.columns:
        st.info(f"Colunas necessárias não encontradas: {x_col}, {y_col}.")
        return

    plot_df = df.copy()
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[y_col])
    plot_df = plot_df.sort_values(y_col, ascending=False).head(top_n)

    if plot_df.empty:
        st.info("Não há valores numéricos disponíveis para gerar o gráfico.")
        return

    if orientation == "h":
        plot_df = plot_df.sort_values(y_col, ascending=True)
        fig = px.bar(
            plot_df,
            x=y_col,
            y=x_col,
            orientation="h",
            color=color_col if color_col in plot_df.columns else None,
            title=title,
            text=y_col,
        )
    else:
        fig = px.bar(
            plot_df,
            x=x_col,
            y=y_col,
            color=color_col if color_col in plot_df.columns else None,
            title=title,
            text=y_col,
        )

    fig.update_layout(
        title_x=0.02,
        margin=dict(l=20, r=20, t=70, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=13),
    )

    safe_plotly_chart(fig)


def class_distribution_chart(df: pd.DataFrame, col: str, title: str):
    """Cria gráfico de distribuição de classes."""
    if df is None or df.empty or col not in df.columns:
        st.info(f"Coluna não encontrada para distribuição: {col}.")
        return

    dist = (
        df[col]
        .fillna("sem_dados")
        .astype(str)
        .value_counts()
        .reset_index()
    )

    dist.columns = [col, "n"]

    fig = px.bar(
        dist,
        x=col,
        y="n",
        text="n",
        title=title,
    )

    fig.update_layout(
        title_x=0.02,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Classe",
        yaxis_title="Número de registros",
    )

    safe_plotly_chart(fig)


# ============================================================
# CARREGAMENTO DAS BASES
# ============================================================

dados = {chave: carregar_csv(nome) for chave, nome in FILES.items()}

base_paises = dados["base_paises"]
eppo_pathogens = dados["eppo_pathogens"]
eppo_hosts = dados["eppo_hosts"]
eppo_distribution = dados["eppo_distribution"]
eppo_categorization = dados["eppo_categorization"]
faostat_clean = dados["faostat_clean"]
faostat_summary = dados["faostat_summary"]
faostat_recent = dados["faostat_recent"]
faostat_importance = dados["faostat_importance"]
faostat_country_year = dados["faostat_country_year"]
faostat_world = dados["faostat_world"]
iefp_index = dados["iefp_index"]
iefp_classificado = dados["iefp_classificado"]
ranking_paises = dados["ranking_paises"]
ranking_patogeno_pais = dados["ranking_patogeno_pais"]
manifesto = dados["manifesto"]


# ============================================================
# CSS — IDENTIDADE VISUAL AZUL E BRANCO
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --azul-principal: #0B3D91;
        --azul-secundario: #1F4E79;
        --azul-claro: #EAF3FF;
        --azul-muito-claro: #F7FBFF;
        --texto: #102A43;
        --cinza: #52606D;
        --branco: #FFFFFF;
        --borda: #D9E8FF;
    }

    .stApp {
        background: linear-gradient(180deg, #F7FBFF 0%, #FFFFFF 42%, #F7FBFF 100%);
        color: var(--texto);
    }

    section[data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #D9E8FF;
    }

    .hero {
        background: linear-gradient(135deg, #0B3D91 0%, #1F4E79 50%, #2D9CDB 100%);
        padding: 42px 36px;
        border-radius: 24px;
        color: white;
        margin-bottom: 28px;
        box-shadow: 0 16px 40px rgba(11, 61, 145, 0.20);
    }

    .hero-title {
        font-size: 42px;
        font-weight: 900;
        line-height: 1.12;
        margin-bottom: 14px;
    }

    .hero-subtitle {
        font-size: 20px;
        line-height: 1.55;
        max-width: 1100px;
        opacity: 0.96;
    }

    .hero-tag {
        display: inline-block;
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.30);
        color: white;
        padding: 8px 14px;
        border-radius: 999px;
        font-size: 14px;
        margin-bottom: 18px;
        font-weight: 600;
    }

    .section-card {
        background: #FFFFFF;
        border: 1px solid #D9E8FF;
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 28px rgba(16, 42, 67, 0.06);
    }

    .method-card {
        background: #F7FBFF;
        border-left: 6px solid #0B3D91;
        border-radius: 16px;
        padding: 22px 24px;
        margin: 16px 0;
        color: #102A43;
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #D9E8FF;
        border-radius: 18px;
        padding: 22px 20px;
        min-height: 142px;
        box-shadow: 0 10px 26px rgba(16, 42, 67, 0.07);
        margin-bottom: 16px;
    }

    .metric-title {
        color: #52606D;
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .04em;
        margin-bottom: 10px;
    }

    .metric-value {
        color: #0B3D91;
        font-size: 30px;
        font-weight: 900;
        line-height: 1.1;
        margin-bottom: 8px;
    }

    .metric-subtitle {
        color: #52606D;
        font-size: 13px;
        line-height: 1.35;
    }

    .small-note {
        color: #52606D;
        font-size: 14px;
        line-height: 1.55;
    }

    .footer {
        background: linear-gradient(135deg, #0B3D91 0%, #1F4E79 100%);
        color: white;
        padding: 28px 30px;
        border-radius: 22px;
        margin-top: 40px;
        box-shadow: 0 16px 40px rgba(11, 61, 145, 0.18);
    }

    .footer a {
        color: #FFFFFF !important;
        font-weight: 700;
        text-decoration: underline;
    }

    .footer-title {
        font-size: 22px;
        font-weight: 900;
        margin-bottom: 8px;
    }

    .footer-text {
        font-size: 15px;
        line-height: 1.65;
        opacity: 0.96;
    }

    h1, h2, h3 {
        color: #0B3D91;
    }

    div[data-testid="stMetricValue"] {
        color: #0B3D91;
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR — FILTROS E CONTROLE
# ============================================================

st.sidebar.markdown("## 🍅 Painel de controle")
st.sidebar.markdown(
    "Filtros interativos aplicados às principais tabelas e gráficos do dashboard."
)

country_col_base = find_col(base_paises, ["country", "country_faostat", "country_eppo"])
class_col_base = find_col(base_paises, ["country_exposure_class"])
bio_col_exposure = find_col(iefp_index, ["biological_class"])
bio_col_pathogens = find_col(eppo_pathogens, ["biological_class", "taxonomic_group", "group"])

available_countries = []
if country_col_base:
    available_countries = (
        base_paises[country_col_base]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

available_classes = []
if class_col_base:
    available_classes = (
        base_paises[class_col_base]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

available_bio_classes = []
if bio_col_exposure:
    available_bio_classes = (
        iefp_index[bio_col_exposure]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )
elif bio_col_pathogens:
    available_bio_classes = (
        eppo_pathogens[bio_col_pathogens]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

selected_countries = st.sidebar.multiselect(
    "Filtrar por país",
    options=available_countries,
    default=[],
    placeholder="Todos os países"
)

selected_classes = st.sidebar.multiselect(
    "Filtrar por classe de exposição",
    options=available_classes,
    default=[],
    placeholder="Todas as classes"
)

selected_bio_classes = st.sidebar.multiselect(
    "Filtrar por classe biológica",
    options=available_bio_classes,
    default=[],
    placeholder="Todas as classes biológicas"
)

top_n = st.sidebar.slider(
    "Número de itens nos rankings",
    min_value=5,
    max_value=40,
    value=20,
    step=5
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Arquivos esperados")
st.sidebar.caption(f"Pasta de dados local: `{DATA_DIR}`")


# ============================================================
# BASES FILTRADAS
# ============================================================

base_paises_f = filtrar_por_paises(base_paises, selected_countries)
base_paises_f = filtrar_por_classe_exposicao(base_paises_f, selected_classes)

ranking_paises_f = filtrar_por_paises(ranking_paises, selected_countries)
ranking_paises_f = filtrar_por_classe_exposicao(ranking_paises_f, selected_classes)

iefp_index_f = filtrar_por_paises(iefp_index, selected_countries)
iefp_index_f = filtrar_por_classe_exposicao(iefp_index_f, selected_classes)
iefp_index_f = filtrar_por_classe_biologica(iefp_index_f, selected_bio_classes)

ranking_patogeno_pais_f = filtrar_por_paises(ranking_patogeno_pais, selected_countries)
ranking_patogeno_pais_f = filtrar_por_classe_exposicao(ranking_patogeno_pais_f, selected_classes)
ranking_patogeno_pais_f = filtrar_por_classe_biologica(ranking_patogeno_pais_f, selected_bio_classes)

eppo_pathogens_f = filtrar_por_classe_biologica(eppo_pathogens, selected_bio_classes)
faostat_country_year_f = filtrar_por_paises(faostat_country_year, selected_countries)


# ============================================================
# HEADER / HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-tag">EPPO Global Database + FAOSTAT | Ciência de Dados aplicada à Agricultura de Precisão</div>
        <div class="hero-title">Inteligência Artificial Aplicada à Vigilância Fitossanitária do Tomateiro</div>
        <div class="hero-subtitle">
            Landing page e dashboard interativo para integrar dados fitossanitários e produtivos,
            com foco em fitopatógenos regulados associados à cultura do tomateiro e construção do
            Índice de Exposição Fitossanitária Potencial do Tomateiro — IEFP-T.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INDICADORES PRINCIPAIS
# ============================================================

country_col = find_col(base_paises_f, ["country", "country_faostat", "country_eppo"])
iso_col = find_col(base_paises_f, ["iso3"])

n_countries = 0
if country_col:
    n_countries = base_paises_f[country_col].dropna().nunique()
elif iso_col:
    n_countries = base_paises_f[iso_col].dropna().nunique()
else:
    n_countries = base_paises_f.shape[0]

pathogen_code_col = find_col(eppo_pathogens_f, ["eppo_code", "code", "organism_code"])
n_pathogens = (
    eppo_pathogens_f[pathogen_code_col].dropna().nunique()
    if pathogen_code_col
    else eppo_pathogens_f.shape[0]
)

dist_code_col = find_col(eppo_distribution, ["eppo_code", "code", "organism_code"])
n_distribution = eppo_distribution.shape[0]

reg_col = find_col(
    eppo_categorization,
    [
        "regulated_status_summary",
        "regulated_status",
        "category",
        "categorization",
        "quarantine_status",
    ]
)

if reg_col:
    n_regulated = eppo_categorization[reg_col].dropna().shape[0]
else:
    n_regulated = eppo_categorization.shape[0]

iefp_country_col = find_col(base_paises_f, ["mean_iefp_t_integrated", "iefp_t_integrated"])
max_iefp = to_numeric(base_paises_f, iefp_country_col).max() if iefp_country_col else np.nan

high_class_count = 0
if class_col_base and class_col_base in base_paises_f.columns:
    high_class_count = (
        base_paises_f[class_col_base]
        .astype(str)
        .str.lower()
        .isin(["alta", "alto", "high"])
        .sum()
    )

production_col = find_col(base_paises_f, ["production_tonnes", "production", "value"])
total_production = to_numeric(base_paises_f, production_col).sum() if production_col else np.nan

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    metric_card(
        "Países analisados",
        format_number(n_countries),
        "Países presentes na base analítica consolidada."
    )

with m2:
    metric_card(
        "Fitopatógenos EPPO",
        format_number(n_pathogens),
        "Organismos associados ao tomateiro após filtragem."
    )

with m3:
    metric_card(
        "Registros de distribuição",
        format_number(n_distribution),
        "Ocorrências geográficas extraídas da EPPO."
    )

with m4:
    metric_card(
        "Países em alta exposição",
        format_number(high_class_count),
        "Classe alta segundo o IEFP-T consolidado por país."
    )

with m5:
    metric_card(
        "Produção considerada",
        format_number(total_production, 1) + " t",
        "Soma da produção recente de tomate na base FAOSTAT."
    )


# ============================================================
# ABAS PRINCIPAIS
# ============================================================

tabs = st.tabs(
    [
        "🏠 Visão geral",
        "🧬 EPPO",
        "🌍 FAOSTAT",
        "📊 IEFP-T",
        "🧠 Metodologia",
        "📁 Dados"
    ]
)


# ============================================================
# ABA 1 — VISÃO GERAL
# ============================================================

with tabs[0]:
    st.markdown("## Visão geral do projeto")

    st.markdown(
        """
        <div class="method-card">
        Este painel foi construído para apresentar, de forma visual e interativa,
        os resultados finais do pipeline EPPO + FAOSTAT aplicado à vigilância fitossanitária
        do tomateiro. A base combina informações sobre organismos, hospedeiros,
        distribuição geográfica, status regulatório e relevância produtiva da cultura.
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2 = st.columns([1.15, 1])

    with c1:
        st.markdown("### Ranking de países por exposição fitossanitária potencial")

        x_country = find_col(ranking_paises_f, ["country", "country_faostat", "country_eppo", "iso3"])
        y_iefp = find_col(ranking_paises_f, ["mean_iefp_t_integrated", "iefp_t_integrated"])

        if x_country and y_iefp:
            top_bar_chart(
                ranking_paises_f,
                x_col=x_country,
                y_col=y_iefp,
                title="Países com maior IEFP-T médio integrado",
                orientation="h",
                top_n=top_n,
                color_col=find_col(ranking_paises_f, ["country_exposure_class"])
            )
        else:
            st.info("Ranking de países não disponível ou colunas principais não encontradas.")

    with c2:
        st.markdown("### Distribuição das classes de exposição")

        class_col = find_col(base_paises_f, ["country_exposure_class"])

        if class_col:
            class_distribution_chart(
                base_paises_f,
                class_col,
                "Distribuição das classes de exposição por país"
            )
        else:
            st.info("Coluna de classe de exposição não encontrada na base por país.")

    st.markdown("### Relação entre produção de tomate e número de fitopatógenos")

    scatter_x = find_col(base_paises_f, ["production_tonnes"])
    scatter_y = find_col(base_paises_f, ["n_tomato_pathogens_present", "n_tomato_pathogens"])
    hover_col = find_col(base_paises_f, ["country", "country_faostat", "iso3"])
    color_col = find_col(base_paises_f, ["country_exposure_class"])

    if scatter_x and scatter_y:
        scatter_df = base_paises_f.copy()
        scatter_df[scatter_x] = pd.to_numeric(scatter_df[scatter_x], errors="coerce")
        scatter_df[scatter_y] = pd.to_numeric(scatter_df[scatter_y], errors="coerce")
        scatter_df = scatter_df.dropna(subset=[scatter_x, scatter_y])

        if not scatter_df.empty:
            fig = px.scatter(
                scatter_df,
                x=scatter_x,
                y=scatter_y,
                color=color_col if color_col else None,
                hover_name=hover_col if hover_col else None,
                size=scatter_x,
                title="Produção de tomate × número de fitopatógenos associados/presentes",
                labels={
                    scatter_x: "Produção de tomate (t)",
                    scatter_y: "Número de fitopatógenos"
                },
            )

            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                title_x=0.02,
            )

            safe_plotly_chart(fig)
        else:
            st.info("Não há valores suficientes para o gráfico de dispersão.")
    else:
        st.info("Colunas necessárias para o gráfico de dispersão não foram encontradas.")

    st.markdown("### Mapa exploratório do IEFP-T por país")

    map_iso = find_col(base_paises_f, ["iso3"])
    map_color = find_col(base_paises_f, ["mean_iefp_t_integrated"])
    map_hover = find_col(base_paises_f, ["country", "country_faostat", "country_eppo"])

    if map_iso and map_color:
        map_df = base_paises_f.copy()
        map_df[map_color] = pd.to_numeric(map_df[map_color], errors="coerce")
        map_df = map_df.dropna(subset=[map_iso, map_color])

        if not map_df.empty:
            fig = px.choropleth(
                map_df,
                locations=map_iso,
                color=map_color,
                hover_name=map_hover if map_hover else map_iso,
                color_continuous_scale="Blues",
                title="Distribuição geográfica do IEFP-T médio integrado",
                labels={map_color: "IEFP-T médio"}
            )

            fig.update_layout(
                title_x=0.02,
                margin=dict(l=20, r=20, t=70, b=20),
                paper_bgcolor="white",
            )

            safe_plotly_chart(fig)
        else:
            st.info("Não há dados suficientes para gerar o mapa.")
    else:
        st.info("Mapa não gerado: colunas ISO3 ou IEFP-T não encontradas.")


# ============================================================
# ABA 2 — EPPO
# ============================================================

with tabs[1]:
    st.markdown("## Dados fitossanitários — EPPO Global Database")

    st.markdown(
        """
        <div class="method-card">
        A EPPO Global Database foi utilizada como fonte fitossanitária para recuperar
        informações sobre organismos associados ao tomateiro, hospedeiros,
        distribuição geográfica e categorização regulatória.
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Classes biológicas dos fitopatógenos")

        bio_col = find_col(eppo_pathogens_f, ["biological_class", "taxonomic_group", "group"])

        if bio_col:
            class_distribution_chart(
                eppo_pathogens_f,
                bio_col,
                "Distribuição dos fitopatógenos por classe biológica"
            )
        else:
            st.info("Coluna de classe biológica não encontrada.")

    with c2:
        st.markdown("### Categorias fitossanitárias/regulatórias")

        reg_col_local = find_col(
            eppo_categorization,
            [
                "regulated_status_summary",
                "regulated_status",
                "category",
                "categorization",
            ]
        )

        if reg_col_local:
            class_distribution_chart(
                eppo_categorization,
                reg_col_local,
                "Distribuição das categorias regulatórias EPPO"
            )
        else:
            st.info("Coluna de categoria regulatória não encontrada.")

    st.markdown("### Status de presença/distribuição EPPO")

    presence_col = find_col(
        eppo_distribution,
        [
            "presence_class",
            "distribution_status",
            "status",
            "presence_status",
        ]
    )

    if presence_col:
        class_distribution_chart(
            eppo_distribution,
            presence_col,
            "Distribuição dos status de presença/distribuição"
        )
    else:
        st.info("Coluna de status de presença/distribuição não encontrada.")

    st.markdown("### Tabelas EPPO")

    ep1, ep2, ep3, ep4 = st.tabs(
        [
            "Fitopatógenos",
            "Hospedeiros",
            "Distribuição",
            "Categorização"
        ]
    )

    with ep1:
        safe_dataframe(eppo_pathogens_f.head(500), height=420)

    with ep2:
        safe_dataframe(eppo_hosts.head(500), height=420)

    with ep3:
        safe_dataframe(eppo_distribution.head(500), height=420)

    with ep4:
        safe_dataframe(eppo_categorization.head(500), height=420)


# ============================================================
# ABA 3 — FAOSTAT
# ============================================================

with tabs[2]:
    st.markdown("## Dados produtivos — FAOSTAT")

    st.markdown(
        """
        <div class="method-card">
        A FAOSTAT foi utilizada para incorporar dados produtivos da cultura do tomateiro,
        incluindo produção, área colhida e rendimento por país e ano. Esses indicadores
        foram usados para representar a importância produtiva do tomateiro em cada país.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Evolução mundial da produção de tomate")

    year_col = find_col(faostat_country_year_f, ["year", "ano"])
    value_col = find_col(faostat_country_year_f, ["value", "production_tonnes", "production"])
    element_col = find_col(faostat_country_year_f, ["element_clean", "element", "indicator"])

    if year_col and value_col:
        fao_line = faostat_country_year_f.copy()
        fao_line[year_col] = pd.to_numeric(fao_line[year_col], errors="coerce")
        fao_line[value_col] = pd.to_numeric(fao_line[value_col], errors="coerce")

        if element_col:
            mask_prod = fao_line[element_col].astype(str).str.lower().str.contains("production|produção", regex=True, na=False)
            if mask_prod.any():
                fao_line = fao_line[mask_prod].copy()

        fao_world_series = (
            fao_line
            .dropna(subset=[year_col, value_col])
            .groupby(year_col, as_index=False)[value_col]
            .sum()
            .sort_values(year_col)
        )

        if not fao_world_series.empty:
            fig = px.line(
                fao_world_series,
                x=year_col,
                y=value_col,
                markers=True,
                title="Evolução da produção mundial de tomate na base FAOSTAT",
                labels={year_col: "Ano", value_col: "Produção"}
            )

            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                title_x=0.02,
            )

            safe_plotly_chart(fig)
        else:
            st.info("Não há dados suficientes para gerar série temporal.")
    else:
        st.info("Colunas de ano e valor não encontradas na base FAOSTAT anual.")

    st.markdown("### Países com maior produção recente de tomate")

    prod_country_col = find_col(
        base_paises_f,
        ["country", "country_faostat", "country_eppo", "iso3"]
    )

    prod_value_col = find_col(base_paises_f, ["production_tonnes"])

    if prod_country_col and prod_value_col:
        top_bar_chart(
            base_paises_f,
            x_col=prod_country_col,
            y_col=prod_value_col,
            title="Ranking de países por produção recente de tomate",
            orientation="h",
            top_n=top_n
        )
    else:
        st.info("Colunas de país/produção não encontradas na base consolidada.")

    st.markdown("### Tabelas FAOSTAT")

    f1, f2, f3, f4 = st.tabs(
        [
            "FAOSTAT limpa",
            "Resumo recente",
            "Importância produtiva",
            "Produção país/ano"
        ]
    )

    with f1:
        safe_dataframe(faostat_clean.head(500), height=420)

    with f2:
        safe_dataframe(faostat_recent.head(500), height=420)

    with f3:
        safe_dataframe(faostat_importance.head(500), height=420)

    with f4:
        safe_dataframe(faostat_country_year_f.head(500), height=420)


# ============================================================
# ABA 4 — IEFP-T
# ============================================================

with tabs[3]:
    st.markdown("## Índice de Exposição Fitossanitária Potencial do Tomateiro — IEFP-T")

    st.markdown(
        """
        <div class="method-card">
        O IEFP-T é um índice exploratório construído para combinar presença/distribuição
        de fitopatógenos, peso regulatório, amplitude fitossanitária e importância produtiva
        do tomateiro. O índice não representa incidência, severidade ou risco oficial, mas
        uma medida analítica de exposição potencial.
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Ranking de países por IEFP-T")

        x_country = find_col(ranking_paises_f, ["country", "country_faostat", "country_eppo", "iso3"])
        y_score = find_col(ranking_paises_f, ["mean_iefp_t_integrated", "iefp_t_integrated"])

        if x_country and y_score:
            top_bar_chart(
                ranking_paises_f,
                x_col=x_country,
                y_col=y_score,
                title="Países com maior exposição fitossanitária potencial",
                orientation="h",
                top_n=top_n,
                color_col=find_col(ranking_paises_f, ["country_exposure_class"])
            )
        else:
            st.info("Ranking de países não disponível.")

    with c2:
        st.markdown("### Distribuição das classes do IEFP-T")

        exp_class_col = find_col(
            iefp_index_f,
            [
                "exposure_class_integrated",
                "exposure_class_integrated_with_zero",
                "exposure_class_log_sensitivity",
            ]
        )

        if exp_class_col:
            class_distribution_chart(
                iefp_index_f,
                exp_class_col,
                "Distribuição das classes de exposição organismo-país"
            )
        else:
            st.info("Coluna de classe de exposição não encontrada no índice.")

    st.markdown("### Top combinações país × fitopatógeno por IEFP-T")

    country_r_col = find_col(ranking_patogeno_pais_f, ["country_faostat", "country_eppo", "country", "iso3"])
    pathogen_r_col = find_col(ranking_patogeno_pais_f, ["scientific_name", "organism_name", "name", "eppo_code"])
    score_r_col = find_col(ranking_patogeno_pais_f, ["iefp_t_integrated", "mean_iefp_t_integrated"])

    if country_r_col and pathogen_r_col and score_r_col:
        plot_df = ranking_patogeno_pais_f.copy()
        plot_df[score_r_col] = pd.to_numeric(plot_df[score_r_col], errors="coerce")
        plot_df = plot_df.dropna(subset=[score_r_col]).sort_values(score_r_col, ascending=False).head(top_n)
        plot_df["combinação"] = (
            plot_df[country_r_col].astype(str) + " — " + plot_df[pathogen_r_col].astype(str)
        )
        plot_df = plot_df.sort_values(score_r_col, ascending=True)

        fig = px.bar(
            plot_df,
            x=score_r_col,
            y="combinação",
            orientation="h",
            title="Combinações país × fitopatógeno com maior IEFP-T integrado",
            text=score_r_col,
            color=find_col(plot_df, ["biological_class"]) if find_col(plot_df, ["biological_class"]) else None
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            title_x=0.02,
            yaxis_title="País — Fitopatógeno",
            xaxis_title="IEFP-T integrado"
        )

        safe_plotly_chart(fig)
    else:
        st.info("Colunas necessárias para o ranking país × fitopatógeno não foram encontradas.")

    st.markdown("### Histograma do IEFP-T integrado")

    hist_col = find_col(iefp_index_f, ["iefp_t_integrated"])

    if hist_col:
        hist_df = iefp_index_f.copy()
        hist_df[hist_col] = pd.to_numeric(hist_df[hist_col], errors="coerce")
        hist_df = hist_df.dropna(subset=[hist_col])

        if not hist_df.empty:
            fig = px.histogram(
                hist_df,
                x=hist_col,
                nbins=40,
                title="Distribuição dos valores do IEFP-T integrado",
                labels={hist_col: "IEFP-T integrado"}
            )

            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                title_x=0.02,
                yaxis_title="Frequência"
            )

            safe_plotly_chart(fig)
        else:
            st.info("Não há valores válidos para o histograma.")
    else:
        st.info("Coluna IEFP-T integrado não encontrada.")

    st.markdown("### Tabelas do IEFP-T")

    i1, i2, i3 = st.tabs(
        [
            "Índice organismo-país",
            "Base por país",
            "Ranking final"
        ]
    )

    with i1:
        safe_dataframe(iefp_index_f.head(700), height=460)

    with i2:
        safe_dataframe(base_paises_f.head(700), height=460)

    with i3:
        safe_dataframe(ranking_patogeno_pais_f.head(700), height=460)


# ============================================================
# ABA 5 — METODOLOGIA
# ============================================================

with tabs[4]:
    st.markdown("## Metodologia científica do projeto")

    st.markdown(
        """
        <div class="method-card">
        <h3>1. Objetivo geral</h3>
        <p>
        Construir uma base analítica integrada para apoiar a vigilância fitossanitária
        do tomateiro, relacionando fitopatógenos regulados associados à cultura com
        indicadores produtivos internacionais de produção, área colhida e rendimento.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="method-card">
        <h3>2. Fonte fitossanitária — EPPO Global Database</h3>
        <p>
        A EPPO foi utilizada para obter informações sobre organismos associados ao tomateiro,
        hospedeiros, distribuição geográfica, status de presença e categorização fitossanitária
        ou regulatória. Os arquivos centrais do pipeline são:
        <code>eppo_pathogens_tomato_clean.csv</code>,
        <code>eppo_hosts_tomato_clean.csv</code>,
        <code>eppo_distribution_tomato_clean.csv</code> e
        <code>eppo_categorization_tomato_clean.csv</code>.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="method-card">
        <h3>3. Fonte produtiva — FAOSTAT</h3>
        <p>
        A FAOSTAT foi utilizada para incorporar informações produtivas da cultura do tomate,
        incluindo produção, área colhida e rendimento por país e por ano. Esses atributos
        permitiram estimar a importância produtiva do tomateiro em cada país.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="method-card">
        <h3>4. Integração das bases</h3>
        <p>
        A integração EPPO + FAOSTAT foi realizada por meio da padronização dos países,
        preferencialmente com códigos ISO3. Essa chave permite relacionar registros de
        presença/distribuição de fitopatógenos com indicadores produtivos da cultura do tomateiro.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="method-card">
        <h3>5. Construção do IEFP-T</h3>
        <p>
        O Índice de Exposição Fitossanitária Potencial do Tomateiro foi construído como uma
        medida exploratória integrando quatro dimensões principais:
        </p>
        <ul>
            <li><b>Peso de presença/distribuição</b>: derivado do status de ocorrência na EPPO;</li>
            <li><b>Peso regulatório</b>: derivado da categorização fitossanitária/regulatória;</li>
            <li><b>Pressão fitossanitária do organismo</b>: baseada em amplitude geográfica e/ou de hospedeiros;</li>
            <li><b>Importância produtiva do tomateiro</b>: baseada em produção e área colhida na FAOSTAT.</li>
        </ul>
        <p>
        Fórmula operacional adotada no pipeline:
        </p>
        <p style="font-size:18px; font-weight:800; color:#0B3D91;">
        IEFP-T = presence_weight × regulatory_weight × pathogen_pressure_score × tomato_importance_score
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="method-card">
        <h3>6. Classes analíticas utilizadas</h3>
        <ul>
            <li><b>Classe fitossanitária/regulatória</b>: praga regulamentada, praga quarentenária, RNQP ou organismo de interesse;</li>
            <li><b>Classe biológica</b>: fungo, bactéria, vírus, viroide, nematoide, fitoplasma ou outra classe identificada;</li>
            <li><b>Classe de exposição potencial</b>: baixa, média ou alta, calculada a partir dos quantis do IEFP-T.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="method-card">
        <h3>7. Limitações e interpretação correta</h3>
        <ul>
            <li>Presença registrada na EPPO não representa incidência, severidade ou dano econômico medido em campo;</li>
            <li>Ausência de registro não significa ausência real do fitopatógeno no país;</li>
            <li>Os dados FAOSTAT são agregados por país e não representam variação regional interna;</li>
            <li>O IEFP-T é um índice exploratório para ciência de dados, não um parecer fitossanitário oficial;</li>
            <li>A classificação baixa, média e alta depende da distribuição dos dados disponíveis no pipeline.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Manifesto de reprodutibilidade do pipeline")

    if not manifesto.empty:
        safe_dataframe(manifesto, height=300)
    else:
        st.info("Arquivo de manifesto de reprodutibilidade não encontrado na pasta data/.")


# ============================================================
# ABA 6 — DADOS
# ============================================================

with tabs[5]:
    st.markdown("## Status dos arquivos e bases carregadas")

    status_rows = []

    for chave, nome in FILES.items():
        df = dados[chave]
        status_rows.append(
            {
                "chave": chave,
                "arquivo": nome,
                "carregado": not df.empty,
                "linhas": df.shape[0],
                "colunas": df.shape[1],
            }
        )

    status_df = pd.DataFrame(status_rows)

    safe_dataframe(status_df, height=420)

    st.markdown("### Visualizar uma base específica")

    file_options = list(FILES.keys())

    selected_file_key = st.selectbox(
        "Selecione uma base para visualizar",
        file_options,
        format_func=lambda key: f"{key} — {FILES[key]}"
    )

    selected_df = dados[selected_file_key]

    if not selected_df.empty:
        st.caption(f"Arquivo selecionado: `{FILES[selected_file_key]}`")
        safe_dataframe(selected_df.head(1000), height=520)

        csv_data = selected_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Baixar esta tabela em CSV",
            data=csv_data,
            file_name=FILES[selected_file_key],
            mime="text/csv"
        )
    else:
        st.warning(
            f"O arquivo `{FILES[selected_file_key]}` não foi encontrado ou está vazio. "
            "Confira se ele foi copiado para a pasta data/."
        )


# ============================================================
# RODAPÉ PROFISSIONAL
# ============================================================

st.markdown(
    f"""
    <div class="footer">
        <div class="footer-title">{AUTHOR_NAME}</div>
        <div class="footer-text">
            {AUTHOR_TITLE}<br>
            {AUTHOR_INSTITUTION}<br><br>
            <b>Lattes:</b> <a href="{AUTHOR_LATTES}" target="_blank">{AUTHOR_LATTES}</a><br>
            <b>LinkedIn:</b> <a href="{AUTHOR_LINKEDIN}" target="_blank">www.linkedin.com/in/paulopedro2</a><br>
            <b>E-mail:</b> <a href="mailto:{AUTHOR_EMAIL}">{AUTHOR_EMAIL}</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)