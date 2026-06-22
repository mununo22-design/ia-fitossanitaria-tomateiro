# Inteligência Artificial Aplicada à Vigilância Fitossanitária do Tomateiro

Landing page/dashboard interativo desenvolvido em Streamlit para análise de dados integrados da EPPO Global Database e FAOSTAT, com foco em fitopatógenos regulados associados à cultura do tomateiro.

## Projeto

Este projeto integra dados fitossanitários da EPPO com indicadores produtivos da FAOSTAT para apoiar análises exploratórias, visualização de dados e construção do Índice de Exposição Fitossanitária Potencial do Tomateiro (IEFP-T).

## Fontes principais

- EPPO Global Database
- FAOSTAT
- Notebook final do projeto: `vers4.0_final_eppo_faostat_tomateiro.ipynb`

## Tecnologias

streamlit
pandas
numpy
plotly
matplotlib
seaborn
scikit-learn
xgboost
shap
scipy

## Como executar localmente

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py

cat > .gitignore << 'EOF'
# Ambiente virtual
.venv/
venv/
env/

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.ipynb_checkpoints/

# Streamlit
.streamlit/secrets.toml

# Sistema
.DS_Store
Thumbs.db

# VS Code
.vscode/

# Arquivos temporários
*.tmp
*.log

# Dados brutos grandes, caso existam
data_raw/
outputs_temp/
