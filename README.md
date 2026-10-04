cat > README.md <<'EOF'
# tcc-deteccao-anomalias

Protótipo (TCC) para detecção de anomalias em séries temporais de consumo diário de energia elétrica, utilizando métodos estatísticos interpretáveis em Python.

**Acesso ao sistema:** https://tcc-deteccao-anomalias-cdsc.streamlit.app/

## Visão geral
- Dataset base: *Individual Household Electric Power Consumption* (UCI).
- Pré-processamento: agregação de medições em nível de minuto para consumo diário (kWh).
- Detectores (em desenvolvimento): Z-score em janela móvel, IQR e média móvel.
- Objetivo: sistema simples, reproduzível e adequado para demonstração acadêmica.

## Estrutura do projeto
- `src/`: módulos principais (preprocessing, detectores, etc.)
- `data/processed/`: dados processados (ex.: `consumo_diario.csv`)
- `tests/`: testes automatizados com Pytest
- Scripts:
  - `run_zscore.py`: executa Z-score e imprime um resumo

> Observação: `data/raw/` não é versionado no GitHub (dataset bruto é grande).

## Requisitos
- Python 3.x

## Instalação
```bash
pip install -r requirements.txt