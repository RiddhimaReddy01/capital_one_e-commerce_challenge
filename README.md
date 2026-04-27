# Capital One E-Commerce Challenge

Data cleaning, metric generation, and Streamlit dashboard for the Capital One e-commerce challenge.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python main.py
```

Run the dashboard:

```bash
streamlit run streamlit_app.py
```

The dashboard opens at `http://localhost:8501`.

## Python Version

The Streamlit deployment uses Python 3.14 through `runtime.txt`.

## Main Files

```text
capital_one_challenge/
|-- main.py              # Runs the full analysis pipeline
|-- streamlit_app.py     # Interactive Streamlit dashboard
|-- requirements.txt     # Python dependencies
|-- runtime.txt          # Streamlit Cloud Python version
|-- README.md
|-- data/
|   |-- raw/             # Original challenge data
|   `-- processed/       # Cleaned data, EDA artifacts, and metric outputs
`-- src/                 # Pipeline, cleaning, validation, metrics, and reporting code
```
