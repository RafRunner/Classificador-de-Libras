# Classificador de Sinais de Libras — Workshop (Prática)

Este repositório contém uma API de **classificação de sinais de Libras**. Inclui:

-   **API Flask** para inferência por HTTP.
-   **Front-end simples (HTML)** para upload de imagens e visualização da predição.

> O fluxo recomendado é rodar **localmente com `venv`** (ambiente virtual do Python).

## 📁 Estrutura sugerida do projeto

```
.
├── app.py                      # API Flask + front-end
├── static/
│   └── banner.png              # Imagem do banner da página
├── templates/
│   └── index.html              # Front-end (form de upload)
├── requirements.txt            # Dependências
├── README.md                   # Documentação
└── model_weights.pth           # Pesos do classificador

```

## 🧰 Requisitos

-   Python **3.10+**
-   `pip` atualizado

### Bibliotecas principais

-   `flask`, `torch`, `torchvision`, `pillow`

Arquivo `requirements.txt` sugerido:

```
flask
torch
torchvision
pillow
```

## 🧪 Como rodar localmente com `venv`

**Windows (PowerShell):**

```powershell
python -m venv venv
.
env\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Rode a aplicação:

```bash
python app.py
```

Acesse: [http://localhost:5000](http://localhost:5000)

## 🌐 API HTTP

### `POST /predict` — Classificação

-   Entrada: imagem no campo `file` (`multipart/form-data`).
-   Saída: JSON com classe predita

## 📝 Licença

MIT License
