# Ajustes-recta

Small Streamlit app to perform ODR fits (errors in X and Y).

## Desplegar en Streamlit Cloud

Pasos rápidos para desplegar esta app en Streamlit Cloud:

1. Asegúrate de que el repositorio está en GitHub y que `requirements.txt` está actualizado.
2. En https://share.streamlit.io selecciona "New app" y conecta con tu cuenta de GitHub.
3. Selecciona este repositorio, branch `main` y en "File in repository" escribe `streamlit_app.py`.
4. Pulsa `Deploy`. Streamlit Cloud instalará las dependencias y publicará la app.

Comandos útiles para pruebas locales:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Nota: desde la barra lateral de la app puedes ajustar la `Fracción relativa para errores cero`.