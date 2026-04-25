# RAIE-SCIS Streamlit App

Aplicacion Streamlit para simulacion RAIE-SCIS.

## Ejecutar en local

```bash
pip install -r requirements.txt
streamlit run raie_scis_app.py
```

## Subir a GitHub

1. Crea un repositorio vacio en GitHub (sin README inicial).
2. Desde esta carpeta, ejecuta:

```bash
git init
git add .
git commit -m "Initial Streamlit app"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/TU_REPO.git
git push -u origin main
```

## Publicar en Streamlit Community Cloud

1. Ve a [https://share.streamlit.io/](https://share.streamlit.io/) e inicia sesion con GitHub.
2. Clic en **Create app**.
3. Selecciona tu repositorio y rama `main`.
4. En **Main file path**, coloca:

```text
raie_scis_app.py
```

5. Clic en **Deploy**.

Si luego cambias codigo y haces `git push`, Streamlit redeploya automaticamente.

## Colaboracion (para otra persona)

- En GitHub, entra a **Settings > Collaborators**.
- Invita el usuario GitHub de la otra persona.
- Esa persona podra clonar, editar y hacer pull requests o push (segun permisos).
