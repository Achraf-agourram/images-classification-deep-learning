# Image classification

## Run with Docker

Build and start the Streamlit image classifier:

```sh
docker compose up --build
```

Open [http://localhost:8501](http://localhost:8501). The app loads
`models/cnn.keras` and its class names from `models/class_names.json`. If the
class-names file is missing, it reads the labels from `data/splits/train.csv`.
Train the model first if those files are not available:

```sh
docker compose run --rm app python -m src.training.train
```

The Compose service mounts `data`, `models`, and `reports` from the project
directory so datasets and generated artifacts persist on the host. The Docker
build excludes these potentially large or locally generated directories.

Stop the app with `Ctrl+C`, then remove the container with:

```sh
docker compose down
```

## Run without Docker

Install dependencies and start the app with:

```sh
python -m pip install -r requirements.txt
streamlit run app/app.py
```
