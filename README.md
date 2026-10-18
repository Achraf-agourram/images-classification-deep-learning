# Intel Image Classification with CNN

A computer vision project that uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify landscape images into six categories: **buildings, forest, glacier, mountain, sea, and street**.

The project covers dataset exploration and validation, train/validation/test splitting, image preprocessing, data augmentation, model training, evaluation, and prediction on a new image.

## Table of Contents

- [Project Objectives](#project-objectives)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Requirements](#requirements)
- [Installation](#installation)
- [Run the Pipeline](#run-the-pipeline)
- [Predict a New Image](#predict-a-new-image)
- [Evaluation Outputs](#evaluation-outputs)
- [Docker](#docker)
- [Tests](#tests)

## Project Objectives

- Explore the dataset and its class distribution.
- Inspect images and identify unreadable or problematic files.
- Create reproducible train, validation, and test splits using image paths and labels.
- Preprocess images consistently before passing them to the CNN.
- Apply data augmentation to training images to improve generalization.
- Train and save a CNN model.
- Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
- Review misclassified images and the most common class confusions.
- Predict the class and probability for a new image.

## Dataset

The project uses the Intel Image Classification dataset. It contains six landscape categories:

| Class | Description |
|---|---|
| `buildings` | Buildings and urban structures |
| `forest` | Forests and wooded areas |
| `glacier` | Glaciers and icy landscapes |
| `mountain` | Mountains and rocky landscapes |
| `sea` | Sea and coastal scenes |
| `street` | Streets and road scenes |

Place the dataset under `data/Intel_Image_Classification/`. The labeled images must be organized in class-specific folders so that the data-splitting script can discover the class names. The `seg_pred` directory contains images intended for prediction and does not provide class labels for supervised training.

The split script creates CSV manifests rather than copying images into new directories. Each manifest stores an image path and its label.

### Generated splits

By default, the split is:

- **70%** training data
- **15%** validation data
- **15%** test data

The split uses stratification and a fixed random seed to make the results reproducible, assuming the dataset remains unchanged.

## Project Structure

```text
.
├── app/
│   └── app.py
├── data/
│   ├── Intel_Image_Classification/
│   │   └── seg_pred/
│   └── splits/
│       ├── train.csv
│       ├── validation.csv
│       ├── test.csv
│       └── class_distribution.csv
├── models/
│   └── cnn.keras
├── reports/
│   ├── figures/
│   │   ├── class_distribution.png
│   │   ├── confusion_matrix.png
│   │   ├── misclassified_images.png
│   │   └── training_curves.png
│   └── metrics/
│       ├── class_distribution.csv
│       ├── classification_report.csv
│       ├── common_confusions.csv
│       ├── confusion_matrix.csv
│       ├── image_audit.csv
│       ├── misclassified_images.csv
│       ├── problematic_images.csv
│       ├── test_metrics.json
│       └── training_history.csv
├── src/
│   ├── data/
│   │   ├── explore_dataset.py
│   │   ├── inspect_images.py
│   │   ├── prepare_dataset.py
│   │   ├── split_dataset.py
│   │   └── visualize_augmentation.py
│   ├── evaluation/
│   │   └── evaluate.py
│   ├── models/
│   │   └── cnn.py
│   ├── prediction/
│   │   └── predict.py
│   └── training/
│       └── train.py
├── tests/
│   ├── test_data.py
│   └── test_prediction.py
├── compose.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

Generated files, Python cache directories, and Git metadata may not be included in this overview.

## Workflow

### 1. Explore the dataset

`src/data/explore_dataset.py` explores the dataset, summarizes the class distribution, and creates dataset overview outputs.

```bash
python -m src.data.explore_dataset
```

### 2. Inspect image files

`src/data/inspect_images.py` checks image files and records audit results, including problematic images.

```bash
python -m src.data.inspect_images
```

### 3. Split the dataset

`src/data/split_dataset.py` creates the train, validation, and test CSV manifests under `data/splits/`.

```bash
python -m src.data.split_dataset
```

Run the splitting step again if the source dataset changes. Do not manually mix images from the training, validation, and test splits.

### 4. Prepare images

`src/data/prepare_dataset.py` loads the CSV manifests and creates TensorFlow datasets. Images are resized to **150 × 150 pixels**, converted to three color channels, and normalized to the **[0, 1]** range. Training data is shuffled and augmented; validation and test data are not augmented.

To check dataset preparation and inspect a batch:

```bash
python -m src.data.prepare_dataset
```

To visualize examples of augmentation:

```bash
python -m src.data.visualize_augmentation
```

### 5. Train the CNN

`src/models/cnn.py` defines the CNN architecture. It uses convolutional layers with ReLU activation and max pooling, followed by global average pooling, a dense layer, dropout, and a six-class softmax output.

`src/training/train.py` trains the model using the training split and monitors performance on the validation split. Training uses early stopping and learning-rate reduction when validation loss stops improving.

```bash
python -m src.training.train
```

The trained model is saved to `models/cnn.keras`. Training history and curves are saved under `reports/`.

### 6. Evaluate the model

`src/evaluation/evaluate.py` evaluates the saved model on the held-out test split. It calculates accuracy, precision, recall, and F1-score, and generates a classification report and confusion matrix. It also saves misclassified image paths and a visualization of example mistakes.

```bash
python -m src.evaluation.evaluate
```

Run evaluation after training has completed successfully.

### 7. Predict a new image

`src/prediction/predict.py` loads the saved model, applies the same image preprocessing used during training, predicts the class, and displays the predicted class and its probability.

```bash
python -m src.prediction.predict "path/to/image.jpg"
```

Replace `path/to/image.jpg` with the path to an image you want to classify. On Windows, a quoted absolute path can also be used, for example:

```bash
python -m src.prediction.predict "C:/Users/YourName/Pictures/landscape.jpg"
```

The prediction script expects the model file and the class labels to be available. When the optional `models/class_names.json` file is absent, it can infer class names from the training CSV, as implemented in the prediction module.

## Requirements

Python and the packages listed in `requirements.txt` are required. The main technologies used in the project are:

- **TensorFlow / Keras** — CNN definition, training, and inference
- **Pandas and NumPy** — tabular data and numerical operations
- **Scikit-learn** — dataset splitting and evaluation metrics
- **Matplotlib** — plots and image visualizations
- **Pillow** — image inspection and loading

Use a Python version compatible with the TensorFlow version specified in your environment. If installation fails, check TensorFlow's supported Python versions and the package versions in `requirements.txt`.

## Installation

Clone the repository and open a terminal in the project root. Create and activate a virtual environment, then install the dependencies.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Make sure the dataset is placed in the expected location before running the data scripts.

## Evaluation Outputs

The pipeline saves its outputs under `reports/`:

| File | Purpose |
|---|---|
| `metrics/image_audit.csv` | Results of the image-file audit |
| `metrics/problematic_images.csv` | Images flagged during inspection |
| `metrics/class_distribution.csv` | Number of images per class or split, as produced by the exploration/splitting scripts |
| `metrics/training_history.csv` | Training and validation metrics by epoch |
| `metrics/test_metrics.json` | Overall test metrics, including accuracy and averaged precision, recall, and F1-score |
| `metrics/classification_report.csv` | Per-class precision, recall, F1-score, and support |
| `metrics/confusion_matrix.csv` | Counts of correct predictions and class confusions |
| `metrics/common_confusions.csv` | Most frequent incorrect class predictions |
| `metrics/misclassified_images.csv` | Paths, true labels, predicted labels, and prediction confidence for incorrect predictions |
| `figures/class_distribution.png` | Class-distribution visualization |
| `figures/training_curves.png` | Training and validation accuracy/loss curves |
| `figures/confusion_matrix.png` | Confusion-matrix visualization |
| `figures/misclassified_images.png` | Examples of incorrectly classified images |

### Metric interpretation

- **Accuracy** measures the proportion of all test images classified correctly.
- **Precision** measures how often predictions for a class are correct.
- **Recall** measures how many images belonging to a class are correctly identified.
- **F1-score** balances precision and recall.
- **Confusion matrix** shows which classes are confused with one another.

The reported values depend on the actual training run and should be read from the generated files rather than assumed in advance.

## Docker

The repository includes `Dockerfile` and `compose.yaml` for containerized execution. From the project root, build and start the services with:

```bash
docker compose up --build
```

The services, commands, mounted data directories, and exposed ports depend on the configuration in `compose.yaml`. Stop the services with:

```bash
docker compose down
```

## Notes

- Run commands from the repository root so that relative paths such as `data/splits/` and `models/cnn.keras` resolve correctly.
- The test split should be kept separate from training and validation; use it for final evaluation rather than for tuning the model.
- Data augmentation should be applied only to training images.
- The probability returned by the CNN is the model's predicted softmax probability, not a guarantee that the prediction is correct.
