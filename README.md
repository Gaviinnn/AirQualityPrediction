# AirQualityPrediction

A project for air quality prediction using an LSTM-based deep learning model.

## Overview

This project aims to predict the Air Quality Index (AQI) and related pollutant concentrations over time using historical air quality data from Madrid. By utilizing an LSTM model, we try to capture temporal dependencies and trends in the time series data.

## Features

- Data preprocessing and feature engineering scripts.
- An LSTM-based neural network for time series prediction.
- Training, evaluation, and visualization scripts.
- Model checkpoints and logs for monitoring training progress.
- Example results visualizations (plots) included.

## Directory Structure

├── data/
│   ├── raw/                      # Folder for raw data
│   │   ├── .gitignore            # Placeholder to keep the folder structure in version control
│   │   └── csvs_per_year/        # Subfolder containing yearly raw data files (ignored in .gitignore)
│   │       ├── madrid_2001.csv
│   │       ├── madrid_2002.csv
│   │       └── ...
│   │   └── stations.csv          # Station information for raw data (ignored in .gitignore)
│   ├── processed/                # Folder for processed data
│       ├── .gitignore            # Placeholder to keep the folder structure in version control
│       ├── cleaned_station_data.csv
│       ├── feature_engineered_data.csv
│       └── preprocessed_station_data.csv
├── outputs/                      # Folder for output files and results
│   ├── model_checkpoint/         # Checkpoints for saved models
│   │   └── best_model.pth
│   ├── plots/                    # Visualizations and plots
│   │   ├── aqi_prediction.png
│   │   └── ...
│   └── predictions/              # Model prediction outputs
│       ├── predictions.csv
│       └── pollutant_statistics.csv
├── src/                          # Source code for the project
│   ├── data/                     # Data preprocessing scripts
│   │   ├── data_preprocessing.py
│   │   └── transform.py
│   ├── models/                   # Model definition files
│   │   ├── __init__.py
│   │   └── model.py
│   ├── train/                    # Training and evaluation scripts
│   │   ├── evaluate.py
│   │   ├── train.py
│   │   ├── utils.py
│   │   ├── main.py
│   │   ├── target_scaler.pkl
│   │   └── test.py
├── .gitignore                    # Git ignore file to exclude unnecessary files
├── config.yaml                   # Configuration file for the project
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation


## Dataset

- The dataset used in this project is from [Kaggle - Air Quality Madrid](https://www.kaggle.com/datasets/decide-soluciones/air-quality-madrid).
- **Note:** The raw data is not included in this repository due to size and licensing. Please download the dataset from the above link and place it in `data/raw/` before running the scripts.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Gaviinnn/AirQualityPrediction.git
   cd AirQualityPrediction
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

## Usage

1. **Preprocessing the data**:  
   Run the following command to preprocess the data:  
   `python src/data/data_preprocessing.py`

2. **Training the model**:  
   Ensure your dataset is properly placed and modify `config.yaml` as needed:  
   `python src/main.py`  
   This script will:  
   - Split the dataset into train/test sets.  
   - Convert data to PyTorch tensors.  
   - Train the LSTM model.  
   - Save the best model checkpoint to `outputs/model_checkpoint/best_model.pth`.

3. **Evaluation**:  
   After training, evaluate the model and generate predictions with:  
   `python src/train/evaluate.py`  
   This will produce a prediction plot in the `outputs/plots/` directory.

## Configuration

All hyperparameters and training configurations (e.g., sequence length, batch size, learning rate, etc.) can be adjusted in the `config.yaml` file.

## Results

- Model predictions vs. actual values are plotted in `outputs/plots/aqi_prediction.png`.  
- Additional pollutant trends and metrics are plotted and logged during training.

---

## Notes

- **Paths**: This repository uses relative paths for better portability. Ensure your working directory is set to the repository root.  
- If you reference external resources, ensure proper attribution in your project documentation.

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). If you haven’t chosen a license yet, you can create one based on your project needs.

---

## Acknowledgements

- Thanks to [Kaggle Datasets](https://www.kaggle.com/datasets/decide-soluciones/air-quality-madrid) for providing the raw data.  
- Any other references or inspirations.






