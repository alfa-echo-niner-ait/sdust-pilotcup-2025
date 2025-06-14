# Model Training

This folder contains all resources related to **LSTM-based air quality forecasting** for the SDUST Pioneer Intelligent Computing 2025 project.


## Contents

- **Data Preparation:** Dataset Loading, scaling, and splitting the dataset for training and testing.
- **Model Architecture:** Implementation of a unified LSTM model using TensorFlow/Keras.
- **Training & Evaluation:** Scripts to train the model, evaluate performance, and visualize results.
- **Model Saving:** Utilities for saving trained models (`.h5`, `.keras`) and scalers (`.pkl`).


## How to Use

1. **Create & activate the virtual environment:**
   Here's a quick guide for creating and activating a Python virtual environment (`.venv`) on both Windows and Linux:

    **Create Virtual Environment**
    ```bash
    python -m venv .venv
    ```
    *(Use `python3` instead of `python` on Linux if needed.)*

    **Activate Virtual Environment**
    **Windows (CMD/PowerShell)**
    ```cmd
    :: CMD
    .\.venv\Scripts\activate

    # PowerShell
    .\.venv\Scripts\Activate.ps1
    ```

    **Linux/macOS (Bash/Zsh)**
    ```bash
    source .venv/bin/activate
    ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
   **Users in china:**
   ```sh
   pip install -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
   ```

## Key Files

- `Train_LSTM.ipynb` — Main notebook for training and evaluating, visualizing the LSTM model.


## Output

- Trained LSTM model files (`.h5`, `.keras`)
- Serialized scalers (`.pkl`)

## Model Details

- **Model config:**

```python
# Build model
model = Sequential(
    [
        Bidirectional(
            LSTM(128, return_sequences=True), input_shape=(time_step, x_data.shape[2])
        ),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1),
    ]
)

model.compile(optimizer='adam', loss='mean_squared_error')

# Callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5, min_lr=1e-5),
]

# Train
history = model.fit(
    x_train,
    y_train,
    validation_split=0.1,
    epochs=30,
    batch_size=64,
    callbacks=callbacks,
    verbose=1,
)
```
- **Model Performance Metrics:**

| Metric | Value  |
|--------|--------|
| MSE    | 1099.55 |
| RMSE   | 33.16   |
| MAE    | 25.63   |
| R²     | 0.33    |

- **Trining history & results:**

    ![image](https://github.com/user-attachments/assets/5b018c1e-17ac-430c-80ca-905fb77c820f)

    ![image](https://github.com/user-attachments/assets/ce708e3d-d920-4fae-a2ad-5a0434cb5287)


## Notes

- For best results, ensure data preprocessing matches the steps used during model training.
- For integration with the Dash app, use the saved model and scaler files from this folder.
- For questions, refer to the main project [README](../README.md) or open an issue.
