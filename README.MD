# Air Quality Analysis, Prediction and Visualization of Major Chinese Cities

This project presents a complete pipeline for air quality forecasting, from data analysis to model training and interactive visualization. Developed for the SDUST Pioneer Intelligent Computing 2025 competition, it is organized into three major parts:

- **📊 Data Analysis**
- **🧠 LSTM Model Training**
- **📈 Dash-based Web App**

## 📁 Repository Structure

```bash
├── analysis/       # Data exploration and preprocessing scripts
├── model/          # LSTM model training and evaluation
├── app/            # Dash web app for PM2.5 prediction and visualization
├── README.md       # Project overview and instructions
```

## 1️⃣ Data Analysis

*Located in the [`analysis/`](analysis/) folder.*

- Performs exploratory data analysis (EDA) on historical air quality data.
- Generates plots and statistics for pollutants such as PM2.5, PM10, CO, SO₂, NO₂, and O₃.
- Handles data cleaning and preparation before model training.

**Highlights:**
- Outlier detection
- Missing value handling
- Feature correlation studies
- Important event analysis

## 2️⃣ LSTM Model Training

*Located in the [`model/`](model/) folder.*

- Implements a unified LSTM architecture using TensorFlow/Keras.
- Trains the model to predict PM2.5 levels using various pollutant features.
- Includes scaler serialization (using `joblib`) and model saving in both `.h5` and `.keras` formats.
- Ready for hyperparameter tuning.

## 3️⃣ Interactive Dash App

*Located in the [`app/`](app/) folder.*

- Built with Plotly Dash for visualization and real-time inference.
- Users can select dates, locations, and view predicted vs. actual PM2.5 levels.
- Visualizes weather data from 2014 to 2025.

**Features:**
- Visualize weather data
- Interactive pollutant time series
- Model prediction interface
- Weather quality map
- User-friendly dashboard

## 🚀 Getting Started

1. **Clone the repository:**
   ```sh
   git clone https://github.com/alfa-echo-niner-ait/sdust-pilotcup-2025.git
   cd sdust-pilotcup-2025
   ```

2. **Create & activate the virtual environment:**
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

3. **Install dependencies:**
   - Each main folder (`analysis/`, `model/`, `app/`) contains its own `requirements.txt`.
   - Install as needed, for example:
     ```sh
     pip install -r analysis/requirements.txt
     pip install -r model/requirements.txt
     pip install -r app/requirements.txt
     ```
   - For users in China:
     ```sh
     pip install -r analysis/requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
     pip install -r model/requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
     pip install -r app/requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
     ```

4. **Run Data Analysis:**
   - Use Jupyter notebooks in [`analysis/`](analysis/).

5. **Train and test the Model:**
   - Run and modify the Jupyther notebook in [`model/`](model/).

6. **Launch the Web App:**
   - Start the Dash app from [`app/`](app/):
     ```sh
     python app/app.py
     ```

## 🧑‍💻 Authors

<table>
  <tr>
    <td><b>Team Name</b></td>
    <td colspan="2">NEXUS</td>
  </tr>
  <tr>
    <td rowspan="4"><b>Team Members</b></td>
    <td>Ayub Ali Emon</td>
    <td>emon@sdust.edu.cn</td>
  </tr>
  <tr>
    <td>Phonesamay Phouthavong</td>
    <td>phonesamay@sdust.edu.cn</td>
  </tr>
  <tr>
    <td>Yu Jinling</td>
    <td>jlyu0201@163.com</td>
  </tr>
  <tr>
    <td>Fu Wenting</td>
    <td>fuwenting04@163.com</td>
  </tr>
  <tr>
    <td><b>Project Supervisor</b></td>
    <td>Jiehan Zhou</td>
    <td>jiehan.zhou@sdust.edu.cn</td>
  </tr>
</table>

🔗 Member of **[CogTwins Lab](https://cogtwins.github.io)**

🏫 **Shandong University of Science and Technology**

## 📄 License

This project is licensed under the MIT License.
