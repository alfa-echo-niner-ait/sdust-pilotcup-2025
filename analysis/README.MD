# Data Analysis

This folder contains all scripts and notebooks for **exploratory data analysis (EDA)** and preprocessing of the weather and air quality dataset used in this project.

## Dataset Details

The Dataset has been collected from **Air Quality Open Data Platform**

https://aqicn.org/data-platform/covid19/

- **Dataset Overview:**

    | Date       | Country | City    | Specie | count | min  | max  | median | variance  |
    |------------|---------|---------|--------|-------|------|------|--------|-----------|
    | 2015-01-04 | CN      | Beijing | pm25   | 450   | 80.0 | 494.0 | 255.0 | 65212.90 |
    | 2015-01-12 | CN      | Beijing | pm25   | 449   | 91.0 | 326.0 | 181.0 | 8349.02  |
    | 2015-02-17 | CN      | Beijing | pm25   | 336   | 13.0 | 197.0 | 68.0  | 10622.50 |
    | 2015-06-11 | CN      | Beijing | pm25   | 416   | 13.0 | 209.0 | 57.0  | 30655.90 |
    | 2015-06-13 | CN      | Beijing | pm25   | 446   | 13.0 | 140.0 | 42.0  | 5978.25  |

- **Date range:** 2014-12-29 00:00:00 to 2025-04-29 00:00:00
- **Total Cities in dataset (52):**
    >['Beijing' 'Jieyang' 'Kunming' 'Hangzhou' 'Chongqing' 'Qingdao' 'Haikou' 'Ürümqi' 'Qiqihar' 'Guiyang' 'Shenzhen' 'Yunfu' 'Xuchang' 'Yinchuan''Shenyang' 'Lhasa' 'Shanghai' 'Changchun' 'Foshan' 'Nanning' 'Fushun' 'Hefei' 'Chengdu' 'Hohhot' 'Qinhuangdao' 'Shijiazhuang' 'Shantou' 'Zhengzhou' 'Nanjing' 'Xining' 'Xi’an' 'Zhuzhou' 'Wuhan' 'Tianjin' 'Changzhou' 'Nanchang' 'Shiyan' 'Xinxiang' 'Suzhou' 'Harbin' 'Lanzhou' 'Jinan' 'Changsha' 'Hegang' 'Anyang' 'Wuxi' 'Taiyuan' 'Guangzhou' 'Fuzhou' 'Ningbo' 'Xiamen' 'Dongguan']
- **Unique Species:** 
    > ['pm25' 'no2' 'co' 'pm10' 'o3' 'so2']
- **Processed Dataset Overview:**
    | Date       | Country | City    | Specie | count | min  | max  | median | variance  | year | month | season |
    |------------|---------|---------|--------|-------|------|------|--------|-----------|------|-------|--------|
    | 2015-01-04 | CN      | Beijing | pm25   | 450   | 80.0 | 494.0 | 255.0 | 65212.90 | 2015 | 1     | Winter |
    | 2015-01-12 | CN      | Beijing | pm25   | 449   | 91.0 | 326.0 | 181.0 | 8349.02  | 2015 | 1     | Winter |
    | 2015-02-17 | CN      | Beijing | pm25   | 336   | 13.0 | 197.0 | 68.0  | 10622.50 | 2015 | 2     | Winter |
    | 2015-06-11 | CN      | Beijing | pm25   | 416   | 13.0 | 209.0 | 57.0  | 30655.90 | 2015 | 6     | Summer |
    | 2015-06-13 | CN      | Beijing | pm25   | 446   | 13.0 | 140.0 | 42.0  | 5978.25  | 2015 | 6     | Summer |

## Contents

- **Data Cleaning:** Handling missing values, outlier detection, and data formatting.
- **Exploratory Analysis:** Statistical summaries, pollutant distribution plots, and time series visualizations.
- **Feature Engineering:** Correlation analysis and creation of new features for model input.
- **Visualization:** Interactive and static plots for understanding trends and anomalies.


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

3. **Run Notebooks:**
   - Open and run the Jupyter notebooks (`*.ipynb`) for step-by-step analysis.

4. **Input Data:**
   - Place raw data files in the root subfolder.
   - Update file paths in scripts as needed.


## Key Files

- `data_analysis.ipynb` — Main EDA notebook for air quality data.
- `china_dataset.csv` — Our original dataset.
- `modified_dataframe.csv` — Processed dataframe saved for using in dash app and model training.
- `china_air_quality_map.html` — Saved map for using in dash app.


## Output

- Cleaned datasets ready for model training (saved in `output/` or as specified).
- Visualizations and summary statistics for reporting and further analysis.

## Notes

- For questions, refer to the main project [README](../README.md) or open an issue.