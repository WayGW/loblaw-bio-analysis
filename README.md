# Loblaw Bio - Clinical Trial Analysis

Comprehensive bioinformatics pipeline for analyzing immune cell population data from clinical trials with machine learning predictions.

## 📋 Project Overview

This project analyzes immune cell populations (B cells, CD4/CD8 T cells, NK cells, monocytes) from clinical trial samples to:
- Calculate relative frequencies of cell populations
- Compare treatment responders vs non-responders with statistical testing
- **Predict treatment response using machine learning (Random Forest & XGBoost)**
- Identify biomarkers for treatment response
- Explore patient cohorts with interactive tools

**Study Context:** Analysis of miraclib and phauximab treatments for melanoma and carcinoma patients.

## 🚀 Quick Start

### Three Ways to Use This Tool

1. **🖥️ Standalone Application (Easiest - No Installation!)**
   - Download `LoblawBio.exe`
   - Double-click to run
   - No Python, Anaconda, or terminal needed!

2. **🎨 GUI Application (Python Required)**
   - Best for users comfortable with Python but prefer graphical interface
   - Includes interactive dashboard with Streamlit

3. **⌨️ Command Line Interface (Advanced)**
   - Full control and automation capabilities
   - Best for scripting and batch processing

---

## Method 1: Standalone Application (.exe)

### For End Users - No Technical Setup Required!

**Requirements:** Windows 10/11 only

**Steps:**
1. Download `LoblawBio.exe` from the releases folder
2. Extract to any location (e.g., `C:\LoblawBio\`)
3. Double-click `LoblawBio.exe`
4. Use the graphical interface to:
   - Browse and select your data file
   - Run complete analysis with one click
   - View results in built-in viewer
   - Access output files

**Features:**
- ✅ No installation required
- ✅ No Python/Anaconda needed
- ✅ Includes all dependencies
- ✅ Built-in results viewer with tabs:
  - 📋 Overview
  - 📊 Frequency Analysis
  - 🔬 Response Analysis
  - 👥 Baseline Cohort
  - 🤖 ML Analysis (NEW!)
  - 📈 Visualizations

**Note:** The standalone version uses a simplified results viewer. For the full interactive dashboard with filtering, use the Python GUI version.

**File Size:** ~700MB (includes Python runtime and all scientific libraries)

---

## Method 2: GUI Application (Python)

### For Users with Python - Best of Both Worlds!

**Prerequisites:**
- Anaconda or Miniconda
- Git (optional)

**Installation:**
```bash
# 1. Clone or download project
cd C:\Users\YourName\Documents\Projects\loblaw-bio-analysis

# 2. Create conda environment
conda env create -f environment.yml

# 3. Activate environment
conda activate loblaw-bio
```

**Launch GUI:**
```bash
python gui_app.py
```

**GUI Features:**
- 📁 **File Browser** - Select your data file with a click
- ▶️ **One-Click Analysis** - Run complete pipeline with progress bar
- 📊 **Dashboard Launcher** - Opens full Streamlit dashboard in browser
- 📝 **Live Log** - See analysis progress in real-time
- 📂 **Quick Access** - Open output folder directly
- 🤖 **Machine Learning** - Automatic ML model training and evaluation

**Analysis Steps (Automated):**
1. Initialize database
2. Load and validate data
3. Calculate frequency statistics
4. Run statistical tests (Mann-Whitney U)
5. Generate cohort summaries
6. **Train ML models (Random Forest & XGBoost)**
7. Create visualizations

**When Complete:**
- Click "View Interactive Dashboard" for full Streamlit interface
- Click "Open Output Files" to browse results

---

## Method 3: Command Line Interface

### For Advanced Users and Automation

**Installation:**
```bash
# 1. Clone or navigate to project
cd loblaw-bio-analysis

# 2. Create conda environment
conda env create -f environment.yml

# 3. Activate environment
conda activate loblaw-bio
```

**Usage:**
```bash
# Display help
python main.py --help

# Initialize database
python main.py --init-db

# Load data from CSV
python main.py --load-data data/raw/cell-count.csv

# Run all analyses (recommended)
python main.py --run-all

# Run individual analyses
python main.py --frequency-analysis    # Frequency calculations
python main.py --response-analysis     # Statistical tests
python main.py --cohort-analysis       # Cohort filtering
python main.py --ml-analysis           # Machine learning (NEW!)

# View database information
python main.py --info

# Launch interactive dashboard
python main.py --dashboard
```

**Interactive Dashboard:**
```bash
# Launch dashboard
streamlit run src/dashboard/app.py

# Or
python main.py --dashboard
```

Dashboard URL: `http://localhost:8501`

**Dashboard Features:**
- 🏠 **Overview**: Project summary and quick statistics
- 📊 **Frequency Analysis**: Calculate and visualize cell population frequencies with filters
- 📈 **Response Analysis**: Statistical comparison of responders vs non-responders
- 👥 **Cohort Explorer**: Interactive filtering and demographic visualization
- 🤖 **ML Analysis**: Model performance, feature importance, ROC curves (NEW!)
- ℹ️ **Database Info**: Database statistics and distributions

---

## 📁 Project Structure
```
loblaw-bio-analysis/
├── data/
│   ├── raw/                        # Original CSV files
│   │   └── cell-count.csv
│   └── processed/                  # SQLite database
│       └── loblaw_trial.db
│
├── src/
│   ├── database/
│   │   ├── schema.py              # Database schema definitions
│   │   └── loader.py              # ETL pipeline
│   ├── analysis/
│   │   ├── summary_stats.py       # Frequency analysis
│   │   ├── statistical_tests.py   # Response analysis
│   │   ├── filtering.py           # Cohort filtering
│   │   └── ml_analysis.py         # Machine learning (NEW!)
│   ├── visualization/
│   │   └── plots.py               # Plotting functions
│   └── dashboard/
│       └── app.py                 # Streamlit dashboard
│
├── outputs/                        # Generated results
│   ├── frequency_table.csv
│   ├── response_analysis_results.csv
│   ├── response_boxplot.png
│   ├── baseline_cohort.csv
│   ├── ml_model_comparison.csv          # NEW!
│   ├── ml_feature_importance.csv        # NEW!
│   └── ml_analysis_results.png          # NEW!
│
├── config/
│   └── config.yaml                # Configuration settings
├── tests/                          # Unit tests
├── environment.yml                # Conda environment
├── main.py                        # CLI script
├── gui_app.py                     # GUI application
├── simple_viewer.py               # Standalone results viewer
├── LoblawBio.spec                 # PyInstaller spec file
└── README.md
```

---

## 🔬 Analysis Components

### Part 1: Database Management
- **SQLite relational database** with normalized schema
- Three tables: `samples` (metadata), `cell_populations` (reference), `cell_counts` (measurements)
- Foreign key constraints and data integrity checks
- Automated data loading with validation

### Part 2: Frequency Analysis
- Calculate relative frequencies (percentages) for each cell population
- Summary statistics (mean, median, std, min, max)
- Flexible filtering by indication, treatment, sample type
- Export to CSV

**Output:** `outputs/frequency_table.csv`

### Part 3: Statistical Analysis
- **Objective:** Identify cell populations that differ between treatment responders and non-responders
- **Method:** Mann-Whitney U test (non-parametric)
- **Multiple testing correction:** Bonferroni (α = 0.05/5 = 0.01)
- **Effect size:** Rank-biserial correlation
- **Filters:** Melanoma + miraclib + PBMC samples

**Key Finding:** No populations showed statistically significant differences after Bonferroni correction (p < 0.01)

**Outputs:**
- `outputs/response_analysis_results.csv`
- `outputs/response_boxplot.png`

### Part 4: Cohort Exploration
- Filter baseline samples (time=0) by indication, treatment, sample type
- Demographic summaries:
  - Samples per project
  - Responder/non-responder counts
  - Gender distribution
  - Age statistics
- Flexible multi-criteria filtering

**Baseline Cohort (Melanoma + Miraclib + PBMC):**
- 656 samples from 656 subjects
- 331 responders, 325 non-responders
- 344 males, 312 females
- Age: 50-79 years (median 64)

**Output:** `outputs/baseline_cohort.csv`

### Part 5: Machine Learning Analysis (NEW!)

**Objective:** Predict treatment response based on cell population frequencies

**Models:**
- **Random Forest Classifier** (ensemble method, 100 trees)
- **XGBoost Classifier** (gradient boosting)

**Features:** 5 cell population frequencies (b_cell, cd8_t_cell, cd4_t_cell, nk_cell, monocyte)

**Target:** Treatment response (responder vs non-responder)

**Evaluation Metrics:**
- ROC-AUC (primary metric)
- Accuracy, Precision, Recall, F1-Score
- 5-fold cross-validation
- Confusion matrices

**Performance (Melanoma + Miraclib + PBMC):**
- **Random Forest:** ROC-AUC = 0.558, Accuracy = 56.9%
- **XGBoost:** ROC-AUC = 0.552, Accuracy = 56.4%

**Key Insights:**
- Balanced feature importance across all cell populations
- Modest predictive power suggests complex, multi-factorial response mechanism
- All cell types contribute similarly to prediction

**Outputs:**
- `outputs/ml_model_comparison.csv` - Model performance metrics
- `outputs/ml_feature_importance.csv` - Feature rankings
- `outputs/ml_analysis_results.png` - ROC curves, confusion matrices, feature importance plots

---

## 📈 Results Summary

### Study Population
- **Total samples:** 10,500
- **Unique subjects:** 3,500
- **Projects:** 3 (prj1, prj2, prj3)
- **Indications:** Melanoma (5,175), Carcinoma (3,903), Healthy (1,422)
- **Treatments:** Miraclib (4,695), Phauximab (4,383), None (1,422)

### Cell Population Frequencies (Mean %)
| Population | Mean | Median | Std |
|------------|------|--------|-----|
| CD4 T cells | 30.3% | 30.2% | 4.8% |
| CD8 T cells | 24.9% | 24.7% | 4.5% |
| Monocytes | 20.0% | 19.8% | 4.2% |
| NK cells | 14.9% | 14.7% | 3.7% |
| B cells | 9.9% | 9.6% | 3.1% |

### Response Analysis
No cell populations showed statistically significant differences between responders and non-responders after Bonferroni correction (p < 0.01).

**P-values:**
- CD4 T cells: 0.0133 (trend, not significant)
- B cells: 0.0557
- NK cells: 0.1211
- Monocytes: 0.1632
- CD8 T cells: 0.6391

### Machine Learning Results
ML models achieved ~56% accuracy, indicating:
- Response prediction from cell frequencies alone is challenging
- Complex, multi-factorial mechanisms likely involved
- Additional features may be needed for improved prediction

---

## 🛠️ Configuration

Edit `config/config.yaml` to customize:
- Database path
- Analysis parameters (indication, treatment, sample type)
- Statistical thresholds
- Cell population names

---

## 📦 Dependencies

See `environment.yml` for complete list. Key packages:
- **Data:** pandas, numpy
- **Statistics:** scipy, scikit-learn
- **Machine Learning:** xgboost, scikit-learn
- **Visualization:** matplotlib, seaborn, plotly
- **Database:** sqlite3
- **Dashboard:** streamlit
- **GUI:** tkinter, Pillow

---

## 🧪 Testing
```bash
# Test individual modules
python src/analysis/summary_stats.py
python src/analysis/statistical_tests.py
python src/analysis/filtering.py
python src/analysis/ml_analysis.py

# Run unit tests (if implemented)
pytest tests/ -v
```

---

## 🛠️ Building Standalone Executable

For developers who want to create the `.exe`:
```bash
# Activate environment
conda activate loblaw-bio

# Build with PyInstaller
pyinstaller LoblawBio.spec

# Output will be in dist/LoblawBio/
```

**Requirements:** Windows, PyInstaller 6.0+

---

## 🐛 Troubleshooting

**Issue:** `ModuleNotFoundError: No module named 'src'`
- **Solution:** Make sure you're running scripts from the project root directory

**Issue:** `FileNotFoundError: cell-count.csv`
- **Solution:** Place CSV file in `data/raw/` folder

**Issue:** Database locked error
- **Solution:** Close all connections and dashboard, then retry

**Issue:** Streamlit won't stop with Ctrl+C
- **Solution:** Close terminal window or find process: `taskkill /IM streamlit.exe /F`

**Issue:** GUI won't launch
- **Solution:** Activate conda environment first: `conda activate loblaw-bio`

**Issue:** .exe won't run or crashes
- **Solution:** 
  - Make sure you're on Windows 10/11
  - Extract the entire folder, don't run from ZIP
  - Check Windows Defender didn't quarantine files

**Issue:** ML analysis fails
- **Solution:** Make sure scikit-learn and xgboost are installed:
```bash
  conda install scikit-learn xgboost -y
```

---

## 👥 Authors

Grayson - Teiko Project Team

---

## 📄 License

Internal use only - Loblaw Bio

---

## 📞 Support

For questions or issues:
- Check the Troubleshooting section above
- Review example notebooks in `notebooks/`
- Contact the project team

---

## 🎯 Quick Reference

| Task | Command |
|------|---------|
| **Easiest - No setup** | Double-click `LoblawBio.exe` |
| **GUI with full features** | `python gui_app.py` |
| **Full pipeline** | `python main.py --run-all` |
| **ML only** | `python main.py --ml-analysis` |
| **Dashboard** | `python main.py --dashboard` |
| **View results** | Click "Open Output Files" in GUI |

---

**Version:** 2.0 (with Machine Learning)  
**Last Updated:** December 2025