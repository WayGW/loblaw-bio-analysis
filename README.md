# Loblaw Bio - Clinical Trial Analysis

Comprehensive bioinformatics pipeline for analyzing immune cell population data from clinical trials.

## 📋 Project Overview

This project analyzes immune cell populations (B cells, CD4/CD8 T cells, NK cells, monocytes) from clinical trial samples to:
- Calculate relative frequencies of cell populations
- Compare treatment responders vs non-responders
- Identify biomarkers for treatment response
- Explore patient cohorts with interactive tools

**Study Context:** Analysis of miraclib and phauximab treatments for melanoma and carcinoma patients.

## 🚀 Quick Start

### Prerequisites
- Anaconda or Miniconda
- Git

### Installation
```bash
# 1. Clone or navigate to project directory
cd C:\Users\waygw\Documents\Projects\Teiko

# 2. Create conda environment
conda env create -f environment.yml

# 3. Activate environment
conda activate loblaw-bio

# 4. Initialize database
python main.py --init-db

# 5. Load data
python main.py --load-data data/raw/cell-count.csv
```

## 📊 Usage

### Command Line Interface
```bash
# Display help
python main.py --help

# Initialize database
python main.py --init-db

# Load data from CSV
python main.py --load-data data/raw/cell-count.csv

# Run all analyses
python main.py --run-all

# Run individual analyses
python main.py --frequency-analysis    # Part 2: Frequency calculations
python main.py --response-analysis     # Part 3: Statistical tests
python main.py --cohort-analysis       # Part 4: Cohort filtering

# View database information
python main.py --info

# Launch interactive dashboard
python main.py --dashboard
```

### Interactive Dashboard
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
- ℹ️ **Database Info**: Database statistics and distributions

## 📁 Project Structure
```
Teiko/
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
│   │   ├── summary_stats.py       # Part 2: Frequency analysis
│   │   ├── statistical_tests.py   # Part 3: Response analysis
│   │   └── filtering.py           # Part 4: Cohort filtering
│   ├── visualization/
│   │   └── plots.py               # Plotting functions
│   └── dashboard/
│       └── app.py                 # Streamlit dashboard
│
├── tests/                          # Unit tests
├── config/
│   └── config.yaml                # Configuration settings
├── notebooks/                      # Jupyter notebooks
├── outputs/                        # Generated plots and results
├── environment.yml                # Conda environment
├── main.py                        # Main execution script
└── README.md
```

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
- **Filters:** Melanoma + miraclib + PBMC samples only

**Key Finding:** CD4 T cells show a trend toward higher frequencies in responders (p=0.0133), but not significant after Bonferroni correction.

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

### Response Analysis (Melanoma + Miraclib + PBMC)
No cell populations showed statistically significant differences between responders and non-responders after Bonferroni correction (p < 0.01).

**Populations tested:**
- CD4 T cells: p=0.0133 (trend, not significant)
- B cells: p=0.0557
- NK cells: p=0.1211
- Monocytes: p=0.1632
- CD8 T cells: p=0.6391

## 🛠️ Configuration

Edit `config/config.yaml` to customize:
- Database path
- Analysis parameters (indication, treatment, sample type)
- Statistical thresholds
- Visualization settings

## 🧪 Testing
```bash
# Test individual modules
python src/analysis/summary_stats.py
python src/analysis/statistical_tests.py
python src/analysis/filtering.py

# Run unit tests (if implemented)
pytest tests/ -v
```

## 📦 Dependencies

See `environment.yml` for complete list. Key packages:
- **Data:** pandas, numpy
- **Statistics:** scipy, statsmodels, pingouin
- **Visualization:** matplotlib, seaborn, plotly
- **Database:** sqlite3, sqlalchemy
- **Dashboard:** streamlit

## 👥 Authors

Grayson - Teiko Project Team

## 📄 License

Internal use only - Loblaw Bio

## 🐛 Troubleshooting

**Issue:** `ModuleNotFoundError: No module named 'src'`
- **Solution:** Make sure you're running scripts from the project root directory

**Issue:** `FileNotFoundError: cell-count.csv`
- **Solution:** Place CSV file in `data/raw/` folder

**Issue:** Database locked error
- **Solution:** Close all connections and dashboard, then retry

**Issue:** Streamlit won't stop with Ctrl+C
- **Solution:** Close terminal window or use `taskkill /F /IM python.exe`

## 📞 Support

For questions or issues, contact the project team or refer to documentation in `notebooks/`.