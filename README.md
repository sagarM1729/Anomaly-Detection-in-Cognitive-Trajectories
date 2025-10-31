# Anomaly Detection in Cognitive Trajectories

## Project Overview
This project implements unsupervised anomaly detection methods to identify atypical Alzheimer's disease progression patterns using longitudinal cognitive assessment data.

## What We Did & How It Works

### The Problem
Alzheimer's disease affects patients differently - some decline rapidly, others slowly, and some show unusual patterns. We needed to automatically identify patients with **atypical cognitive trajectories** to help researchers understand non-standard disease progression.

### Our Solution
We built a machine learning pipeline that analyzes **1,943 patients** from the ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset with **11,017 longitudinal visits** tracking cognitive decline over time.

### How It Works (Step-by-Step)

1. **Data Loading** (`data_loader.py`)
   - Load ADNI dataset with MMSE scores, FAQ assessments, and diagnosis codes
   - Clean and standardize visit months (baseline, 6m, 12m, 18m, 24m, etc.)
   - Filter patients with at least 3 visits → **742 valid patients**

2. **Feature Engineering** (`preprocessing.py`)
   - Extract 6 types of trajectory features from each patient:
     - Cognitive decline **slopes** (how fast they decline)
     - **Acceleration** (is decline speeding up or slowing down?)
     - **Variability** (consistent or erratic changes?)
     - **Change from baseline** (how much total decline?)
     - **Ratios** (relative changes between assessments)
     - **Timing** (when did changes occur?)
   - Total: **46 features** → Reduce to **11 principal components** using PCA

3. **Anomaly Detection** (`anomaly_detection.py`)
   - **Isolation Forest**: Identifies patients whose trajectories are statistically unusual
   - **DBSCAN Clustering**: Groups similar patients, flags outliers as "noise"
   - Both methods work together to find consensus anomalies

4. **Cross-Validation** (`anomaly_detection.py`)
   - Split data at different time points (12, 18, 24 months)
   - Check if anomalies detected early are still anomalous later
   - Ensures our results are **temporally consistent**

5. **Visualization** (`visualization.py`)
   - Generate trajectory plots, t-SNE embeddings, cluster analysis
   - Create interactive dashboard to explore anomalies

### What the Results Show

✅ **60 anomalous patients detected** out of 742 (8.1%)
- These patients show **atypical cognitive decline patterns**
- Some decline faster than expected, others stabilize unexpectedly

✅ **59.8% method agreement**
- Isolation Forest and DBSCAN agree on 35 patients
- High confidence these are truly atypical cases

✅ **0.900 cross-validation correlation**
- Anomaly scores are **highly consistent over time**
- Early detection (12 months) matches later detection (24 months)

✅ **48.2% noise rate in DBSCAN**
- About half of patients form clear trajectory clusters
- Other half show unique patterns (potential research targets)

### Clinical Significance
- **For Researchers**: Focus on these 60 atypical patients to study non-standard Alzheimer's progression
- **For Clinicians**: Early identification of unusual trajectories may indicate need for personalized treatment
- **For Drug Trials**: Stratify patients by progression patterns for better trial design

## Team Contributions

This project was developed collaboratively by 8 students, with each member contributing to specific components:

### Student 1: Project Architecture & Data Pipeline
- **Responsibilities**: 
  - Overall project structure and architecture design
  - Real dataset loader (`src/data_loader.py`) for ADNI data harmonization
  - Column mapping configuration and data validation
  - CSV parsing, visit month alignment, and patient filtering logic
- **Key Deliverables**: Complete data ingestion pipeline with flexible column mapping
- **Challenges & Solutions**:
  - **Problem**: ADNI dataset had inconsistent visit codes (bl, sc, m06, y1, etc.)
  - **Fix**: Created `_viscode_to_month()` with regex parsing to standardize all visit codes to numeric months

### Student 2: Feature Engineering & Preprocessing
- **Responsibilities**:
  - Trajectory feature extraction (`src/preprocessing.py`)
  - Implementation of slope, acceleration, and variability features
  - Missing data interpolation and standardization
  - PCA dimensionality reduction (46 → 11 components)
- **Key Deliverables**: Feature engineering module with 6 types of trajectory features
- **Challenges & Solutions**:
  - **Problem**: Feature dimension mismatch during cross-validation (trying to use 8 months when only 3 available)
  - **Fix**: Added `visit_months` parameter to dynamically adjust feature extraction based on available data

### Student 3: Isolation Forest Implementation
- **Responsibilities**:
  - Isolation Forest anomaly detection algorithm
  - Hyperparameter tuning (contamination, n_estimators)
  - Anomaly scoring and ranking system
  - Integration with main analysis pipeline
- **Key Deliverables**: `src/anomaly_detection.py` (Isolation Forest components)
- **Challenges & Solutions**:
  - **Problem**: Initial contamination rate detected too few anomalies (only 20 out of 742 patients)
  - **Fix**: Tuned contamination parameter to 0.08 (8%) based on clinical literature for Alzheimer's atypical cases

### Student 4: DBSCAN Implementation
- **Responsibilities**:
  - DBSCAN clustering for density-based anomaly detection
  - Automated parameter tuning (eps, min_samples) with silhouette scoring
  - Noise rate optimization to balance cluster quality
  - K-distance plot analysis for parameter selection
- **Key Deliverables**: DBSCAN module with intelligent parameter optimization
- **Challenges & Solutions**:
  - **Problem**: Initial DBSCAN flagged 97.4% of patients as noise (too strict parameters)
  - **Fix**: Improved tuning with noise rate constraints (5-50%) and combined scoring (silhouette - noise_rate×0.3), reduced noise to 48.2%

### Student 5: Cross-Validation & Model Evaluation
- **Responsibilities**:
  - Temporal cross-validation implementation
  - Dynamic visit month handling for training splits
  - Consistency analysis across time periods
  - Correlation metrics and validation reporting
- **Key Deliverables**: `CrossValidator` class with 0.900 score correlation
- **Challenges & Solutions**:
  - **Problem**: "No valid trajectories found" errors during temporal splits due to insufficient visits per patient
  - **Fix**: Added per-patient visit count validation (≥3 visits before/after cutoff) in `temporal_split()` method

### Student 6: Visualization & Reporting
- **Responsibilities**:
  - Comprehensive visualization suite (`src/visualization.py`)
  - Trajectory plots, t-SNE embeddings, cluster analysis
  - Interactive Plotly dashboards
  - Summary report generation and quality metrics display
- **Key Deliverables**: 4 static plots + interactive HTML dashboard
- **Challenges & Solutions**:
  - **Problem**: Summary report crashed when cross-validation metrics were missing
  - **Fix**: Created `_format_float()` helper function to safely handle None values in report generation

### Student 7: SHAP Analysis & Interpretability
- **Responsibilities**:
  - Feature importance analysis using SHAP values
  - Permutation-based explainability for anomaly scores
  - Integration of interpretability methods
  - Documentation of feature contributions to anomaly detection
- **Key Deliverables**: SHAP integration module and feature importance rankings
- **Challenges & Solutions**:
  - **Problem**: SHAP computation was slow for 742 patients with 11 PCA features
  - **Fix**: Used TreeExplainer optimized for Isolation Forest and limited background samples to 100 patients

### Student 8: Documentation & Integration
- **Responsibilities**:
  - Main analysis pipeline (`src/main_analysis.py`) integration
  - README.md and USAGE_GUIDE.md comprehensive documentation
  - Jupyter notebook for interactive analysis
  - Configuration file setup and CLI argument handling
  - Testing, debugging, and final quality assurance
- **Key Deliverables**: Complete documentation, end-to-end pipeline, and user guides
- **Challenges & Solutions**:
  - **Problem**: Multiple module integration errors and path issues across different components
  - **Fix**: Standardized imports, added error handling, created comprehensive config.yaml for centralized settings

---

## Features
- **Real Dataset Integration**: Directly ingest ADNI (or similar) longitudinal datasets with flexible column mapping
- **Feature Engineering**: Trajectory-based features with interpolation and dimensionality reduction
- **Anomaly Detection**: Isolation Forest and DBSCAN algorithms for outlier identification
- **Visualization**: Comprehensive plots for trajectory analysis and anomaly interpretation
- **Validation**: Cross-validation and consensus anomaly detection

## Project Structure
```
alzheimer_anomaly_detection/
│   ├── data_loader.py       # Real-world dataset ingestion and harmonization
├── data/                    # Input datasets
├── src/                     # Source code modules
├── notebooks/               # Jupyter analysis notebooks
├── results/                 # Output plots and reports
├── config/                  # Configuration files
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

### 1. Clone or download the repository
```bash
cd c:\Dev\ml\alzheimer_anomaly_detection
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- numpy>=1.24.0
- pandas>=2.0.0
- scikit-learn>=1.3.0
- matplotlib>=3.7.0
- seaborn>=0.12.0
- scipy>=1.11.0
- shap>=0.42.0
- plotly>=5.15.0
- pyyaml>=6.0

## Quick Start

### Run the complete analysis pipeline:
```bash
# Windows
python src\main_analysis.py

# Linux/Mac
python src/main_analysis.py
```

### Command-line options:
```bash
# Use a custom dataset
python src\main_analysis.py --data-file path\to\your_data.csv

# Specify a different configuration file
python src\main_analysis.py --config path\to\config.yaml
```

## Usage

### Running the Analysis

1. **Default analysis with bundled ADNI dataset:**
   ```bash
   python src\main_analysis.py
   ```
   
   This will:
   - Load `ADNIMERGE_23Jun2024.csv` from the project root
   - Process 1,943 patients with 11,017 longitudinal visits
   - Extract trajectory features and detect anomalies
   - Generate results in `results/analysis_YYYYMMDD_HHMMSS/`

2. **Use a custom dataset:**
   ```bash
   python src\main_analysis.py --data-file C:\path\to\your_data.csv
   ```
   
   Your CSV should include columns mappable to:
   - Patient ID, visit dates/months, MMSE, FAQ scores
   - Demographics: age, gender, education, APOE4 status
   - Diagnosis labels (optional)
   
   See `config/config.yaml` for column mapping configuration.

3. **Interactive exploration:**
   ```bash
   jupyter notebook notebooks\anomaly_detection_analysis.ipynb
   ```

### Expected Output

After running the analysis, check `results/analysis_YYYYMMDD_HHMMSS/` for:

**Reports:**
- `summary_report.txt` - Executive summary of anomaly detection results
- `analysis_results.json` - Complete analysis metadata and configuration
- `data_quality_report.json` - Data completeness and quality metrics
- `patient_anomaly_scores.csv` - Patient-level anomaly rankings with demographics

**Visualizations:**
- `trajectory_overview.png` - Cognitive trajectories colored by anomaly status
- `anomaly_embedding.png` - t-SNE visualization of feature space
- `cluster_analysis.png` - DBSCAN clustering results
- `interactive_dashboard.html` - Interactive Plotly dashboard (open in browser)

### Example Terminal Output
```
============================================================
COGNITIVE TRAJECTORY ANOMALY DETECTION PIPELINE
============================================================

1. Loading real-world dataset...
   Loaded data from: ADNIMERGE_23Jun2024.csv
   Dataset shape: (11017, 121)
   Unique patients: 1943
   Average visits per patient: 5.67

2. Validating data quality...
   Total patients: 1943
   Data completeness: 70.88%
   MMSE missing: 15.01%

3. Engineering trajectory features...
   Built features for 742 patients
   Feature matrix shape: (742, 46)
   PCA reduced to 11 components (90.4% variance)

4. Detecting anomalies...
   Isolation Forest: 60 anomalies (8.1%)
   DBSCAN: 358 noise points (48.2%)
   Consensus: 60 anomalies
   Agreement rate: 59.8%

5. Cross-validating results...
   Split at month 18: 722 valid patients
   Split at month 24: 729 valid patients
   Score correlation: 0.900
   Common patients: 721

6. Analyzing feature importance...
   SHAP analysis completed for 100 samples

7. Generating visualizations...
   All visualizations generated successfully

ANALYSIS COMPLETE!
Results saved to: results\analysis_20251031_222644
```

## Configuration

Edit `config/config.yaml` to customize the analysis:

```yaml
# Dataset settings
data:
  real_dataset_path: 'ADNIMERGE_23Jun2024.csv'  # Path to your CSV
  min_visits_required: 3                         # Minimum visits per patient
  visit_months: [0, 6, 12, 18, 24, 36, 48, 60]  # Harmonized schedule

# Anomaly detection parameters
isolation_forest:
  contamination: 0.08      # Expected outlier fraction (8%)
  n_estimators: 512        # Number of trees

dbscan:
  eps_range: [0.5, 2.5]    # Distance parameter search range
  min_samples_range: [3, 8] # Minimum cluster size range

# Feature engineering
features:
  interpolation_method: 'linear'  # Handle missing values
  max_missing_rate: 0.3           # Reject patients with >30% missing
  pca_variance_threshold: 0.9     # Keep 90% variance

# Cross-validation
cross_validation:
  test_months: [12, 18, 24]  # Temporal split points
```

## Methods
- **Isolation Forest**: Tree-based anomaly detection with contamination tuning
- **DBSCAN**: Density-based clustering to identify noise points as anomalies
- **Feature Engineering**: Trajectory slopes, acceleration, variability, change from baseline
- **Dimensionality Reduction**: PCA to 11 principal components
- **Validation**: Temporal cross-validation (mean correlation: 0.90) and consensus anomaly scoring

## Key Results

### Anomaly Detection Performance
- **Isolation Forest**: Detects 60 anomalies (8.1% of cohort)
- **DBSCAN**: Identifies 358 noise points (48.2% of cohort)
- **Method Agreement**: 59.8% (both methods converge on high-risk patients)
- **Cross-Validation**: Score correlation 0.900 across temporal splits

### Top Anomalous Patients (Example)
| Rank | Patient ID | Anomaly Score | Age | Gender | APOE4 | Diagnosis |
|------|-----------|---------------|-----|--------|-------|-----------|
| 1 | 4041 | 1.000 | 77.9 | Female | 0 | CN |
| 2 | 4015 | 0.988 | 73.6 | Female | 1 | MCI |
| 3 | 4240 | 0.971 | 70.8 | Male | 2 | MCI |
| 4 | 50 | 0.926 | 77.6 | Male | 0 | MCI |
| 5 | 625 | 0.922 | 75.9 | Male | 0 | MCI |

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install missing packages
   ```bash
   pip install -r requirements.txt
   ```

2. **FileNotFoundError (dataset)**: Ensure `ADNIMERGE_23Jun2024.csv` is in project root or specify path
   ```bash
   python src\main_analysis.py --data-file C:\path\to\data.csv
   ```

3. **Memory issues**: Reduce dataset size or adjust PCA components in config
   ```yaml
   features:
     pca_variance_threshold: 0.8  # Fewer components
   ```

4. **Cross-validation errors**: Ensure dataset has sufficient longitudinal coverage
   - Minimum 3 visits per patient
   - Data spanning multiple time points

## Key Metrics
- Anomaly scores and rankings
- Cluster membership and noise detection
- Feature importance via SHAP analysis (permutation-based)
- Clinical plausibility assessment (demographics, APOE4, diagnosis)
- Temporal consistency validation (0.900 correlation)