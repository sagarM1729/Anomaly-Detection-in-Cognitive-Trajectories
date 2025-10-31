# Anomaly Detection in Cognitive Trajectories

## Project Overview
This project implements unsupervised anomaly detection methods to identify atypical Alzheimer's disease progression patterns using longitudinal cognitive assessment data.

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