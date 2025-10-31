# Anomaly Detection in Cognitive Trajectories - Installation and Usage Guide

## Quick Start

1. **Clone or download the project files to your local machine**

2. **Navigate to the project directory:**
   ```cmd
   cd C:\Dev\ml\alzheimer_anomaly_detection
   ```

3. **Install dependencies:**
   ```cmd
   pip install -r requirements.txt
   ```

4. **Run the complete analysis (uses the ADNI dataset configured in `config.yaml`):**
   ```cmd
   python src\main_analysis.py
   ```

5. **Open the Jupyter notebook for interactive analysis:**
   ```cmd
   jupyter notebook notebooks\anomaly_detection_analysis.ipynb
   ```

## Running with Different Data Sources

### ADNI (Bundled) Dataset
The configuration shipped with the project points to `ADNIMERGE_23Jun2024.csv`. Running `python src\main_analysis.py` will automatically load, harmonize, and analyse this dataset.

### Custom Real-World Dataset
Provide a CSV file that contains longitudinal cognitive assessments and run:

```cmd
python src\main_analysis.py --data-file path\to\your_dataset.csv
```

Update `config/config.yaml` (see **Configuration** below) to map your column names to the expected fields.

## Project Structure

```
alzheimer_anomaly_detection/
├── data/                           # Input datasets
├── src/                           # Source code modules
│   ├── data_loader.py            # Real-world dataset ingestion and harmonisation
│   ├── preprocessing.py          # Feature engineering
│   ├── anomaly_detection.py      # ML algorithms
│   ├── visualization.py          # Plotting and charts
│   └── main_analysis.py          # Main pipeline orchestration
├── notebooks/                    # Jupyter analysis notebooks
│   └── anomaly_detection_analysis.ipynb
├── results/                      # Output files and plots
├── config/                       # Configuration files
│   └── config.yaml              # Analysis parameters
├── requirements.txt              # Python dependencies
└── README.md                    # Project documentation
```

## Key Features

### 1. Feature Engineering
- Trajectory-based features (slopes, acceleration, variability)
- PCA dimensionality reduction
- Missing data interpolation
- Standardization and preprocessing

### 2. Anomaly Detection
- **Isolation Forest**: Tree-based anomaly detection
- **DBSCAN**: Density-based clustering for outlier identification
- Consensus anomaly detection combining both methods
- Hyperparameter tuning and optimization

### 3. Visualization
- Trajectory plots with anomaly highlighting
- t-SNE embeddings of feature space
- Cluster analysis and silhouette plots
- Interactive dashboards with Plotly

### 4. Validation
- Temporal cross-validation
- Consistency analysis across time splits
- Feature importance analysis with SHAP
- Clinical characteristic analysis

## Configuration

Edit `config/config.yaml` to customize:

```yaml
# Data configuration
data:
   real_dataset_path: 'ADNIMERGE_23Jun2024.csv'  # Path to your dataset
   min_visits_required: 3                    # Minimum number of visits per patient
   visit_months: [0, 6, 12, 18, 24, 36, 48, 60] # Harmonised visit schedule for features

# Real dataset column mapping
real_data:
   patient_id_col: 'RID'
   visit_code_col: 'VISCODE'
   visit_month_col: 'Month'
   exam_date_col: 'EXAMDATE'
   mmse_col: 'MMSE'
   faq_col: 'FAQ'
   age_col: 'AGE'
   gender_col: 'PTGENDER'
   education_col: 'PTEDUCAT'
   apoe4_col: 'APOE4'
   diagnosis_col: 'DX'

# Isolation Forest settings
isolation_forest:
   contamination: 0.08
   n_estimators: 512

# DBSCAN settings
dbscan:
   eps_range: [0.3, 1.5]
   min_samples_range: [3, 10]
```

## Output Files

After running the analysis, check the `results/` directory for:

- `analysis_results.json` - Complete analysis results
- `patient_anomaly_scores.csv` - Patient-level anomaly scores
- `summary_report.txt` - Text summary of findings
- `trajectory_overview.png` - Trajectory visualization
- `anomaly_embedding.png` - Feature space analysis
- `cluster_analysis.png` - Clustering results
- `interactive_dashboard.html` - Interactive exploration tool

## Interpreting Results

### Anomaly Scores
- **Isolation Forest**: Lower scores indicate higher anomaly likelihood
- **DBSCAN**: Points labeled as "noise" (-1) are potential anomalies
- **Combined Score**: Weighted combination of both methods (higher = more anomalous)

### Clinical Patterns
- **Rapid Decline**: Steep cognitive deterioration (MMSE decline > 0.6 points/month)
- **Plateau**: Stable cognition (minimal change in both MMSE and FAQ)
- **Fluctuating**: High variability in cognitive performance
- **Normal**: Typical gradual decline pattern

### Validation Metrics
- **Agreement Rate**: How often both methods agree on anomaly status
- **Score Correlation**: Consistency of anomaly rankings across time
- **Pattern Detection**: Method performance by true pattern type

## Customization

### Adding New Features
1. Modify `preprocessing.py` to extract additional trajectory features
2. Update `_extract_trajectory_features()` method
3. Adjust PCA variance threshold if needed

### Different Algorithms
1. Add new detection methods to `anomaly_detection.py`
2. Implement fit/predict interface similar to existing methods
3. Update consensus analysis to include new method

### Enhanced Visualization
1. Add plotting functions to `visualization.py`
2. Integrate with main pipeline in `main_analysis.py`
3. Update Jupyter notebook with new visualizations

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```cmd
   pip install -r requirements.txt
   ```

2. **Memory Issues**: Reduce dataset size in `config.yaml`
   - Limit the analysis to fewer visit months in `data.visit_months`
   - Filter the input CSV to a focused cohort before running the pipeline

3. **SHAP Errors**: Install SHAP for feature importance
   ```cmd
   pip install shap
   ```

4. **Plotting Errors**: Check matplotlib backend
   ```python
   import matplotlib
   matplotlib.use('TkAgg')  # or 'Qt5Agg'
   ```

### Performance Optimization

1. **Reduce PCA Components**: Lower variance threshold
2. **Fewer Patients**: Smaller dataset for testing
3. **Parallel Processing**: Increase `n_jobs` parameter
4. **Skip Cross-Validation**: Comment out CV section for faster runs

## Research Applications

### Academic Use
- Study atypical Alzheimer's progression patterns
- Validate on real longitudinal cohort data
- Compare with other anomaly detection methods
- Investigate protective/risk factors in anomalous patients

### Clinical Applications
- Early identification of unusual decline patterns
- Personalized monitoring strategies
- Intervention targeting for different anomaly types
- Quality assurance for cognitive assessment data

## Citation

If you use this code in your research, please cite:

```
Anomaly Detection in Cognitive Trajectories: Identifying Atypical Alzheimer's Progression
Using Isolation Forest and DBSCAN for Unsupervised Outlier Analysis
```

## Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the Jupyter notebook for detailed examples
3. Examine the generated results and summary reports
4. Modify configuration parameters for your specific use case

## License

This project is provided for educational and research purposes. Please ensure compliance with data privacy regulations when working with real patient data.