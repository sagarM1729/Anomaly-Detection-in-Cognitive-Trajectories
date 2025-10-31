"""
Utility functions and helper modules for the anomaly detection project.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


def calculate_trajectory_metrics(df: pd.DataFrame, patient_id: int, 
                                visit_months: List[int]) -> Dict:
    """
    Calculate comprehensive trajectory metrics for a single patient.
    """
    patient_data = df[df['patient_id'] == patient_id].copy()
    
    if len(patient_data) < 2:
        return {'error': 'Insufficient data points'}
    
    metrics = {}
    
    # Basic metrics
    metrics['n_visits'] = len(patient_data)
    metrics['follow_up_months'] = patient_data['visit_month'].max() - patient_data['visit_month'].min()
    
    # MMSE metrics
    if not patient_data['mmse'].isnull().all():
        mmse_values = patient_data['mmse'].dropna()
        if len(mmse_values) >= 2:
            months_mmse = patient_data.loc[mmse_values.index, 'visit_month']
            
            # Linear slope
            mmse_slope = np.polyfit(months_mmse, mmse_values, 1)[0]
            metrics['mmse_slope'] = mmse_slope
            metrics['mmse_baseline'] = mmse_values.iloc[0]
            metrics['mmse_final'] = mmse_values.iloc[-1]
            metrics['mmse_change'] = mmse_values.iloc[-1] - mmse_values.iloc[0]
            metrics['mmse_range'] = mmse_values.max() - mmse_values.min()
            metrics['mmse_cv'] = mmse_values.std() / mmse_values.mean() if mmse_values.mean() > 0 else 0
    
    # FAQ metrics
    if not patient_data['faq'].isnull().all():
        faq_values = patient_data['faq'].dropna()
        if len(faq_values) >= 2:
            months_faq = patient_data.loc[faq_values.index, 'visit_month']
            
            # Linear slope
            faq_slope = np.polyfit(months_faq, faq_values, 1)[0]
            metrics['faq_slope'] = faq_slope
            metrics['faq_baseline'] = faq_values.iloc[0]
            metrics['faq_final'] = faq_values.iloc[-1]
            metrics['faq_change'] = faq_values.iloc[-1] - faq_values.iloc[0]
            metrics['faq_range'] = faq_values.max() - faq_values.min()
            metrics['faq_cv'] = faq_values.std() / faq_values.mean() if faq_values.mean() > 0 else np.inf
    
    return metrics


def classify_trajectory_pattern(metrics: Dict) -> str:
    """
    Classify trajectory pattern based on metrics.
    """
    if 'error' in metrics:
        return 'insufficient_data'
    
    mmse_slope = metrics.get('mmse_slope', 0)
    faq_slope = metrics.get('faq_slope', 0)
    mmse_cv = metrics.get('mmse_cv', 0)
    
    # Rapid decline: steep MMSE decline and/or steep FAQ increase
    if mmse_slope < -0.6 or faq_slope > 0.4:
        return 'rapid_decline'
    
    # Plateau: minimal change in both measures
    elif abs(mmse_slope) < 0.1 and abs(faq_slope) < 0.1:
        return 'plateau'
    
    # Fluctuating: high coefficient of variation
    elif mmse_cv > 0.15:
        return 'fluctuating'
    
    # Normal decline
    else:
        return 'normal_decline'


def evaluate_anomaly_detection_performance(true_labels: np.ndarray, 
                                         predicted_labels: np.ndarray,
                                         scores: Optional[np.ndarray] = None) -> Dict:
    """
    Evaluate anomaly detection performance.
    """
    from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
    
    # Basic classification metrics
    report = classification_report(true_labels, predicted_labels, output_dict=True)
    
    metrics = {
        'accuracy': report['accuracy'],
        'precision_anomaly': report['True']['precision'] if 'True' in report else 0,
        'recall_anomaly': report['True']['recall'] if 'True' in report else 0,
        'f1_anomaly': report['True']['f1-score'] if 'True' in report else 0,
        'precision_normal': report['False']['precision'] if 'False' in report else 0,
        'recall_normal': report['False']['recall'] if 'False' in report else 0,
        'f1_normal': report['False']['f1-score'] if 'False' in report else 0
    }
    
    # ROC-AUC and PR-AUC if scores are provided
    if scores is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(true_labels, scores)
            metrics['pr_auc'] = average_precision_score(true_labels, scores)
        except:
            metrics['roc_auc'] = np.nan
            metrics['pr_auc'] = np.nan
    
    return metrics


def generate_patient_report(patient_id: int, df: pd.DataFrame, 
                          anomaly_results: Dict, patient_idx: int) -> str:
    """
    Generate a detailed report for a specific patient.
    """
    patient_data = df[df['patient_id'] == patient_id].copy()
    
    if len(patient_data) == 0:
        return f"No data found for patient {patient_id}"
    
    # Get patient demographics
    demo = patient_data.iloc[0]
    
    # Calculate trajectory metrics
    visit_months = [0, 6, 12, 18, 24]
    metrics = calculate_trajectory_metrics(df, patient_id, visit_months)
    
    # Get anomaly results
    iso_score = anomaly_results.get('iso_scores', [0])[patient_idx] if patient_idx < len(anomaly_results.get('iso_scores', [])) else 0
    iso_anomaly = anomaly_results.get('iso_predictions', [1])[patient_idx] == -1 if patient_idx < len(anomaly_results.get('iso_predictions', [])) else False
    db_anomaly = anomaly_results.get('dbscan_predictions', [False])[patient_idx] if patient_idx < len(anomaly_results.get('dbscan_predictions', [])) else False
    
    report = f"""
PATIENT ANOMALY REPORT
Patient ID: {patient_id}
{'='*50}

DEMOGRAPHICS:
  Age: {demo.get('age', 'N/A'):.1f} years
  Gender: {demo.get('gender', 'N/A')}
  Education: {demo.get('education', 'N/A'):.1f} years
  APOE4 Copies: {demo.get('apoe4_copies', 'N/A')}
  True Pattern: {demo.get('pattern_type', 'N/A')}

TRAJECTORY METRICS:
  Number of visits: {metrics.get('n_visits', 'N/A')}
  Follow-up duration: {metrics.get('follow_up_months', 'N/A')} months
  
  MMSE:
    Baseline: {metrics.get('mmse_baseline', 'N/A'):.1f}
    Final: {metrics.get('mmse_final', 'N/A'):.1f}
    Slope: {metrics.get('mmse_slope', 'N/A'):.3f} points/month
    Total change: {metrics.get('mmse_change', 'N/A'):.1f}
    
  FAQ:
    Baseline: {metrics.get('faq_baseline', 'N/A'):.1f}
    Final: {metrics.get('faq_final', 'N/A'):.1f}
    Slope: {metrics.get('faq_slope', 'N/A'):.3f} points/month
    Total change: {metrics.get('faq_change', 'N/A'):.1f}

ANOMALY DETECTION:
  Isolation Forest Score: {iso_score:.3f}
  Isolation Forest Anomaly: {iso_anomaly}
  DBSCAN Noise Point: {db_anomaly}
  
CLINICAL INTERPRETATION:
"""
    
    # Add clinical interpretation
    if iso_anomaly or db_anomaly:
        report += "  This patient shows ATYPICAL cognitive decline patterns.\n"
        
        if metrics.get('mmse_slope', 0) < -0.6:
            report += "  - Rapid MMSE decline detected\n"
        if metrics.get('faq_slope', 0) > 0.4:
            report += "  - Rapid functional decline detected\n"
        if metrics.get('mmse_cv', 0) > 0.15:
            report += "  - High cognitive variability detected\n"
        if abs(metrics.get('mmse_slope', 0)) < 0.1 and abs(metrics.get('faq_slope', 0)) < 0.1:
            report += "  - Cognitive plateau pattern detected\n"
            
        report += "  RECOMMENDATION: Consider closer monitoring and evaluation.\n"
    else:
        report += "  This patient shows TYPICAL cognitive decline patterns.\n"
        report += "  RECOMMENDATION: Continue standard monitoring protocol.\n"
    
    return report


class AnomalyResultsManager:
    """
    Class to manage and analyze anomaly detection results.
    """
    
    def __init__(self, patient_ids: np.ndarray, anomaly_results: Dict, df: pd.DataFrame):
        self.patient_ids = patient_ids
        self.anomaly_results = anomaly_results
        self.df = df
        self._create_results_dataframe()
    
    def _create_results_dataframe(self):
        """Create a comprehensive results dataframe."""
        self.results_df = pd.DataFrame({
            'patient_id': self.patient_ids,
            'iso_score': self.anomaly_results.get('iso_scores', np.zeros(len(self.patient_ids))),
            'iso_anomaly': self.anomaly_results.get('iso_predictions', np.ones(len(self.patient_ids))) == -1,
            'dbscan_noise': self.anomaly_results.get('dbscan_predictions', np.zeros(len(self.patient_ids), dtype=bool)),
            'cluster_label': self.anomaly_results.get('cluster_labels', np.zeros(len(self.patient_ids))),
            'combined_score': self.anomaly_results.get('combined_scores', np.zeros(len(self.patient_ids)))
        })
        
        # Add demographics
        demo_data = self.df.groupby('patient_id').first()[
            ['age', 'gender', 'education', 'apoe4_copies', 'pattern_type']
        ].reset_index()
        
        self.results_df = self.results_df.merge(demo_data, on='patient_id', how='left')
        
        # Add trajectory metrics
        self._add_trajectory_metrics()
    
    def _add_trajectory_metrics(self):
        """Add trajectory metrics to results dataframe."""
        visit_months = [0, 6, 12, 18, 24]
        
        metrics_list = []
        for pid in self.patient_ids:
            metrics = calculate_trajectory_metrics(self.df, pid, visit_months)
            metrics['patient_id'] = pid
            metrics_list.append(metrics)
        
        metrics_df = pd.DataFrame(metrics_list)
        self.results_df = self.results_df.merge(metrics_df, on='patient_id', how='left')
    
    def get_top_anomalies(self, n: int = 10, method: str = 'combined') -> pd.DataFrame:
        """Get top N anomalies by specified method."""
        if method == 'combined':
            return self.results_df.nlargest(n, 'combined_score')
        elif method == 'isolation_forest':
            return self.results_df.nsmallest(n, 'iso_score')  # Lower scores = more anomalous
        else:
            raise ValueError("Method must be 'combined' or 'isolation_forest'")
    
    def get_pattern_analysis(self) -> pd.DataFrame:
        """Analyze anomaly detection by true pattern type."""
        return self.results_df.groupby('pattern_type').agg({
            'iso_anomaly': ['sum', 'count', 'mean'],
            'dbscan_noise': ['sum', 'count', 'mean'],
            'combined_score': ['mean', 'std']
        }).round(3)
    
    def get_demographic_analysis(self) -> Dict:
        """Analyze anomaly detection by demographic factors."""
        analysis = {}
        
        # Age analysis
        analysis['age'] = {
            'normal_mean': self.results_df[~self.results_df['iso_anomaly']]['age'].mean(),
            'anomaly_mean': self.results_df[self.results_df['iso_anomaly']]['age'].mean(),
            'difference': self.results_df[self.results_df['iso_anomaly']]['age'].mean() - 
                         self.results_df[~self.results_df['iso_anomaly']]['age'].mean()
        }
        
        # APOE4 analysis
        analysis['apoe4'] = self.results_df.groupby('apoe4_copies')['iso_anomaly'].agg(['sum', 'count', 'mean']).to_dict()
        
        # Gender analysis
        analysis['gender'] = self.results_df.groupby('gender')['iso_anomaly'].agg(['sum', 'count', 'mean']).to_dict()
        
        # Education analysis
        analysis['education'] = {
            'normal_mean': self.results_df[~self.results_df['iso_anomaly']]['education'].mean(),
            'anomaly_mean': self.results_df[self.results_df['iso_anomaly']]['education'].mean(),
            'difference': self.results_df[self.results_df['iso_anomaly']]['education'].mean() - 
                         self.results_df[~self.results_df['iso_anomaly']]['education'].mean()
        }
        
        return analysis
    
    def generate_summary_statistics(self) -> Dict:
        """Generate comprehensive summary statistics."""
        total_patients = len(self.results_df)
        
        summary = {
            'total_patients': total_patients,
            'iso_anomalies': int(self.results_df['iso_anomaly'].sum()),
            'iso_anomaly_rate': self.results_df['iso_anomaly'].mean(),
            'dbscan_noise': int(self.results_df['dbscan_noise'].sum()),
            'dbscan_noise_rate': self.results_df['dbscan_noise'].mean(),
            'consensus_anomalies': int((self.results_df['iso_anomaly'] & self.results_df['dbscan_noise']).sum()),
            'union_anomalies': int((self.results_df['iso_anomaly'] | self.results_df['dbscan_noise']).sum()),
            'agreement_rate': (self.results_df['iso_anomaly'] == self.results_df['dbscan_noise']).mean()
        }
        
        # Score statistics
        summary['iso_score_stats'] = {
            'mean': self.results_df['iso_score'].mean(),
            'std': self.results_df['iso_score'].std(),
            'min': self.results_df['iso_score'].min(),
            'max': self.results_df['iso_score'].max()
        }
        
        # Pattern-specific statistics
        summary['pattern_detection_rates'] = {}
        for pattern in self.results_df['pattern_type'].unique():
            pattern_data = self.results_df[self.results_df['pattern_type'] == pattern]
            summary['pattern_detection_rates'][pattern] = pattern_data['iso_anomaly'].mean()
        
        return summary
    
    def export_results(self, filepath: str, format: str = 'csv'):
        """Export results to file."""
        if format == 'csv':
            self.results_df.to_csv(filepath, index=False)
        elif format == 'excel':
            self.results_df.to_excel(filepath, index=False)
        else:
            raise ValueError("Format must be 'csv' or 'excel'")


def main():
    """Test utility functions."""
    print("Utility functions module loaded successfully!")


if __name__ == "__main__":
    main()