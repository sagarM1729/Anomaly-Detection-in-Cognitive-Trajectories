"""
Main analysis pipeline for anomaly detection in cognitive trajectories.
Integrates all components and runs the complete analysis workflow.
"""

import os
import sys
import numpy as np
import pandas as pd
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Import project modules
from data_loader import RealDatasetLoader
from preprocessing import TrajectoryFeatureEngine, DataValidator
from anomaly_detection import AnomalyDetector, CrossValidator
from visualization import AnomalyVisualizer

# For interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Feature importance analysis will be limited.")


class CognitiveAnomalyPipeline:
    """
    Complete pipeline for anomaly detection in cognitive trajectories.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the pipeline with configuration."""
        self.config_path = config_path
        self.load_config()
        self.data_config = self.config.get('data', {})
        self.real_dataset_path = self.data_config.get('real_dataset_path')
        if not self.data_config.get('use_real_dataset', True):
            print("Warning: Synthetic data mode is no longer supported; defaulting to real dataset loading.")
        self.real_data_summary = {}

        # Initialize components
        self.real_data_loader = RealDatasetLoader(config_path)
        self.feature_engine = TrajectoryFeatureEngine(config_path)
        self.anomaly_detector = AnomalyDetector(config_path)
        self.cross_validator = CrossValidator(config_path)
        self.visualizer = AnomalyVisualizer(config_path)

        # Results storage
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create results directory
        self.results_dir = Path("results") / f"analysis_{self.timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        print(f"Pipeline initialized. Results will be saved to: {self.results_dir}")
    
    def load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def run_complete_analysis(
        self,
        data_file: Optional[str] = None
    ) -> Dict:
        """
        Run the complete anomaly detection analysis pipeline.
        """
        print("="*60)
        print("COGNITIVE TRAJECTORY ANOMALY DETECTION PIPELINE")
        print("="*60)

        dataset_source = data_file or self.real_dataset_path
        if not dataset_source:
            raise ValueError("No dataset path provided. Set 'real_dataset_path' in the configuration or supply --data-file.")

        # Step 1: Data Loading
        print("\n1. Loading real-world dataset...")
        df = self.load_real_data(dataset_source)
        
        # Step 2: Data validation and quality assessment
        print("\n2. Validating data quality...")
        self.validate_data_quality(df)
        
        # Step 3: Feature engineering
        print("\n3. Engineering trajectory features...")
        patient_ids, features, processed_features = self.engineer_features(df)
        
        # Step 4: Anomaly detection
        print("\n4. Detecting anomalies...")
        anomaly_results = self.detect_anomalies(processed_features)
        
        # Step 5: Cross-validation
        print("\n5. Cross-validating results...")
        cv_results = self.cross_validate_anomalies(df, patient_ids)
        
        # Step 6: Feature importance analysis
        print("\n6. Analyzing feature importance...")
        importance_results = self.analyze_feature_importance(processed_features, anomaly_results)
        
        # Step 7: Visualization
        print("\n7. Generating visualizations...")
        self.generate_visualizations(df, processed_features, patient_ids, anomaly_results)
        
        # Step 8: Results compilation
        print("\n8. Compiling results...")
        final_results = self.compile_results(
            df, patient_ids, anomaly_results, cv_results, importance_results
        )
        
        # Step 9: Save results
        print("\n9. Saving results...")
        self.save_results(final_results)
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE!")
        print(f"Results saved to: {self.results_dir}")
        print(f"{'='*60}")
        
        return final_results
    
    def load_real_data(self, data_file: Optional[str] = None) -> pd.DataFrame:
        """Load and harmonize a real-world cognitive dataset."""
        dataset_path = data_file or self.real_dataset_path
        if not dataset_path:
            raise ValueError("No dataset path provided for real data loading.")

        df = self.real_data_loader.load_dataset(dataset_path)
        self.real_data_summary = self.real_data_loader.get_summary()

        data_path = self.real_data_summary.get('data_path', dataset_path)
        print(f"Loaded data from: {data_path}")
        print(f"Dataset shape (after harmonization): {df.shape}")
        print(f"Unique patients: {df['patient_id'].nunique()}")
        mean_visits = self.real_data_summary.get('mean_visits_per_patient')
        if mean_visits is not None:
            print(f"Average visits per patient: {mean_visits:.2f}")
        else:
            print("Average visits per patient: N/A")

        if 'diagnosis' in df.columns:
            diagnosis_counts = df.groupby('patient_id').first()['diagnosis'].value_counts(dropna=False)
            print("Diagnosis distribution (baseline):")
            print(diagnosis_counts.head(10))

        return df
    
    def validate_data_quality(self, df: pd.DataFrame):
        """Validate and report data quality."""
        validator = DataValidator()
        quality_metrics = validator.validate_data_quality(df, self.config['data']['visit_months'])
        validator.print_quality_report(quality_metrics)
        
        # Store quality metrics
        self.results['data_quality'] = quality_metrics
        
        # Save quality report
        quality_path = self.results_dir / "data_quality_report.json"
        with open(quality_path, 'w') as f:
            json.dump(quality_metrics, f, indent=2, default=str)
    
    def engineer_features(self, df: pd.DataFrame) -> tuple:
        """Engineer features from trajectory data."""
        # Build trajectory features
        patient_ids, features, valid_patients = self.feature_engine.build_trajectory_features(df)
        
        # Preprocess features
        processed_features = self.feature_engine.preprocess_features(features, fit_transform=True)
        
        # Store feature information
        self.results['feature_info'] = {
            'original_patients': df['patient_id'].nunique(),
            'valid_patients': len(valid_patients),
            'raw_features_shape': features.shape,
            'processed_features_shape': processed_features.shape,
            'feature_names': self.feature_engine.get_feature_importance_names()
        }
        
        print(f"Feature engineering completed:")
        print(f"  Valid patients: {len(valid_patients)}")
        print(f"  Raw features shape: {features.shape}")
        print(f"  Processed features shape: {processed_features.shape}")
        
        return patient_ids, features, processed_features
    
    def detect_anomalies(self, features: np.ndarray) -> Dict:
        """Detect anomalies using multiple methods."""
        results = {}
        
        # Isolation Forest
        print("  Fitting Isolation Forest...")
        self.anomaly_detector.fit_isolation_forest(features)
        iso_predictions, iso_scores = self.anomaly_detector.predict_isolation_forest(features)
        
        results['iso_predictions'] = iso_predictions
        results['iso_scores'] = iso_scores
        
        # DBSCAN
        print("  Fitting DBSCAN...")
        self.anomaly_detector.fit_dbscan(features)
        dbscan_predictions, cluster_labels = self.anomaly_detector.predict_dbscan(features)
        
        results['dbscan_predictions'] = dbscan_predictions
        results['cluster_labels'] = cluster_labels
        
        # Consensus analysis
        print("  Computing consensus anomalies...")
        consensus = self.anomaly_detector.consensus_anomaly_detection(
            iso_predictions, dbscan_predictions
        )
        results['consensus'] = consensus
        
        # Anomaly ranking
        anomaly_ranks, combined_scores = self.anomaly_detector.rank_anomalies(
            iso_scores, dbscan_predictions
        )
        results['anomaly_ranks'] = anomaly_ranks
        results['combined_scores'] = combined_scores
        
        # Summary statistics
        results['summary'] = {
            'iso_anomaly_count': int(np.sum(iso_predictions == -1)),
            'iso_anomaly_rate': float(np.mean(iso_predictions == -1)),
            'dbscan_noise_count': int(np.sum(dbscan_predictions)),
            'dbscan_noise_rate': float(np.mean(dbscan_predictions)),
            'consensus_count': int(consensus['intersection_count']),
            'agreement_rate': float(consensus['agreement_rate'])
        }
        
        print(f"  Isolation Forest: {results['summary']['iso_anomaly_count']} anomalies")
        print(f"  DBSCAN: {results['summary']['dbscan_noise_count']} noise points")
        print(f"  Consensus: {results['summary']['consensus_count']} anomalies")
        print(f"  Agreement rate: {results['summary']['agreement_rate']:.3f}")
        
        return results
    
    def cross_validate_anomalies(self, df: pd.DataFrame, patient_ids: np.ndarray) -> Dict:
        """Cross-validate anomaly detection results."""
        try:
            # Create temporal splits
            splits = self.cross_validator.temporal_split(df, patient_ids)
            
            if len(splits) == 0:
                print("  Warning: No valid temporal splits found")
                return {'error': 'No valid temporal splits'}
            
            print(f"  Created {len(splits)} temporal splits")
            
            # Validate consistency
            consistency_results, all_scores = self.cross_validator.validate_anomaly_consistency(
                self.anomaly_detector, splits, self.feature_engine, df
            )
            
            print(f"  Cross-validation completed")
            if 'common_patients' in consistency_results:
                print(f"  Common patients across splits: {consistency_results['common_patients']}")
                print(f"  Score correlation: {consistency_results.get('score_correlation_mean', 'N/A'):.3f}")
            
            return {
                'consistency_results': consistency_results,
                'temporal_scores': all_scores,
                'n_splits': len(splits)
            }
            
        except Exception as e:
            print(f"  Warning: Cross-validation failed: {str(e)}")
            return {'error': str(e)}
    
    def analyze_feature_importance(self, features: np.ndarray, anomaly_results: Dict) -> Dict:
        """Analyze feature importance for anomaly detection."""
        importance_results = {}
        
        try:
            if SHAP_AVAILABLE:
                print("  Computing SHAP feature importance...")
                
                # Use Isolation Forest for SHAP analysis
                if hasattr(self.anomaly_detector.isolation_forest, 'decision_function'):
                    explainer = shap.Explainer(self.anomaly_detector.isolation_forest.decision_function, features)
                    shap_values = explainer(features[:100])  # Limit for computational efficiency
                    
                    importance_results['shap_values'] = shap_values.values
                    importance_results['shap_feature_importance'] = np.mean(np.abs(shap_values.values), axis=0)
                    importance_results['feature_names'] = self.feature_engine.get_feature_importance_names()
                    
                    print(f"  SHAP analysis completed for {len(shap_values)} samples")
                
            else:
                print("  SHAP not available, using basic feature analysis...")
                
            # Basic feature statistics for anomalies vs normal
            iso_anomalies = anomaly_results['iso_predictions'] == -1
            
            if np.sum(iso_anomalies) > 0 and np.sum(~iso_anomalies) > 0:
                # Compare feature distributions
                anomaly_features = features[iso_anomalies]
                normal_features = features[~iso_anomalies]
                
                feature_differences = np.mean(anomaly_features, axis=0) - np.mean(normal_features, axis=0)
                feature_importance_basic = np.abs(feature_differences)
                
                importance_results['basic_feature_importance'] = feature_importance_basic
                importance_results['feature_differences'] = feature_differences
                
                print(f"  Basic feature analysis completed")
            
        except Exception as e:
            print(f"  Warning: Feature importance analysis failed: {str(e)}")
            importance_results['error'] = str(e)
        
        return importance_results
    
    def generate_visualizations(self, df: pd.DataFrame, features: np.ndarray, 
                              patient_ids: np.ndarray, anomaly_results: Dict):
        """Generate comprehensive visualizations."""
        try:
            # Trajectory overview
            print("  Creating trajectory overview...")
            fig1 = self.visualizer.plot_trajectory_overview(
                df, patient_ids, anomaly_results['consensus'],
                save_path=self.results_dir / "trajectory_overview.png"
            )
            
            # Anomaly embedding
            print("  Creating anomaly embedding plots...")
            fig2 = self.visualizer.plot_anomaly_embedding(
                features, patient_ids, anomaly_results,
                save_path=self.results_dir / "anomaly_embedding.png"
            )
            
            # Cluster analysis
            print("  Creating cluster analysis...")
            fig3 = self.visualizer.plot_cluster_analysis(
                features, anomaly_results['cluster_labels'], patient_ids,
                save_path=self.results_dir / "cluster_analysis.png"
            )
            
            # Interactive dashboard
            print("  Creating interactive dashboard...")
            fig4 = self.visualizer.create_interactive_dashboard(
                df, features, patient_ids, anomaly_results,
                save_path=self.results_dir / "interactive_dashboard.html"
            )
            
            print("  All visualizations generated successfully")
            
        except Exception as e:
            print(f"  Warning: Visualization generation failed: {str(e)}")
    
    def compile_results(self, df: pd.DataFrame, patient_ids: np.ndarray,
                       anomaly_results: Dict, cv_results: Dict, 
                       importance_results: Dict) -> Dict:
        """Compile all results into a comprehensive summary."""
        
        # Top anomalous patients
        top_anomaly_indices = anomaly_results['anomaly_ranks'][:20]
        top_anomalies = [(int(patient_ids[i]), float(anomaly_results['combined_scores'][i])) 
                        for i in top_anomaly_indices]
        
        # Patient-level results
        patient_results = pd.DataFrame({
            'patient_id': patient_ids,
            'iso_score': anomaly_results['iso_scores'],
            'iso_anomaly': anomaly_results['iso_predictions'] == -1,
            'dbscan_noise': anomaly_results['dbscan_predictions'],
            'cluster_label': anomaly_results['cluster_labels'],
            'combined_score': anomaly_results['combined_scores'],
            'anomaly_rank': range(len(patient_ids))
        })
        
        # Sort by combined anomaly score
        patient_results = patient_results.sort_values('combined_score', ascending=False)
        patient_results['anomaly_rank'] = range(len(patient_results))
        
        # Add demographics and clinical data when available
        demo_columns = [
            col for col in [
                'age',
                'gender',
                'education',
                'apoe4_copies',
                'pattern_type',
                'diagnosis',
                'baseline_diagnosis'
            ] if col in df.columns
        ]

        if demo_columns:
            demo_data = df.groupby('patient_id').first()[demo_columns].reset_index()
            patient_results = patient_results.merge(demo_data, on='patient_id', how='left')
        
        # Compile final results
        final_results = {
            'analysis_timestamp': self.timestamp,
            'config': self.config,
            'dataset_info': {
                'total_patients': int(df['patient_id'].nunique()),
                'valid_patients': len(patient_ids),
                'total_visits': len(df),
                'feature_dims': self.results['feature_info']['processed_features_shape']
            },
            'real_data_summary': self.real_data_summary if self.real_data_summary else None,
            'anomaly_summary': anomaly_results['summary'],
            'top_anomalies': top_anomalies,
            'patient_results': patient_results.to_dict('records'),
            'cross_validation': cv_results.get('consistency_results', {}),
            'feature_importance': importance_results,
            'data_quality': self.results.get('data_quality', {}),
            'feature_info': self.results.get('feature_info', {})
        }
        
        return final_results
    
    def save_results(self, results: Dict):
        """Save all results to files."""
        # Save main results as JSON
        results_json = results.copy()
        
        # Convert pandas DataFrame to dict for JSON serialization
        if 'patient_results' in results_json:
            results_json['patient_results'] = results['patient_results']
        
        # Remove non-serializable items
        for key in ['shap_values']:
            if key in results_json.get('feature_importance', {}):
                del results_json['feature_importance'][key]
        
        results_path = self.results_dir / "analysis_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2, default=str)
        
        # Save patient-level results as CSV
        patient_df = pd.DataFrame(results['patient_results'])
        patient_path = self.results_dir / "patient_anomaly_scores.csv"
        patient_df.to_csv(patient_path, index=False)
        
        # Generate and save summary report
        summary_report = self.visualizer.generate_summary_report(
            results, save_path=self.results_dir / "summary_report.txt"
        )
        
        print(f"Results saved:")
        print(f"  JSON results: {results_path}")
        print(f"  Patient scores: {patient_path}")
        print(f"  Summary report: {self.results_dir / 'summary_report.txt'}")


def main():
    """Run the complete anomaly detection pipeline."""
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Cognitive Trajectory Anomaly Detection')
    parser.add_argument('--config', default='config/config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--data-file', default=None,
                       help='Path to a real-world longitudinal dataset (CSV)')
    
    args = parser.parse_args()
    
    try:
        # Initialize and run pipeline
        pipeline = CognitiveAnomalyPipeline(args.config)
        results = pipeline.run_complete_analysis(data_file=args.data_file)
        
        # Print summary
        print(f"\nANALYSIS SUMMARY:")
        print(f"Valid patients analyzed: {results['dataset_info']['valid_patients']}")
        print(f"Isolation Forest anomalies: {results['anomaly_summary']['iso_anomaly_count']}")
        print(f"DBSCAN noise points: {results['anomaly_summary']['dbscan_noise_count']}")
        print(f"Consensus anomalies: {results['anomaly_summary']['consensus_count']}")
        
        if results['top_anomalies']:
            print(f"\nTop 5 Most Anomalous Patients:")
            for i, (pid, score) in enumerate(results['top_anomalies'][:5]):
                print(f"  {i+1}. Patient {pid}: {score:.3f}")
        
        return results
        
    except Exception as e:
        print(f"Error running analysis pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()