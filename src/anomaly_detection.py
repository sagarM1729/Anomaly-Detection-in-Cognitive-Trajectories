"""
Anomaly detection module implementing Isolation Forest and DBSCAN algorithms
for identifying atypical cognitive decline patterns.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score
from typing import Tuple, Dict, List, Optional
import yaml
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetector:
    """
    Comprehensive anomaly detection using multiple unsupervised methods.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the anomaly detector with configuration."""
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.iso_config = config['isolation_forest']
                self.db_config = config['dbscan']
        else:
            # Default configuration
            self.iso_config = {
                'contamination': 0.08,
                'n_estimators': 512,
                'max_samples': 'auto',
                'random_state': 42,
                'n_jobs': -1
            }
            self.db_config = {
                'eps_range': [0.3, 1.5],
                'min_samples_range': [3, 10],
                'metric': 'euclidean'
            }
        
        self.isolation_forest = None
        self.dbscan = None
        self.optimal_eps = None
        self.optimal_min_samples = None
    
    def fit_isolation_forest(self, X: np.ndarray) -> 'IsolationForest':
        """
        Fit Isolation Forest model for anomaly detection.
        """
        print("Fitting Isolation Forest...")
        
        self.isolation_forest = IsolationForest(
            contamination=self.iso_config['contamination'],
            n_estimators=self.iso_config['n_estimators'],
            max_samples=self.iso_config['max_samples'],
            random_state=self.iso_config['random_state'],
            n_jobs=self.iso_config['n_jobs']
        )
        
        self.isolation_forest.fit(X)
        
        print(f"Isolation Forest fitted with contamination={self.iso_config['contamination']}")
        return self.isolation_forest
    
    def predict_isolation_forest(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies using Isolation Forest.
        
        Returns:
            predictions: Binary predictions (-1 for anomaly, 1 for normal)
            scores: Anomaly scores (lower = more anomalous)
        """
        if self.isolation_forest is None:
            raise ValueError("Isolation Forest not fitted. Call fit_isolation_forest first.")
        
        predictions = self.isolation_forest.predict(X)
        scores = self.isolation_forest.decision_function(X)
        
        return predictions, scores
    
    def tune_dbscan_parameters(self, X: np.ndarray) -> Tuple[float, int]:
        """
        Tune DBSCAN parameters using k-distance plot and silhouette analysis.
        Optimized to reduce excessive noise rate by balancing cluster quality and noise tolerance.
        """
        print("Tuning DBSCAN parameters...")
        
        # Generate parameter grid - wider eps range for better cluster formation
        eps_values = np.linspace(self.db_config['eps_range'][0], 
                                self.db_config['eps_range'][1], 15)
        min_samples_values = range(self.db_config['min_samples_range'][0],
                                  self.db_config['min_samples_range'][1] + 1)
        
        best_score = -1
        best_eps = eps_values[0]
        best_min_samples = min_samples_values[0]
        best_noise_rate = 1.0
        
        # Try different parameter combinations
        for eps in eps_values:
            for min_samples in min_samples_values:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, 
                               metric=self.db_config['metric'])
                labels = dbscan.fit_predict(X)
                
                # Calculate noise rate
                noise_rate = np.sum(labels == -1) / len(labels)
                
                # Skip if noise rate is too high (>50%) or too low (<5%)
                if noise_rate > 0.5 or noise_rate < 0.05:
                    continue
                
                # Skip if all points are in one cluster
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters < 2:
                    continue
                
                # Calculate silhouette score (exclude noise points)
                non_noise_mask = labels != -1
                if np.sum(non_noise_mask) < 10:
                    continue
                
                try:
                    silhouette = silhouette_score(X[non_noise_mask], labels[non_noise_mask])
                    
                    # Combined score: balance silhouette quality and reasonable noise rate
                    # Prefer lower noise rates when silhouette scores are similar
                    combined_score = silhouette - (noise_rate * 0.3)
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_eps = eps
                        best_min_samples = min_samples
                        best_noise_rate = noise_rate
                except:
                    continue
        
        print(f"Best DBSCAN parameters: eps={best_eps:.3f}, min_samples={best_min_samples}")
        print(f"Expected noise rate: {best_noise_rate:.1%}")
        print(f"Quality score: {best_score:.3f}")
        
        self.optimal_eps = best_eps
        self.optimal_min_samples = best_min_samples
        
        return best_eps, best_min_samples
    
    def fit_dbscan(self, X: np.ndarray, eps: Optional[float] = None, 
                   min_samples: Optional[int] = None) -> 'DBSCAN':
        """
        Fit DBSCAN model for anomaly detection.
        """
        print("Fitting DBSCAN...")
        
        if eps is None or min_samples is None:
            eps, min_samples = self.tune_dbscan_parameters(X)
        
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples,
                            metric=self.db_config['metric'])
        self.dbscan.fit(X)
        
        # Count clusters and noise points
        labels = self.dbscan.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"DBSCAN results: {n_clusters} clusters, {n_noise} noise points")
        print(f"Noise rate: {n_noise/len(labels):.3f}")
        
        return self.dbscan
    
    def predict_dbscan(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies using DBSCAN.
        
        Returns:
            noise_predictions: Binary predictions (True for noise/anomaly)
            cluster_labels: Cluster labels (-1 for noise)
        """
        if self.dbscan is None:
            raise ValueError("DBSCAN not fitted. Call fit_dbscan first.")
        
        cluster_labels = self.dbscan.labels_
        noise_predictions = cluster_labels == -1
        
        return noise_predictions, cluster_labels
    
    def compute_k_distance(self, X: np.ndarray, k: int = 4) -> np.ndarray:
        """
        Compute k-distances for eps parameter tuning in DBSCAN.
        """
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors.fit(X)
        distances, indices = neighbors.kneighbors(X)
        
        # Sort distances for k-distance plot
        k_distances = np.sort(distances[:, k-1])[::-1]
        
        return k_distances
    
    def consensus_anomaly_detection(self, iso_predictions: np.ndarray, 
                                   dbscan_predictions: np.ndarray) -> Dict:
        """
        Combine results from multiple anomaly detection methods.
        """
        # Convert isolation forest predictions (-1, 1) to boolean (True, False)
        iso_anomalies = iso_predictions == -1
        db_anomalies = dbscan_predictions
        
        # Consensus methods
        consensus_results = {
            'intersection': iso_anomalies & db_anomalies,  # Both methods agree
            'union': iso_anomalies | db_anomalies,         # Either method detects
            'iso_only': iso_anomalies & ~db_anomalies,     # Only Isolation Forest
            'dbscan_only': db_anomalies & ~iso_anomalies,  # Only DBSCAN
            'agreement_rate': np.mean(iso_anomalies == db_anomalies)
        }
        
        # Count anomalies by method
        consensus_results['iso_count'] = np.sum(iso_anomalies)
        consensus_results['dbscan_count'] = np.sum(db_anomalies)
        consensus_results['intersection_count'] = np.sum(consensus_results['intersection'])
        consensus_results['union_count'] = np.sum(consensus_results['union'])
        
        return consensus_results
    
    def rank_anomalies(self, iso_scores: np.ndarray, 
                      dbscan_predictions: np.ndarray) -> np.ndarray:
        """
        Rank all samples by anomaly likelihood combining both methods.
        """
        # Normalize isolation forest scores to [0, 1]
        iso_scores_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
        
        # DBSCAN contribution (noise points get higher anomaly score)
        dbscan_scores = dbscan_predictions.astype(float)
        
        # Combined anomaly score (weighted average)
        combined_scores = 0.7 * (1 - iso_scores_norm) + 0.3 * dbscan_scores
        
        # Rank from most anomalous to least anomalous
        anomaly_ranks = np.argsort(combined_scores)[::-1]
        
        return anomaly_ranks, combined_scores


class CrossValidator:
    """
    Cross-validation for anomaly detection with temporal considerations.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.cv_config = config['cross_validation']
        else:
            self.cv_config = {
                'n_splits': 5,
                'test_months': [18, 24],
                'validation_method': 'temporal'
            }
    
    def temporal_split(self, df: pd.DataFrame, patient_ids: np.ndarray) -> List[Tuple]:
        """
        Create temporal splits for cross-validation.
        Split based on data availability, ensuring sufficient visits for feature extraction.
        """
        splits = []
        
        for test_month in self.cv_config['test_months']:
            # For each patient, check if they have:
            # 1. At least 3 visits before test_month (for training trajectory)
            # 2. At least 1 visit at or after test_month (for test)
            
            valid_train_patients = []
            for patient_id in patient_ids:
                patient_data = df[df['patient_id'] == patient_id]
                
                # Training visits: before test_month
                train_visits = patient_data[patient_data['visit_month'] < test_month]
                # Test visits: at or after test_month
                test_visits = patient_data[patient_data['visit_month'] >= test_month]
                
                # Need minimum 3 visits for training, at least 1 for test
                if len(train_visits) >= 3 and len(test_visits) >= 1:
                    valid_train_patients.append(patient_id)
            
            if len(valid_train_patients) > 20:  # Minimum viable split size
                train_mask = (df['visit_month'] < test_month) & df['patient_id'].isin(valid_train_patients)
                test_mask = (df['visit_month'] >= test_month) & df['patient_id'].isin(valid_train_patients)
                
                splits.append({
                    'train_patients': np.array(valid_train_patients),
                    'test_month': test_month,
                    'train_mask': train_mask,
                    'test_mask': test_mask
                })
                print(f"  Split at month {test_month}: {len(valid_train_patients)} valid patients")
            else:
                print(f"  Skipping split at month {test_month}: insufficient patients ({len(valid_train_patients)})")
        
        return splits
    
    def validate_anomaly_consistency(self, detector: AnomalyDetector, 
                                   splits: List[Dict], 
                                   feature_engine, df: pd.DataFrame) -> Dict:
        """
        Validate that anomalies are consistently detected across time splits.
        """
        consistency_results = {}
        all_anomaly_scores = {}
        
        for i, split in enumerate(splits):
            print(f"Validating split {i+1}: test_month={split['test_month']}")
            
            try:
                # Get training data up to test month - filter for valid patients only
                train_df = df[split['train_mask']].copy()
                
                # Determine available visit months in training data
                available_months = sorted(train_df['visit_month'].unique())
                print(f"  Available visit months in training: {available_months}")
                
                # Ensure we only include patients with enough visits
                patient_visit_counts = train_df.groupby('patient_id').size()
                patients_with_enough_visits = patient_visit_counts[patient_visit_counts >= 3].index
                train_df = train_df[train_df['patient_id'].isin(patients_with_enough_visits)]
                
                if len(train_df) == 0:
                    print(f"  Warning: No training data available for split {i+1}")
                    continue
                
                # Build features for training period using only available visit months
                patient_ids_split, train_features, valid_patients_split = feature_engine.build_trajectory_features(
                    train_df, 
                    visit_months=available_months
                )
                
                if len(valid_patients_split) < 10:
                    print(f"  Warning: Only {len(valid_patients_split)} valid patients in split {i+1}, skipping")
                    continue
                
                train_features = feature_engine.preprocess_features(train_features, fit_transform=True)
                
                # Fit models
                detector.fit_isolation_forest(train_features)
                detector.fit_dbscan(train_features)
                
                # Predict on training data
                iso_pred, iso_scores = detector.predict_isolation_forest(train_features)
                db_pred, _ = detector.predict_dbscan(train_features)
                
                # Store results
                split_key = f'month_{split["test_month"]}'
                all_anomaly_scores[split_key] = {
                    'patients': patient_ids_split,
                    'iso_scores': iso_scores,
                    'iso_predictions': iso_pred,
                    'db_predictions': db_pred
                }
                print(f"  Successfully validated split with {len(patient_ids_split)} patients")
                
            except Exception as e:
                print(f"  Error in split {i+1}: {str(e)}")
                continue
        
        # Analyze consistency across splits
        if len(all_anomaly_scores) > 1:
            consistency_results = self._analyze_consistency(all_anomaly_scores)
        else:
            consistency_results = {'error': 'Insufficient valid splits for consistency analysis'}
        
        return consistency_results, all_anomaly_scores
    
    def _analyze_consistency(self, all_scores: Dict) -> Dict:
        """Analyze consistency of anomaly detection across temporal splits."""
        consistency_metrics = {}
        
        # Find common patients across splits
        split_keys = list(all_scores.keys())
        common_patients = set(all_scores[split_keys[0]]['patients'])
        
        for key in split_keys[1:]:
            common_patients = common_patients.intersection(
                set(all_scores[key]['patients'])
            )
        
        common_patients = list(common_patients)
        
        if len(common_patients) < 10:
            return {'error': 'Insufficient common patients across splits'}
        
        # Calculate consistency metrics
        consistency_metrics['common_patients'] = len(common_patients)
        consistency_metrics['total_splits'] = len(split_keys)
        
        # Correlation of anomaly scores
        score_correlations = []
        for i in range(len(split_keys)):
            for j in range(i+1, len(split_keys)):
                key1, key2 = split_keys[i], split_keys[j]
                
                # Get scores for common patients
                scores1 = all_scores[key1]['iso_scores']
                scores2 = all_scores[key2]['iso_scores']
                
                # Align by patient ID
                patients1 = all_scores[key1]['patients']
                patients2 = all_scores[key2]['patients']
                
                common_idx1 = [np.where(patients1 == p)[0][0] for p in common_patients]
                common_idx2 = [np.where(patients2 == p)[0][0] for p in common_patients]
                
                correlation = np.corrcoef(scores1[common_idx1], scores2[common_idx2])[0, 1]
                score_correlations.append(correlation)
        
        consistency_metrics['score_correlation_mean'] = np.mean(score_correlations)
        consistency_metrics['score_correlation_std'] = np.std(score_correlations)
        
        return consistency_metrics


def main():
    """Test the anomaly detection pipeline."""
    # This would typically load preprocessed data
    print("Anomaly detection module created successfully!")
    print("Integration with main pipeline required for full testing.")


if __name__ == "__main__":
    main()