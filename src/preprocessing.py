"""
Data preprocessing module for feature engineering and trajectory analysis.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Dict, Optional
import yaml
import warnings
warnings.filterwarnings('ignore')

from data_loader import RealDatasetLoader


class TrajectoryFeatureEngine:
    """
    Feature engineering for longitudinal cognitive trajectory analysis.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the feature engine with configuration."""
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.config = config['features']
                self.visit_months = config['data']['visit_months']
        else:
            # Default configuration
            self.config = {
                'interpolation_method': 'linear',
                'max_missing_rate': 0.3,
                'standardize': True,
                'pca_variance_threshold': 0.9
            }
            self.visit_months = [0, 6, 12, 18, 24]
        
        self.scaler = StandardScaler()
        self.pca = None
        self.feature_names = []
    
    def build_trajectory_features(self, df: pd.DataFrame, 
                                  visit_months: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Build feature matrix from longitudinal cognitive data.
        
        Args:
            df: DataFrame with longitudinal data
            visit_months: Optional override for visit schedule (useful for temporal CV splits)
        
        Returns:
            patient_ids: Array of patient IDs
            feature_matrix: Feature matrix (n_patients, n_features)
            valid_patients: List of patient IDs with sufficient data
        """
        print("Building trajectory features...")
        
        # Use provided visit months or default from config
        months_to_use = visit_months if visit_months is not None else self.visit_months
        
        # Group by patient and process each trajectory
        trajectories = []
        patient_ids = []
        valid_patients = []
        
        for patient_id, group in df.groupby('patient_id'):
            # Pivot to wide format
            group_pivot = group.set_index('visit_month')[['mmse', 'faq']]
            
            # Reindex to ensure all visit months are present
            group_pivot = group_pivot.reindex(months_to_use)
            
            # Check missing data rate
            missing_rate = group_pivot.isnull().mean().mean()
            if missing_rate > self.config['max_missing_rate']:
                continue
            
            # Interpolate missing values
            if self.config['interpolation_method'] == 'linear':
                group_pivot = group_pivot.interpolate(method='linear', limit_direction='both')
            else:
                # Forward fill then backward fill
                group_pivot = group_pivot.fillna(method='ffill').fillna(method='bfill')
            
            # Extract various trajectory features
            features = self._extract_trajectory_features(group_pivot, months_to_use)
            
            trajectories.append(features)
            patient_ids.append(patient_id)
            valid_patients.append(patient_id)
        
        if not trajectories:
            raise ValueError("No valid trajectories found after preprocessing")
        
        feature_matrix = np.vstack(trajectories)
        
        print(f"Built features for {len(valid_patients)} patients")
        print(f"Feature matrix shape: {feature_matrix.shape}")
        
        return np.array(patient_ids), feature_matrix, valid_patients
    
    def _extract_trajectory_features(self, trajectory: pd.DataFrame, 
                                     visit_months: Optional[List[int]] = None) -> np.ndarray:
        """Extract multiple types of features from a single trajectory."""
        features = []
        feature_names = []
        
        # Use provided visit months or default from config
        months_to_use = visit_months if visit_months is not None else self.visit_months
        
        # 1. Raw values (flattened time series)
        raw_features = trajectory.values.flatten()
        features.extend(raw_features)
        for visit in months_to_use:
            for measure in ['mmse', 'faq']:
                feature_names.append(f'{measure}_month_{visit}')
        
        # 2. Slope features (linear trend)
        months = np.array(months_to_use)
        for col in trajectory.columns:
            values = trajectory[col].values
            mask = ~np.isnan(values)
            if mask.sum() >= 2:
                slope = np.polyfit(months[mask], values[mask], 1)[0]
            else:
                slope = np.nan
            features.append(slope)
            feature_names.append(f'{col}_slope')
        
        # 3. Acceleration features (second derivative)
        for col in trajectory.columns:
            values = trajectory[col].values
            mask = ~np.isnan(values)
            if mask.sum() >= 3:
                # Second difference as approximation of acceleration
                acceleration = np.mean(np.diff(values[mask], 2))
            else:
                acceleration = np.nan
            features.append(acceleration)
            feature_names.append(f'{col}_acceleration')
        
        # 4. Variability features
        for col in trajectory.columns:
            values = trajectory[col].values
            mask = ~np.isnan(values)
            if mask.sum() >= 2:
                observed = values[mask]
                # Coefficient of variation
                cv = np.std(observed) / (np.mean(observed) + 1e-8)
                range_val = np.max(observed) - np.min(observed)
            else:
                cv = np.nan
                range_val = np.nan
            features.append(cv)
            feature_names.append(f'{col}_cv')
            features.append(range_val)
            feature_names.append(f'{col}_range')
        
        # 5. Change from baseline features
        baseline_mmse = trajectory['mmse'].iloc[0]
        baseline_faq = trajectory['faq'].iloc[0]
        
        for i, visit in enumerate(months_to_use[1:], 1):
            mmse_change = trajectory['mmse'].iloc[i] - baseline_mmse if not np.isnan(trajectory['mmse'].iloc[i]) else np.nan
            faq_change = trajectory['faq'].iloc[i] - baseline_faq if not np.isnan(trajectory['faq'].iloc[i]) else np.nan
            features.extend([mmse_change, faq_change])
            feature_names.extend([f'mmse_change_month_{visit}', f'faq_change_month_{visit}'])
        
        # 6. Ratio features
        for i in range(len(months_to_use)):
            mmse_val = trajectory['mmse'].iloc[i]
            faq_val = trajectory['faq'].iloc[i]
            if not np.isnan(mmse_val) and not np.isnan(faq_val):
                ratio = mmse_val / (faq_val + 1e-8)
            else:
                ratio = np.nan
            features.append(ratio)
            feature_names.append(f'mmse_faq_ratio_month_{months_to_use[i]}')
        
        # Store feature names for the first trajectory
        if not self.feature_names:
            self.feature_names = feature_names
        
        return np.array(features)
    
    def preprocess_features(self, feature_matrix: np.ndarray, 
                          fit_transform: bool = True) -> np.ndarray:
        """
        Preprocess feature matrix with standardization and dimensionality reduction.
        """
        print("Preprocessing features...")
        
        # Handle any remaining NaN values
        imputer = SimpleImputer(strategy='median')
        feature_matrix = imputer.fit_transform(feature_matrix)
        
        # Standardization
        if self.config['standardize']:
            if fit_transform:
                feature_matrix = self.scaler.fit_transform(feature_matrix)
            else:
                feature_matrix = self.scaler.transform(feature_matrix)
        
        # PCA for dimensionality reduction
        if self.config['pca_variance_threshold'] < 1.0:
            if fit_transform:
                self.pca = PCA(n_components=self.config['pca_variance_threshold'], 
                              random_state=42)
                feature_matrix = self.pca.fit_transform(feature_matrix)
                print(f"PCA reduced features to {feature_matrix.shape[1]} components")
                print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
            else:
                if self.pca is not None:
                    feature_matrix = self.pca.transform(feature_matrix)
        
        print(f"Final feature matrix shape: {feature_matrix.shape}")
        return feature_matrix
    
    def get_feature_importance_names(self) -> List[str]:
        """Get interpretable feature names for importance analysis."""
        if self.pca is not None:
            return [f'PC_{i+1}' for i in range(self.pca.n_components_)]
        else:
            return self.feature_names
    
    def inverse_transform_features(self, features: np.ndarray) -> np.ndarray:
        """Inverse transform features for interpretation."""
        if self.pca is not None:
            features = self.pca.inverse_transform(features)
        
        if self.config['standardize']:
            features = self.scaler.inverse_transform(features)
        
        return features


class DataValidator:
    """Utility class for data quality validation."""
    
    @staticmethod
    def validate_data_quality(df: pd.DataFrame, visit_months: List[int]) -> Dict:
        """Validate data quality and return quality metrics."""
        quality_metrics = {}
        
        # Basic statistics
        quality_metrics['total_patients'] = df['patient_id'].nunique()
        quality_metrics['total_visits'] = len(df)
        quality_metrics['expected_visits'] = quality_metrics['total_patients'] * len(visit_months)
        
        # Missing data analysis
        quality_metrics['missing_mmse'] = df['mmse'].isnull().sum()
        quality_metrics['missing_faq'] = df['faq'].isnull().sum()
        quality_metrics['missing_rate_mmse'] = df['mmse'].isnull().mean()
        quality_metrics['missing_rate_faq'] = df['faq'].isnull().mean()
        
        # Patient-level missingness
        patient_missing = df.groupby('patient_id')[['mmse', 'faq']].apply(
            lambda x: x.isnull().mean().mean()
        )
        quality_metrics['patients_high_missing'] = (patient_missing > 0.5).sum()
        
        # Visit completeness
        visit_counts = df['patient_id'].value_counts()
        quality_metrics['complete_patients'] = (visit_counts == len(visit_months)).sum()
        quality_metrics['partial_patients'] = quality_metrics['total_patients'] - quality_metrics['complete_patients']
        
        # Value ranges
        quality_metrics['mmse_range'] = [df['mmse'].min(), df['mmse'].max()]
        quality_metrics['faq_range'] = [df['faq'].min(), df['faq'].max()]
        
        return quality_metrics
    
    @staticmethod
    def print_quality_report(quality_metrics: Dict):
        """Print a formatted data quality report."""
        print("\n" + "="*50)
        print("DATA QUALITY REPORT")
        print("="*50)
        
        print(f"Total patients: {quality_metrics['total_patients']}")
        print(f"Total visits: {quality_metrics['total_visits']}")
        print(f"Expected visits: {quality_metrics['expected_visits']}")
        print(f"Data completeness: {quality_metrics['total_visits']/quality_metrics['expected_visits']:.2%}")
        
        print(f"\nMissing Data:")
        print(f"  MMSE missing: {quality_metrics['missing_mmse']} ({quality_metrics['missing_rate_mmse']:.2%})")
        print(f"  FAQ missing: {quality_metrics['missing_faq']} ({quality_metrics['missing_rate_faq']:.2%})")
        
        print(f"\nPatient Completeness:")
        print(f"  Complete patients: {quality_metrics['complete_patients']}")
        print(f"  Partial patients: {quality_metrics['partial_patients']}")
        print(f"  High missing patients: {quality_metrics['patients_high_missing']}")
        
        print(f"\nValue Ranges:")
        print(f"  MMSE: {quality_metrics['mmse_range'][0]:.1f} - {quality_metrics['mmse_range'][1]:.1f}")
        print(f"  FAQ: {quality_metrics['faq_range'][0]:.1f} - {quality_metrics['faq_range'][1]:.1f}")
        
        print("="*50)


def main():
    """Test the preprocessing pipeline using the configured real dataset."""
    loader = RealDatasetLoader('config/config.yaml')
    df = loader.load_dataset()
    
    # Validate data quality
    visit_months = [0, 6, 12, 18, 24]
    validator = DataValidator()
    quality_metrics = validator.validate_data_quality(df, visit_months)
    validator.print_quality_report(quality_metrics)
    
    # Build features
    feature_engine = TrajectoryFeatureEngine('config/config.yaml')
    patient_ids, feature_matrix, valid_patients = feature_engine.build_trajectory_features(df)
    
    # Preprocess features
    processed_features = feature_engine.preprocess_features(feature_matrix)
    
    print(f"\nFeature engineering completed:")
    print(f"Original patients: {df['patient_id'].nunique()}")
    print(f"Valid patients: {len(valid_patients)}")
    print(f"Feature matrix shape: {processed_features.shape}")


if __name__ == "__main__":
    main()