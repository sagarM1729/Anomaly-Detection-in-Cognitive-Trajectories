"""
Visualization module for anomaly detection results and trajectory analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import yaml
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")


class AnomalyVisualizer:
    """
    Comprehensive visualization for anomaly detection in cognitive trajectories.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize visualizer with configuration."""
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.viz_config = config['visualization']
                self.visit_months = config['data']['visit_months']
        else:
            # Default configuration
            self.viz_config = {
                'figure_size': [12, 8],
                'dpi': 300,
                'style': 'seaborn-v0_8',
                'color_palette': 'viridis'
            }
            self.visit_months = [0, 6, 12, 18, 24]
        
        # Set up matplotlib
        plt.rcParams['figure.figsize'] = self.viz_config['figure_size']
        plt.rcParams['figure.dpi'] = self.viz_config['dpi']
        
        # Color schemes
        self.colors = {
            'normal': '#1f77b4',
            'anomaly': '#d62728',
            'noise': '#ff7f0e',
            'cluster': sns.color_palette("Set1", 10)
        }
    
    def plot_trajectory_overview(self, df: pd.DataFrame, patient_ids: np.ndarray,
                               anomaly_labels: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot overview of all trajectories with anomaly highlighting.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Cognitive Trajectory Overview', fontsize=16, fontweight='bold')
        
        # Filter to valid patients
        df_valid = df[df['patient_id'].isin(patient_ids)].copy()
        
        # Add anomaly labels
        df_valid['is_anomaly'] = df_valid['patient_id'].map(
            lambda x: x in patient_ids[anomaly_labels.get('union', np.array([False]*len(patient_ids)))]
        )
        
        # Plot 1: MMSE trajectories
        self._plot_individual_trajectories(
            df_valid, 'mmse', 'MMSE Trajectories', axes[0, 0]
        )
        
        # Plot 2: FAQ trajectories  
        self._plot_individual_trajectories(
            df_valid, 'faq', 'FAQ Trajectories', axes[0, 1]
        )
        
        # Plot 3: Average trajectories by group
        self._plot_average_trajectories(df_valid, axes[1, 0])
        
        # Plot 4: Distribution comparison
        self._plot_distribution_comparison(df_valid, axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.viz_config['dpi'], bbox_inches='tight')
        
        return fig
    
    def _plot_individual_trajectories(self, df: pd.DataFrame, measure: str, 
                                    title: str, ax: plt.Axes):
        """Plot individual patient trajectories."""
        # Normal patients
        normal_patients = df[~df['is_anomaly']]['patient_id'].unique()
        for pid in normal_patients[:50]:  # Limit for visibility
            patient_data = df[df['patient_id'] == pid]
            ax.plot(patient_data['visit_month'], patient_data[measure], 
                   color=self.colors['normal'], alpha=0.3, linewidth=0.5)
        
        # Anomalous patients
        anomaly_patients = df[df['is_anomaly']]['patient_id'].unique()
        for pid in anomaly_patients:
            patient_data = df[df['patient_id'] == pid]
            ax.plot(patient_data['visit_month'], patient_data[measure], 
                   color=self.colors['anomaly'], alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Visit Month')
        ax.set_ylabel(measure.upper())
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Legend
        ax.plot([], [], color=self.colors['normal'], label='Normal', alpha=0.7)
        ax.plot([], [], color=self.colors['anomaly'], label='Anomaly', linewidth=2)
        ax.legend()
    
    def _plot_average_trajectories(self, df: pd.DataFrame, ax: plt.Axes):
        """Plot average trajectories for normal vs anomalous groups."""
        # Calculate group averages
        normal_avg = df[~df['is_anomaly']].groupby('visit_month')[['mmse', 'faq']].mean()
        anomaly_avg = df[df['is_anomaly']].groupby('visit_month')[['mmse', 'faq']].mean()
        
        # Plot MMSE
        ax.plot(normal_avg.index, normal_avg['mmse'], 
               color=self.colors['normal'], marker='o', linewidth=3, 
               label='Normal MMSE', markersize=8)
        ax.plot(anomaly_avg.index, anomaly_avg['mmse'], 
               color=self.colors['anomaly'], marker='s', linewidth=3,
               label='Anomaly MMSE', markersize=8)
        
        # Secondary y-axis for FAQ
        ax2 = ax.twinx()
        ax2.plot(normal_avg.index, normal_avg['faq'], 
                color=self.colors['normal'], marker='o', linewidth=3, 
                linestyle='--', label='Normal FAQ', markersize=8)
        ax2.plot(anomaly_avg.index, anomaly_avg['faq'], 
                color=self.colors['anomaly'], marker='s', linewidth=3,
                linestyle='--', label='Anomaly FAQ', markersize=8)
        
        ax.set_xlabel('Visit Month')
        ax.set_ylabel('MMSE Score', color='black')
        ax2.set_ylabel('FAQ Score', color='gray')
        ax.set_title('Average Trajectories by Group')
        ax.grid(True, alpha=0.3)
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    def _plot_distribution_comparison(self, df: pd.DataFrame, ax: plt.Axes):
        """Plot distribution comparison between normal and anomalous groups."""
        # Calculate change from baseline
        baseline_data = df[df['visit_month'] == 0][['patient_id', 'mmse', 'faq']]
        baseline_data.columns = ['patient_id', 'baseline_mmse', 'baseline_faq']
        
        final_data = df[df['visit_month'] == max(self.visit_months)][['patient_id', 'mmse', 'faq', 'is_anomaly']]
        
        comparison_df = final_data.merge(baseline_data, on='patient_id')
        comparison_df['mmse_change'] = comparison_df['mmse'] - comparison_df['baseline_mmse']
        comparison_df['faq_change'] = comparison_df['faq'] - comparison_df['baseline_faq']
        
        # Box plots
        positions = [1, 2, 4, 5]
        normal_mmse = comparison_df[~comparison_df['is_anomaly']]['mmse_change'].dropna()
        anomaly_mmse = comparison_df[comparison_df['is_anomaly']]['mmse_change'].dropna()
        normal_faq = comparison_df[~comparison_df['is_anomaly']]['faq_change'].dropna()
        anomaly_faq = comparison_df[comparison_df['is_anomaly']]['faq_change'].dropna()
        
        bp = ax.boxplot([normal_mmse, anomaly_mmse, normal_faq, anomaly_faq], 
                       positions=positions, patch_artist=True)
        
        # Color the boxes
        colors = [self.colors['normal'], self.colors['anomaly'], 
                 self.colors['normal'], self.colors['anomaly']]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xticklabels(['Normal\nMMSE', 'Anomaly\nMMSE', 'Normal\nFAQ', 'Anomaly\nFAQ'])
        ax.set_ylabel('Change from Baseline')
        ax.set_title('Change from Baseline Distribution')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    def plot_anomaly_embedding(self, features: np.ndarray, patient_ids: np.ndarray,
                             anomaly_results: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot 2D embedding of features with anomaly highlighting.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Space Analysis', fontsize=16, fontweight='bold')
        
        # Compute t-SNE embedding
        print("Computing t-SNE embedding...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
        embedding = tsne.fit_transform(features)
        
        # Plot 1: Isolation Forest results
        iso_anomalies = anomaly_results.get('iso_predictions', np.array([False]*len(features))) == -1
        self._plot_embedding_with_labels(
            embedding, iso_anomalies, 'Isolation Forest Anomalies', axes[0, 0]
        )
        
        # Plot 2: DBSCAN results
        db_anomalies = anomaly_results.get('dbscan_predictions', np.array([False]*len(features)))
        self._plot_embedding_with_labels(
            embedding, db_anomalies, 'DBSCAN Noise Points', axes[0, 1]
        )
        
        # Plot 3: Consensus anomalies
        consensus = anomaly_results.get('consensus', {})
        consensus_anomalies = consensus.get('intersection', np.array([False]*len(features)))
        self._plot_embedding_with_labels(
            embedding, consensus_anomalies, 'Consensus Anomalies', axes[1, 0]
        )
        
        # Plot 4: Anomaly scores
        iso_scores = anomaly_results.get('iso_scores', np.zeros(len(features)))
        self._plot_embedding_with_scores(
            embedding, iso_scores, 'Anomaly Scores', axes[1, 1]
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.viz_config['dpi'], bbox_inches='tight')
        
        return fig
    
    def _plot_embedding_with_labels(self, embedding: np.ndarray, labels: np.ndarray,
                                  title: str, ax: plt.Axes):
        """Plot embedding with binary labels."""
        # Normal points
        normal_mask = ~labels
        ax.scatter(embedding[normal_mask, 0], embedding[normal_mask, 1], 
                  c=self.colors['normal'], alpha=0.6, s=50, label='Normal')
        
        # Anomalous points
        anomaly_mask = labels
        if np.any(anomaly_mask):
            ax.scatter(embedding[anomaly_mask, 0], embedding[anomaly_mask, 1], 
                      c=self.colors['anomaly'], alpha=0.8, s=100, 
                      marker='^', label='Anomaly', edgecolors='black')
        
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_embedding_with_scores(self, embedding: np.ndarray, scores: np.ndarray,
                                  title: str, ax: plt.Axes):
        """Plot embedding with continuous scores."""
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                           c=scores, cmap='coolwarm', alpha=0.7, s=50)
        
        plt.colorbar(scatter, ax=ax, label='Anomaly Score')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    def plot_cluster_analysis(self, features: np.ndarray, cluster_labels: np.ndarray,
                            patient_ids: np.ndarray, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot cluster analysis results.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Cluster Analysis', fontsize=16, fontweight='bold')
        
        # Compute embedding for visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
        embedding = tsne.fit_transform(features)
        
        # Plot 1: Cluster visualization
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        for i, label in enumerate(unique_labels):
            if label == -1:
                # Noise points
                mask = cluster_labels == label
                axes[0, 0].scatter(embedding[mask, 0], embedding[mask, 1], 
                                 c=self.colors['noise'], alpha=0.6, s=50, 
                                 marker='x', label='Noise')
            else:
                # Cluster points
                mask = cluster_labels == label
                color = self.colors['cluster'][i % len(self.colors['cluster'])]
                axes[0, 0].scatter(embedding[mask, 0], embedding[mask, 1], 
                                 c=color, alpha=0.7, s=50, 
                                 label=f'Cluster {label}')
        
        axes[0, 0].set_xlabel('t-SNE 1')
        axes[0, 0].set_ylabel('t-SNE 2')
        axes[0, 0].set_title(f'DBSCAN Clustering ({n_clusters} clusters)')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Cluster sizes
        cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
        cluster_sizes.plot(kind='bar', ax=axes[0, 1], color=self.colors['cluster'][:len(cluster_sizes)])
        axes[0, 1].set_xlabel('Cluster Label')
        axes[0, 1].set_ylabel('Number of Patients')
        axes[0, 1].set_title('Cluster Sizes')
        axes[0, 1].tick_params(axis='x', rotation=0)
        
        # Plot 3: Distance to cluster centers
        if n_clusters > 0:
            self._plot_cluster_distances(features, cluster_labels, axes[1, 0])
        
        # Plot 4: Silhouette analysis
        if n_clusters > 1:
            self._plot_silhouette_analysis(features, cluster_labels, axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.viz_config['dpi'], bbox_inches='tight')
        
        return fig
    
    def _plot_cluster_distances(self, features: np.ndarray, labels: np.ndarray, ax: plt.Axes):
        """Plot distances to cluster centers."""
        from sklearn.metrics.pairwise import euclidean_distances
        
        unique_labels = np.unique(labels[labels != -1])
        distances = []
        cluster_names = []
        
        for label in unique_labels:
            mask = labels == label
            if np.sum(mask) > 1:
                cluster_data = features[mask]
                center = np.mean(cluster_data, axis=0)
                dists = euclidean_distances(cluster_data, center.reshape(1, -1)).flatten()
                distances.extend(dists)
                cluster_names.extend([f'Cluster {label}'] * len(dists))
        
        if distances:
            df_dist = pd.DataFrame({'distance': distances, 'cluster': cluster_names})
            sns.boxplot(data=df_dist, x='cluster', y='distance', ax=ax)
            ax.set_title('Distance to Cluster Centers')
            ax.tick_params(axis='x', rotation=45)
    
    def _plot_silhouette_analysis(self, features: np.ndarray, labels: np.ndarray, ax: plt.Axes):
        """Plot silhouette analysis."""
        from sklearn.metrics import silhouette_samples, silhouette_score
        
        # Remove noise points for silhouette analysis
        non_noise_mask = labels != -1
        if np.sum(non_noise_mask) < 2:
            ax.text(0.5, 0.5, 'Insufficient data for silhouette analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        features_clean = features[non_noise_mask]
        labels_clean = labels[non_noise_mask]
        
        if len(np.unique(labels_clean)) < 2:
            ax.text(0.5, 0.5, 'Need at least 2 clusters for silhouette analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        silhouette_avg = silhouette_score(features_clean, labels_clean)
        sample_silhouette_values = silhouette_samples(features_clean, labels_clean)
        
        y_lower = 10
        for i, label in enumerate(np.unique(labels_clean)):
            cluster_silhouette_values = sample_silhouette_values[labels_clean == label]
            cluster_silhouette_values.sort()
            
            size_cluster_i = cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = self.colors['cluster'][i % len(self.colors['cluster'])]
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values,
                           facecolor=color, edgecolor=color, alpha=0.7)
            
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(label))
            y_lower = y_upper + 10
        
        ax.axvline(x=silhouette_avg, color="red", linestyle="--", 
                  label=f'Average Score: {silhouette_avg:.3f}')
        ax.set_xlabel('Silhouette Coefficient Values')
        ax.set_ylabel('Cluster Label')
        ax.set_title('Silhouette Analysis')
        ax.legend()
    
    def create_interactive_dashboard(self, df: pd.DataFrame, features: np.ndarray,
                                   patient_ids: np.ndarray, anomaly_results: Dict,
                                   save_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive Plotly dashboard for exploration.
        """
        # Compute t-SNE for visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
        embedding = tsne.fit_transform(features)
        
        # Prepare data
        df_plot = pd.DataFrame({
            'patient_id': patient_ids,
            'tsne_1': embedding[:, 0],
            'tsne_2': embedding[:, 1],
            'iso_score': anomaly_results.get('iso_scores', np.zeros(len(patient_ids))),
            'iso_anomaly': anomaly_results.get('iso_predictions', np.ones(len(patient_ids))) == -1,
            'dbscan_anomaly': anomaly_results.get('dbscan_predictions', np.zeros(len(patient_ids), dtype=bool)),
            'cluster_label': anomaly_results.get('cluster_labels', np.zeros(len(patient_ids)))
        })
        
        # Add trajectory summaries
        trajectory_summary = df.groupby('patient_id').agg({
            'mmse': ['first', 'last', lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0],
            'faq': ['first', 'last', lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0]
        }).round(3)
        
        trajectory_summary.columns = ['mmse_baseline', 'mmse_final', 'mmse_slope',
                                    'faq_baseline', 'faq_final', 'faq_slope']
        
        df_plot = df_plot.merge(trajectory_summary, left_on='patient_id', 
                               right_index=True, how='left')
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Space (t-SNE)', 'Anomaly Scores', 
                          'MMSE Trajectories', 'FAQ Trajectories'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: t-SNE with anomaly highlighting
        normal_mask = ~(df_plot['iso_anomaly'] | df_plot['dbscan_anomaly'])
        anomaly_mask = df_plot['iso_anomaly'] | df_plot['dbscan_anomaly']
        
        fig.add_trace(
            go.Scatter(
                x=df_plot[normal_mask]['tsne_1'],
                y=df_plot[normal_mask]['tsne_2'],
                mode='markers',
                marker=dict(size=8, color='blue', opacity=0.6),
                name='Normal',
                hovertemplate='Patient: %{customdata}<br>t-SNE 1: %{x:.2f}<br>t-SNE 2: %{y:.2f}',
                customdata=df_plot[normal_mask]['patient_id']
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_plot[anomaly_mask]['tsne_1'],
                y=df_plot[anomaly_mask]['tsne_2'],
                mode='markers',
                marker=dict(size=12, color='red', symbol='triangle-up', opacity=0.8),
                name='Anomaly',
                hovertemplate='Patient: %{customdata}<br>t-SNE 1: %{x:.2f}<br>t-SNE 2: %{y:.2f}',
                customdata=df_plot[anomaly_mask]['patient_id']
            ),
            row=1, col=1
        )
        
        # Plot 2: Anomaly scores
        fig.add_trace(
            go.Scatter(
                x=df_plot['patient_id'],
                y=df_plot['iso_score'],
                mode='markers',
                marker=dict(size=8, color=df_plot['iso_score'], colorscale='Viridis', opacity=0.7),
                name='ISO Score',
                hovertemplate='Patient: %{x}<br>Anomaly Score: %{y:.3f}'
            ),
            row=1, col=2
        )
        
        # Add trajectory plots for top anomalies
        top_anomalies = df_plot.nlargest(10, 'iso_score')['patient_id'].values
        
        for i, pid in enumerate(top_anomalies[:5]):  # Show top 5
            patient_data = df[df['patient_id'] == pid]
            
            fig.add_trace(
                go.Scatter(
                    x=patient_data['visit_month'],
                    y=patient_data['mmse'],
                    mode='lines+markers',
                    name=f'Patient {pid}',
                    line=dict(width=3),
                    hovertemplate='Month: %{x}<br>MMSE: %{y:.1f}'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=patient_data['visit_month'],
                    y=patient_data['faq'],
                    mode='lines+markers',
                    name=f'Patient {pid}',
                    line=dict(width=3),
                    hovertemplate='Month: %{x}<br>FAQ: %{y:.1f}',
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Interactive Anomaly Detection Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="t-SNE 1", row=1, col=1)
        fig.update_yaxes(title_text="t-SNE 2", row=1, col=1)
        fig.update_xaxes(title_text="Patient ID", row=1, col=2)
        fig.update_yaxes(title_text="Anomaly Score", row=1, col=2)
        fig.update_xaxes(title_text="Visit Month", row=2, col=1)
        fig.update_yaxes(title_text="MMSE Score", row=2, col=1)
        fig.update_xaxes(title_text="Visit Month", row=2, col=2)
        fig.update_yaxes(title_text="FAQ Score", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def generate_summary_report(self, results: Dict, save_path: Optional[str] = None) -> str:
        """
        Generate a text summary report of anomaly detection results.
        """
        def _format_float(value) -> str:
            if value is None:
                return "N/A"
            if isinstance(value, (int, float, np.floating, np.integer)):
                return f"{float(value):.3f}"
            try:
                return f"{float(value):.3f}"
            except (TypeError, ValueError):
                return "N/A"

        report = []
        report.append("="*60)
        report.append("ANOMALY DETECTION SUMMARY REPORT")
        report.append("="*60)
        
        # Dataset summary
        if 'dataset_info' in results:
            info = results['dataset_info']
            report.append(f"\nDataset Information:")
            report.append(f"  Total patients: {info.get('total_patients', 'N/A')}")
            report.append(f"  Valid patients: {info.get('valid_patients', 'N/A')}")
            report.append(f"  Feature dimensions: {info.get('feature_dims', 'N/A')}")
        
        # Anomaly detection results
        if 'anomaly_summary' in results:
            summary = results['anomaly_summary']
            report.append(f"\nAnomaly Detection Results:")
            report.append(f"  Isolation Forest anomalies: {summary.get('iso_anomaly_count', 'N/A')}")
            report.append(f"  DBSCAN noise points: {summary.get('dbscan_noise_count', 'N/A')}")
            report.append(f"  Consensus anomalies: {summary.get('consensus_count', 'N/A')}")
            report.append(f"  Method agreement rate: {_format_float(summary.get('agreement_rate'))}")
        
        # Top anomalies
        if 'top_anomalies' in results:
            top = results['top_anomalies']
            report.append(f"\nTop 10 Most Anomalous Patients:")
            for i, (pid, score) in enumerate(top[:10]):
                report.append(f"  {i+1:2d}. Patient {pid}: {score:.3f}")
        
        # Cross-validation results
        if 'cross_validation' in results:
            cv = results['cross_validation']
            report.append(f"\nCross-Validation Results:")
            report.append(f"  Score correlation (mean): {_format_float(cv.get('score_correlation_mean'))}")
            report.append(f"  Score correlation (std): {_format_float(cv.get('score_correlation_std'))}")
            report.append(f"  Common patients: {cv.get('common_patients', 'N/A')}")
        
        report.append("\n" + "="*60)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text


def main():
    """Test visualization components."""
    print("Visualization module created successfully!")
    print("Integration with main pipeline required for full testing.")


if __name__ == "__main__":
    main()