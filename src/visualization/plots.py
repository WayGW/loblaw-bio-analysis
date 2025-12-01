"""
Visualization module for clinical trial analysis.

Creates static and interactive plots for data exploration.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class TrialVisualizer:
    """Create visualizations for clinical trial data."""
    
    def __init__(self, style: str = 'whitegrid', palette: str = 'colorblind'):
        """
        Initialize visualizer with style settings.
        
        Args:
            style: Seaborn style (default: 'whitegrid')
            palette: Color palette (default: 'colorblind')
        """
        sns.set_style(style)
        self.palette = palette
        
    def create_response_boxplot(
        self,
        data: pd.DataFrame,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6),
        dpi: int = 300
    ) -> plt.Figure:
        """
        Create boxplot comparing responders vs non-responders.
        
        Args:
            data: DataFrame with columns ['population', 'response', 'percentage']
            save_path: Optional path to save figure
            figsize: Figure size (width, height)
            dpi: Resolution for saved figure
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create boxplot
        sns.boxplot(
            data=data,
            x='population',
            y='percentage',
            hue='response',
            palette=self.palette,
            ax=ax
        )
        
        # Customize
        ax.set_xlabel('Cell Population', fontsize=12, fontweight='bold')
        ax.set_ylabel('Relative Frequency (%)', fontsize=12, fontweight='bold')
        ax.set_title(
            'Cell Population Frequencies: Responders vs Non-Responders',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        # Rotate x labels for readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            ['Non-Responders', 'Responders'],
            title='Response',
            loc='upper right'
        )
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"✓ Figure saved: {save_path}")
        
        return fig
    
    def create_response_boxplot_with_stats(
        self,
        data: pd.DataFrame,
        stats_results: pd.DataFrame,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6),
        dpi: int = 300
    ) -> plt.Figure:
        """
        Create boxplot with p-values annotated.
        
        Args:
            data: DataFrame with frequency data
            stats_results: DataFrame with statistical test results
            save_path: Optional path to save figure
            figsize: Figure size
            dpi: Resolution
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create boxplot
        sns.boxplot(
            data=data,
            x='population',
            y='percentage',
            hue='response',
            palette=self.palette,
            ax=ax
        )
        
        # Add p-values
        populations = stats_results['population'].values
        p_values = stats_results['p_value'].values
        
        # Get y-axis limits for annotation positioning
        y_max = data['percentage'].max()
        y_range = data['percentage'].max() - data['percentage'].min()
        
        for i, (pop, p_val) in enumerate(zip(populations, p_values)):
            # Format p-value
            if p_val < 0.001:
                p_text = 'p < 0.001***'
            elif p_val < 0.01:
                p_text = f'p = {p_val:.3f}**'
            elif p_val < 0.05:
                p_text = f'p = {p_val:.3f}*'
            else:
                p_text = f'p = {p_val:.3f}'
            
            # Add text annotation
            ax.text(
                i, 
                y_max + 0.02 * y_range,
                p_text,
                ha='center',
                va='bottom',
                fontsize=8,
                fontweight='bold' if p_val < 0.05 else 'normal'
            )
        
        # Customize
        ax.set_xlabel('Cell Population', fontsize=12, fontweight='bold')
        ax.set_ylabel('Relative Frequency (%)', fontsize=12, fontweight='bold')
        ax.set_title(
            'Cell Population Frequencies: Responders vs Non-Responders\n(Mann-Whitney U Test with Bonferroni Correction)',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            ['Non-Responders', 'Responders'],
            title='Response',
            loc='upper right'
        )
        
        # Extend y-axis to fit annotations
        ax.set_ylim(top=y_max + 0.1 * y_range)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"✓ Figure saved: {save_path}")
        
        return fig
    
    def create_interactive_boxplot(
        self,
        data: pd.DataFrame,
        stats_results: Optional[pd.DataFrame] = None
    ) -> go.Figure:
        """
        Create interactive Plotly boxplot.
        
        Args:
            data: DataFrame with frequency data
            stats_results: Optional statistical results to show in hover
            
        Returns:
            Plotly figure object
        """
        # Create figure
        fig = px.box(
            data,
            x='population',
            y='percentage',
            color='response',
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={
                'population': 'Cell Population',
                'percentage': 'Relative Frequency (%)',
                'response': 'Response'
            },
            title='Cell Population Frequencies: Responders vs Non-Responders'
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Cell Population',
            yaxis_title='Relative Frequency (%)',
            font=dict(size=12),
            hovermode='closest',
            legend=dict(
                title='Response',
                orientation='v',
                yanchor='top',
                y=1,
                xanchor='right',
                x=1
            )
        )
        
        # Add p-values as annotations if provided
        if stats_results is not None:
            annotations = []
            for i, row in stats_results.iterrows():
                if row['p_value'] < 0.05:
                    sig_marker = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*'
                    annotations.append(
                        dict(
                            x=row['population'],
                            y=data[data['population'] == row['population']]['percentage'].max() * 1.05,
                            text=f"p={row['p_value']:.3f}{sig_marker}",
                            showarrow=False,
                            font=dict(size=10, color='red')
                        )
                    )
            
            if annotations:
                fig.update_layout(annotations=annotations)
        
        return fig
    
    def create_frequency_heatmap(
        self,
        summary_stats: pd.DataFrame,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 300
    ) -> plt.Figure:
        """
        Create heatmap of mean frequencies by population.
        
        Args:
            summary_stats: DataFrame with population statistics
            save_path: Optional path to save figure
            figsize: Figure size
            dpi: Resolution
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data for heatmap
        heatmap_data = summary_stats[['mean']].T
        
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Mean Frequency (%)'},
            ax=ax
        )
        
        ax.set_title('Mean Cell Population Frequencies', fontsize=14, fontweight='bold')
        ax.set_xlabel('Cell Population', fontsize=12)
        ax.set_ylabel('')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"✓ Figure saved: {save_path}")
        
        return fig
    
    def create_demographics_plot(
        self,
        cohort_data: pd.DataFrame,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 5),
        dpi: int = 300
    ) -> plt.Figure:
        """
        Create demographic summary plots.
        
        Args:
            cohort_data: DataFrame with cohort information
            save_path: Optional path to save
            figsize: Figure size
            dpi: Resolution
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot 1: Gender distribution
        gender_counts = cohort_data['gender'].value_counts()
        axes[0].pie(
            gender_counts.values,
            labels=['Male' if x == 'M' else 'Female' for x in gender_counts.index],
            autopct='%1.1f%%',
            colors=sns.color_palette(self.palette, len(gender_counts))
        )
        axes[0].set_title('Gender Distribution', fontweight='bold')
        
        # Plot 2: Response distribution
        response_counts = cohort_data['response'].value_counts()
        axes[1].bar(
            range(len(response_counts)),
            response_counts.values,
            color=sns.color_palette(self.palette, len(response_counts))
        )
        axes[1].set_xticks(range(len(response_counts)))
        axes[1].set_xticklabels(
            ['Responders' if x == 'yes' else 'Non-Responders' for x in response_counts.index],
            rotation=45,
            ha='right'
        )
        axes[1].set_ylabel('Count')
        axes[1].set_title('Response Distribution', fontweight='bold')
        
        # Plot 3: Age distribution
        axes[2].hist(
            cohort_data['age'],
            bins=15,
            color=sns.color_palette(self.palette)[0],
            edgecolor='black'
        )
        axes[2].axvline(
            cohort_data['age'].mean(),
            color='red',
            linestyle='--',
            label=f"Mean: {cohort_data['age'].mean():.1f}"
        )
        axes[2].set_xlabel('Age (years)')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Age Distribution', fontweight='bold')
        axes[2].legend()
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"✓ Figure saved: {save_path}")
        
        return fig


if __name__ == "__main__":
    # Test visualization
    import sqlite3
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from src.analysis.statistical_tests import ResponseAnalyzer
    from src.analysis.filtering import CohortFilter
    
    print("Testing visualizations...")
    
    conn = sqlite3.connect("data/processed/loblaw_trial.db")
    
    # Get data for plotting
    analyzer = ResponseAnalyzer(conn)
    data = analyzer.get_response_data()
    stats_results = analyzer.analyze_all_populations()
    
    # Create visualizer
    viz = TrialVisualizer()
    
    # Test 1: Basic boxplot
    print("\n1. Creating response boxplot...")
    fig1 = viz.create_response_boxplot(data, save_path='outputs/boxplot_basic.png')
    plt.close()
    
    # Test 2: Boxplot with statistics
    print("\n2. Creating boxplot with p-values...")
    fig2 = viz.create_response_boxplot_with_stats(
        data,
        stats_results,
        save_path='outputs/boxplot_stats.png'
    )
    plt.close()
    
    # Test 3: Interactive plot
    print("\n3. Creating interactive boxplot...")
    fig3 = viz.create_interactive_boxplot(data, stats_results)
    fig3.write_html('outputs/boxplot_interactive.html')
    print("✓ Interactive plot saved: outputs/boxplot_interactive.html")
    
    # Test 4: Demographics
    print("\n4. Creating demographics plot...")
    cohort = CohortFilter(conn)
    cohort_data = cohort.get_baseline_samples()
    fig4 = viz.create_demographics_plot(
        cohort_data,
        save_path='outputs/demographics.png'
    )
    plt.close()
    
    conn.close()
    print("\n✓ Visualization tests complete!")
    print("Check the 'outputs/' folder for generated plots.")