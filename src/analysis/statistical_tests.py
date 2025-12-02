"""
Part 3: Statistical analysis comparing responders vs non-responders.

Performs statistical tests to identify cell populations with
significant differences between treatment responders and non-responders.
"""

import pandas as pd
import numpy as np
import sqlite3
from scipy import stats
from typing import Tuple, Dict, Optional


class ResponseAnalyzer:
    """Analyze response differences in cell populations."""
    
    def __init__(self, db_connection: sqlite3.Connection):
        """
        Initialize response analyzer.
        
        Args:
            db_connection: Active SQLite connection
        """
        self.conn = db_connection
    
    def get_response_data(
        self,
        indication: str = 'melanoma',
        treatment: str = 'miraclib',
        sample_type: str = 'PBMC'
    ) -> pd.DataFrame:
        """
        Get cell population frequencies for responders vs non-responders.
        
        Args:
            indication: Disease indication (default: 'melanoma')
            treatment: Treatment name (default: 'miraclib')
            sample_type: Sample type (default: 'PBMC')
            
        Returns:
            DataFrame with columns:
            - sample_id
            - response (yes/no)
            - population
            - percentage
        """
        query = """
        WITH total_counts AS (
            SELECT sample_id, SUM(count) as total_count
            FROM cell_counts
            GROUP BY sample_id
        )
        SELECT 
            s.sample_id,
            s.response,
            cp.population_name as population,
            cc.count,
            (cc.count * 100.0 / tc.total_count) as percentage
        FROM cell_counts cc
        JOIN samples s ON cc.sample_id = s.sample_id
        JOIN cell_populations cp ON cc.population_id = cp.population_id
        JOIN total_counts tc ON cc.sample_id = tc.sample_id
        WHERE 
            s.indication = ?
            AND s.treatment = ?
            AND s.sample_type = ?
            AND s.response IN ('yes', 'no')
        ORDER BY s.sample_id, cp.population_name
        """
        
        df = pd.read_sql_query(
            query, 
            self.conn, 
            params=(indication, treatment, sample_type)
        )
        
        return df
    
    def test_normality(self, data: np.ndarray) -> Tuple[float, bool]:
        """
        Test if data is normally distributed (Shapiro-Wilk test).
        
        Args:
            data: Array of values
            
        Returns:
            (p_value, is_normal) tuple
        """
        if len(data) < 3:
            return (np.nan, False)
        
        stat, p_value = stats.shapiro(data)
        is_normal = p_value > 0.05
        
        return (p_value, is_normal)
    
    def compare_groups(
        self,
        responders: np.ndarray,
        non_responders: np.ndarray
    ) -> Dict:
        """
        Compare two groups using appropriate statistical test.
        
        Performs:
        1. Normality test (Shapiro-Wilk)
        2. Mann-Whitney U test (non-parametric, safer default)
        3. Effect size (rank-biserial correlation for Mann-Whitney)
        
        Args:
            responders: Data from responders
            non_responders: Data from non-responders
            
        Returns:
            Dictionary with test results
        """
        # Test normality
        _, resp_normal = self.test_normality(responders)
        _, non_resp_normal = self.test_normality(non_responders)
        
        # Perform Mann-Whitney U test (non-parametric)
        statistic, p_value = stats.mannwhitneyu(
            responders, 
            non_responders, 
            alternative='two-sided'
        )
        
        # Calculate effect size (rank-biserial correlation)
        n1, n2 = len(responders), len(non_responders)
        effect_size = 1 - (2*statistic) / (n1 * n2)
        
        return {
            'test_used': 'mann_whitney_u',
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'responders_normal': resp_normal,
            'non_responders_normal': non_resp_normal
        }
    
    def analyze_all_populations(self, indication='melanoma', treatment='miraclib', 
                               sample_type='PBMC', alpha=0.05):
        """
        Analyze all cell populations comparing responders vs non-responders.
        
        Returns:
            DataFrame with statistical test results for each population
        """
        populations = ['b_cell', 'cd8_t_cell', 'cd4_t_cell', 'nk_cell', 'monocyte']
        
        # Get data for all populations
        data = self.get_response_data(indication, treatment, sample_type)
        
        results = []
        
        # Bonferroni correction
        n_tests = len(populations)
        alpha_corrected = alpha / n_tests
        
        for pop in populations:
            pop_data = data[data['population'] == pop]
            
            if len(pop_data) == 0:
                continue
            
            # Split by response
            responders = pop_data[pop_data['response'] == 'yes']['percentage'].values
            non_responders = pop_data[pop_data['response'] == 'no']['percentage'].values
            
            # Perform Mann-Whitney U test
            statistic, p_value = stats.mannwhitneyu(responders, non_responders, alternative='two-sided')
            
            # Calculate effect size (rank-biserial correlation)
            n1, n2 = len(responders), len(non_responders)
            effect_size = 1 - (2 * statistic) / (n1 * n2)
            
            # Bonferroni corrected p-value
            p_value_corrected = min(p_value * n_tests, 1.0)  # Cap at 1.0
            
            # Determine significance with corrected alpha
            significant = p_value < alpha_corrected
            
            results.append({
                'population': pop,
                'n_responders': int(n1),
                'n_non_responders': int(n2),
                'median_responders': float(np.median(responders)),
                'median_non_responders': float(np.median(non_responders)),
                'mean_responders': float(np.mean(responders)),
                'mean_non_responders': float(np.mean(non_responders)),
                'std_responders': float(np.std(responders)),
                'std_non_responders': float(np.std(non_responders)),
                'p_value': float(p_value),
                'p_value_corrected': float(p_value_corrected),
                'significant': bool(significant),
                'effect_size': float(effect_size)
            })
        
        return pd.DataFrame(results)
    
    def print_results_summary(self, results_df: pd.DataFrame) -> None:
        """
        Print formatted summary of statistical results.
        
        Args:
            results_df: DataFrame from analyze_all_populations()
        """
        print("\n" + "="*80)
        print("STATISTICAL ANALYSIS: RESPONDERS VS NON-RESPONDERS")
        print("="*80)
        
        print(f"\nSample sizes:")
        print(f"  Responders: {results_df['n_responders'].iloc[0]}")
        print(f"  Non-responders: {results_df['n_non_responders'].iloc[0]}")
        
        print(f"\nTest used: Mann-Whitney U Test")
        print(f"Multiple testing correction: Bonferroni")
        print(f"Significance threshold: p < {0.05/len(results_df):.4f}")
        
        print("\nResults by population:")
        print("-" * 80)
        
        for _, row in results_df.iterrows():
            sig_marker = "***" if row['significant'] else "ns"
            
            print(f"\n{row['population'].upper()}")
            print(f"  Median:  Responders={row['median_responders']:.2f}%  |  "
                  f"Non-responders={row['median_non_responders']:.2f}%")
            print(f"  Mean:    Responders={row['mean_responders']:.2f}%  |  "
                  f"Non-responders={row['mean_non_responders']:.2f}%")
            print(f"  p-value: {row['p_value']:.4f}  {sig_marker}")
            print(f"  Effect size: {row['effect_size']:.3f}")
        
        print("\n" + "="*80)
        
        # Summary of significant populations
        sig_pops = results_df[results_df['significant']]['population'].tolist()
        if sig_pops:
            print(f"\nSignificant populations (p < 0.01): {', '.join(sig_pops)}")
        else:
            print("\nNo populations showed significant differences after correction.")
        
        print("="*80 + "\n")


if __name__ == "__main__":
    # Test statistical analysis
    import sqlite3
    
    print("Testing response analyzer...")
    
    conn = sqlite3.connect("data/processed/loblaw_trial.db")
    analyzer = ResponseAnalyzer(conn)
    
    # Test 1: Get response data
    print("\n1. Fetching response data...")
    data = analyzer.get_response_data()
    print(f"   ✓ Retrieved {len(data)} rows")
    print(f"   Responders: {(data['response'] == 'yes').sum()}")
    print(f"   Non-responders: {(data['response'] == 'no').sum()}")
    
    # Test 2: Run full analysis
    print("\n2. Running statistical analysis...")
    results = analyzer.analyze_all_populations()
    
    # Test 3: Print results
    analyzer.print_results_summary(results)
    
    conn.close()
    print("✓ Response analyzer test complete!")