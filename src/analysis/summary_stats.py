"""
Part 2: Summary statistics and frequency calculations.

Calculate relative frequencies of cell populations for each sample.
"""

import pandas as pd
import sqlite3
from typing import Optional


class FrequencyAnalyzer:
    """Calculate cell population frequencies."""
    
    def __init__(self, db_connection: sqlite3.Connection):
        """
        Initialize frequency analyzer.
        
        Args:
            db_connection: Active SQLite connection
        """
        self.conn = db_connection
    
    def calculate_frequency_table(
        self,
        indication: Optional[str] = None,
        treatment: Optional[str] = None,
        sample_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate relative frequency of each cell population.
        
        For each sample, computes:
        - Total cell count (sum of all 5 populations)
        - Relative frequency (percentage) of each population
        
        Args:
            indication: Filter by indication (e.g., 'melanoma')
            treatment: Filter by treatment (e.g., 'miraclib')
            sample_type: Filter by sample type (e.g., 'PBMC')
            
        Returns:
            DataFrame with columns:
            - sample: sample_id
            - total_count: total cells in sample
            - population: cell population name
            - count: raw count
            - percentage: relative frequency (%)
        """
        # Build WHERE clause based on filters
        where_clauses = []
        if indication:
            where_clauses.append(f"s.indication = '{indication}'")
        if treatment:
            where_clauses.append(f"s.treatment = '{treatment}'")
        if sample_type:
            where_clauses.append(f"s.sample_type = '{sample_type}'")
        
        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        query = f"""
        WITH total_counts AS (
            SELECT 
                sample_id, 
                SUM(count) as total_count
            FROM cell_counts
            GROUP BY sample_id
        )
        SELECT 
            cc.sample_id as sample,
            tc.total_count,
            cp.population_name as population,
            cc.count,
            ROUND((cc.count * 100.0 / tc.total_count), 2) as percentage
        FROM cell_counts cc
        JOIN total_counts tc ON cc.sample_id = tc.sample_id
        JOIN cell_populations cp ON cc.population_id = cp.population_id
        JOIN samples s ON cc.sample_id = s.sample_id
        {where_sql}
        ORDER BY cc.sample_id, cp.population_name
        """
        
        df = pd.read_sql_query(query, self.conn)
        
        return df
    
    def get_summary_statistics(
        self,
        indication: Optional[str] = None,
        treatment: Optional[str] = None,
        sample_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get summary statistics for cell populations.
        
        Returns mean, median, std, min, max for each population's percentage.
        
        Args:
            indication: Filter by indication
            treatment: Filter by treatment
            sample_type: Filter by sample type
            
        Returns:
            DataFrame with summary statistics by population
        """
        freq_df = self.calculate_frequency_table(indication, treatment, sample_type)
        
        summary = freq_df.groupby('population')['percentage'].agg([
            ('mean', 'mean'),
            ('median', 'median'),
            ('std', 'std'),
            ('min', 'min'),
            ('max', 'max'),
            ('count', 'count')
        ]).round(2)
        
        return summary
    
    def compare_populations(
        self,
        indication: Optional[str] = None,
        treatment: Optional[str] = None,
        sample_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compare average frequencies across populations.
        
        Returns a simple comparison table showing which populations
        are most/least abundant.
        
        Args:
            indication: Filter by indication
            treatment: Filter by treatment
            sample_type: Filter by sample type
            
        Returns:
            DataFrame sorted by mean percentage (descending)
        """
        summary = self.get_summary_statistics(indication, treatment, sample_type)
        summary = summary.sort_values('mean', ascending=False)
        
        return summary


def export_frequency_table(
    db_connection: sqlite3.Connection,
    output_path: str,
    **filters
) -> None:
    """
    Export frequency table to CSV.
    
    Args:
        db_connection: SQLite connection
        output_path: Path to save CSV file
        **filters: Optional filters (indication, treatment, sample_type)
    """
    analyzer = FrequencyAnalyzer(db_connection)
    df = analyzer.calculate_frequency_table(**filters)
    df.to_csv(output_path, index=False)
    print(f"✓ Frequency table exported to: {output_path}")


if __name__ == "__main__":
    # Test frequency calculations
    import sqlite3
    
    print("Testing frequency analyzer...")
    
    conn = sqlite3.connect("data/processed/loblaw_trial.db")
    analyzer = FrequencyAnalyzer(conn)
    
    # Test 1: All samples
    print("\n1. Calculating frequencies for all samples...")
    freq_df = analyzer.calculate_frequency_table()
    print(f"   ✓ Generated {len(freq_df)} rows")
    print("\nFirst 10 rows:")
    print(freq_df.head(10))
    
    # Test 2: Summary statistics
    print("\n2. Summary statistics for all samples:")
    summary = analyzer.get_summary_statistics()
    print(summary)
    
    # Test 3: Filtered - melanoma + miraclib + PBMC
    print("\n3. Frequencies for melanoma + miraclib + PBMC:")
    filtered = analyzer.calculate_frequency_table(
        indication='melanoma',
        treatment='miraclib',
        sample_type='PBMC'
    )
    print(f"   ✓ {len(filtered)} rows")
    print("\nSummary:")
    print(analyzer.get_summary_statistics(
        indication='melanoma',
        treatment='miraclib',
        sample_type='PBMC'
    ))
    
    conn.close()
    print("\n✓ Frequency analyzer test complete!")