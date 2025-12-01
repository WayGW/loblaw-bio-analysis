"""
Part 4: Data subset analysis and cohort exploration.

Filter and summarize specific subsets of clinical trial data.
"""

import pandas as pd
import sqlite3
from typing import Dict, Optional


class CohortFilter:
    """Filter and analyze specific patient cohorts."""
    
    def __init__(self, db_connection: sqlite3.Connection):
        """
        Initialize cohort filter.
        
        Args:
            db_connection: Active SQLite connection
        """
        self.conn = db_connection
    
    def get_baseline_samples(
        self,
        indication: str = 'melanoma',
        treatment: str = 'miraclib',
        sample_type: str = 'PBMC',
        baseline_time: int = 0
    ) -> pd.DataFrame:
        """
        Get baseline samples (time_from_treatment_start = 0).
        
        Args:
            indication: Disease indication
            treatment: Treatment name
            sample_type: Sample type
            baseline_time: Timepoint (default: 0 for baseline)
            
        Returns:
            DataFrame with sample metadata
        """
        query = """
        SELECT 
            sample_id,
            project,
            subject,
            indication,
            age,
            gender,
            treatment,
            response,
            sample_type,
            time_from_treatment_start
        FROM samples
        WHERE 
            indication = ?
            AND treatment = ?
            AND sample_type = ?
            AND time_from_treatment_start = ?
        ORDER BY project, subject
        """
        
        df = pd.read_sql_query(
            query,
            self.conn,
            params=(indication, treatment, sample_type, baseline_time)
        )
        
        return df
    
    def summarize_cohort(self, df: pd.DataFrame) -> Dict:
        """
        Summarize cohort demographics and characteristics.
        
        Args:
            df: DataFrame from get_baseline_samples()
            
        Returns:
            Dictionary with summary statistics
        """
        if len(df) == 0:
            return {
                'total_samples': 0,
                'message': 'No samples found with specified filters'
            }
        
        summary = {
            'total_samples': len(df),
            'unique_subjects': df['subject'].nunique(),
            'samples_per_project': df['project'].value_counts().to_dict(),
            'responders': (df['response'] == 'yes').sum(),
            'non_responders': (df['response'] == 'no').sum(),
            'not_applicable': (df['response'] == 'not_applicable').sum(),
            'males': (df['gender'] == 'M').sum(),
            'females': (df['gender'] == 'F').sum(),
            'age_mean': df['age'].mean(),
            'age_median': df['age'].median(),
            'age_range': (df['age'].min(), df['age'].max())
        }
        
        return summary
    
    def print_cohort_summary(self, summary: Dict) -> None:
        """
        Print formatted cohort summary.
        
        Args:
            summary: Dictionary from summarize_cohort()
        """
        if summary.get('message'):
            print(f"\n{summary['message']}\n")
            return
        
        print("\n" + "="*60)
        print("COHORT SUMMARY")
        print("="*60)
        
        print(f"\nTotal samples: {summary['total_samples']}")
        print(f"Unique subjects: {summary['unique_subjects']}")
        
        print(f"\nSamples per project:")
        for project, count in summary['samples_per_project'].items():
            print(f"  {project}: {count}")
        
        print(f"\nResponse distribution:")
        print(f"  Responders: {summary['responders']}")
        print(f"  Non-responders: {summary['non_responders']}")
        if summary['not_applicable'] > 0:
            print(f"  Not applicable: {summary['not_applicable']}")
        
        print(f"\nGender distribution:")
        print(f"  Males: {summary['males']}")
        print(f"  Females: {summary['females']}")
        
        print(f"\nAge statistics:")
        print(f"  Mean: {summary['age_mean']:.1f} years")
        print(f"  Median: {summary['age_median']:.1f} years")
        print(f"  Range: {summary['age_range'][0]}-{summary['age_range'][1]} years")
        
        print("="*60 + "\n")
    
    def filter_by_criteria(
        self,
        indication: Optional[str] = None,
        treatment: Optional[str] = None,
        sample_type: Optional[str] = None,
        response: Optional[str] = None,
        gender: Optional[str] = None,
        min_age: Optional[int] = None,
        max_age: Optional[int] = None,
        timepoint: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Flexible filtering with multiple criteria.
        
        Args:
            indication: Filter by indication
            treatment: Filter by treatment
            sample_type: Filter by sample type
            response: Filter by response (yes/no/not_applicable)
            gender: Filter by gender (M/F)
            min_age: Minimum age
            max_age: Maximum age
            timepoint: Time from treatment start
            
        Returns:
            Filtered DataFrame
        """
        query = "SELECT * FROM samples WHERE 1=1"
        params = []
        
        if indication:
            query += " AND indication = ?"
            params.append(indication)
        if treatment:
            query += " AND treatment = ?"
            params.append(treatment)
        if sample_type:
            query += " AND sample_type = ?"
            params.append(sample_type)
        if response:
            query += " AND response = ?"
            params.append(response)
        if gender:
            query += " AND gender = ?"
            params.append(gender)
        if min_age:
            query += " AND age >= ?"
            params.append(min_age)
        if max_age:
            query += " AND age <= ?"
            params.append(max_age)
        if timepoint is not None:
            query += " AND time_from_treatment_start = ?"
            params.append(timepoint)
        
        df = pd.read_sql_query(query, self.conn, params=params)
        
        return df


if __name__ == "__main__":
    # Test cohort filtering
    import sqlite3
    
    print("Testing cohort filter...")
    
    conn = sqlite3.connect("data/processed/loblaw_trial.db")
    cohort = CohortFilter(conn)
    
    # Test 1: Get baseline melanoma samples
    print("\n1. Filtering baseline melanoma + miraclib + PBMC samples...")
    baseline_df = cohort.get_baseline_samples()
    print(f"   ✓ Found {len(baseline_df)} samples")
    
    # Test 2: Summarize cohort
    print("\n2. Summarizing cohort...")
    summary = cohort.summarize_cohort(baseline_df)
    cohort.print_cohort_summary(summary)
    
    # Test 3: Flexible filtering
    print("3. Testing flexible filtering...")
    filtered = cohort.filter_by_criteria(
        indication='melanoma',
        treatment='miraclib',
        response='yes',
        gender='F',
        min_age=50
    )
    print(f"   ✓ Found {len(filtered)} female responders age 50+ with melanoma")
    
    conn.close()
    print("✓ Cohort filter test complete!")