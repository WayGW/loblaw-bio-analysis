"""
Data loading and ETL pipeline for Loblaw Bio clinical trial data.

This module handles:
- Reading CSV data
- Data validation and cleaning
- Loading into SQLite database
- Quality checks
"""

import pandas as pd
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any


class DataLoader:
    """Handles ETL pipeline for clinical trial data."""
    
    # Required columns in CSV
    REQUIRED_COLUMNS = [
         'project', 'subject', 'condition', 'age', 'sex', 'treatment',
    'response', 'sample', 'sample_type', 'time_from_treatment_start', 
    'b_cell', 'cd8_t_cell', 'cd4_t_cell', 'nk_cell', 'monocyte'
    ]
    
    # Cell population columns
    CELL_POPULATIONS = ['b_cell', 'cd8_t_cell', 'cd4_t_cell', 'nk_cell', 'monocyte']
    
    def __init__(self, db_connection: sqlite3.Connection):
        """
        Initialize data loader.
        
        Args:
            db_connection: Active SQLite connection
        """
        self.conn = db_connection
    
    def load_csv_to_database(self, csv_path: str) -> Dict[str, Any]:
        """
        Complete ETL pipeline: Read CSV, validate, clean, and load.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Summary dictionary with loading statistics
        """
        print(f"\n{'='*60}")
        print("DATA LOADING PIPELINE")
        print(f"{'='*60}\n")
        
        # Step 1: Read CSV
        print(f"1. Reading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"   ✓ Loaded {len(df)} rows")
        
        # Step 2: Validate schema
        print("\n2. Validating schema...")
        self._validate_required_columns(df)
        print("   ✓ All required columns present")
        
        # Step 3: Clean data
        print("\n3. Cleaning data...")
        df = self._clean_response_column(df)
        df = self._standardize_values(df)
        print("   ✓ Data cleaning complete")
        
        # Step 4: Load to database
        print("\n4. Loading to database...")
        self._load_samples_table(df)
        self._load_cell_counts_table(df)
        print("   ✓ Data loaded successfully")
        
        # Step 5: Quality checks
        print("\n5. Running quality checks...")
        self._run_quality_checks()
        print("   ✓ All quality checks passed")
        
        # Step 6: Generate summary
        summary = self._generate_load_summary(df)
        
        print(f"\n{'='*60}")
        print("LOADING COMPLETE")
        print(f"{'='*60}\n")
        
        return summary
    
    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        """Check that all required columns are present."""
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def _clean_response_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle NA values in response column.
        
        Logic:
        - If treatment == "none" and response is NA → "not_applicable"
        - If treatment != "none" and response is NA → Raise error
        """
        na_count = df['response'].isna().sum()
        
        if na_count > 0:
            # Fill NA for untreated samples
            df['response'] = df.apply(
                lambda row: 'not_applicable' 
                if pd.isna(row['response']) and row['treatment'] == 'none' 
                else row['response'],
                axis=1
            )
            
            # Check for unexpected NAs
            remaining_na = df['response'].isna().sum()
            if remaining_na > 0:
                problematic = df[df['response'].isna()]['sample'].tolist()
                raise ValueError(
                    f"Found {remaining_na} unexpected NA values in 'response' "
                    f"for treated patients: {problematic}"
                )
            
            print(f"   ✓ Cleaned {na_count} NA values in 'response' column")
        
        return df
    
    
    def _standardize_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize categorical values."""
        # Lowercase treatment names
        df['treatment'] = df['treatment'].str.lower()
        
        # Uppercase gender
        df['sex'] = df['sex'].str.upper()
        
        # Lowercase condition/indication
        df['condition'] = df['condition'].str.lower()
        
        # Lowercase response
        df['response'] = df['response'].str.lower()
        
        return df
    
    def _fix_column_name(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix the time_from_b_cell column name to time_from_treatment_start."""
        if 'time_from_b_cell' in df.columns:
            df = df.rename(columns={'time_from_b_cell': 'time_from_treatment_start'})
        return df
    
    def _load_samples_table(self, df: pd.DataFrame) -> None:
        """Load data into samples table."""
        cursor = self.conn.cursor()
        
        # Prepare data for samples table
        samples_data = df[[
            'sample', 'project', 'subject', 'condition', 'age', 'sex',
            'treatment', 'response', 'sample_type', 'time_from_treatment_start'
        ]].drop_duplicates(subset=['sample'])
        
        # Insert into samples table
        for _, row in samples_data.iterrows():
            cursor.execute("""
                INSERT OR REPLACE INTO samples 
                (sample_id, project, subject, indication, age, gender, 
                 treatment, response, sample_type, time_from_treatment_start)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['sample'],
                row['project'],
                row['subject'],
                row['condition'],
                row['age'],
                row['sex'],
                row['treatment'],
                row['response'],
                row['sample_type'],
                row['time_from_treatment_start']
            ))
        
        self.conn.commit()
        print(f"   ✓ Loaded {len(samples_data)} samples")
    
    def _load_cell_counts_table(self, df: pd.DataFrame) -> None:
        """Load cell count data into cell_counts table."""
        cursor = self.conn.cursor()
        
        # Get population IDs
        cursor.execute("SELECT population_id, population_name FROM cell_populations")
        pop_map = {name: pid for pid, name in cursor.fetchall()}
        
        # Insert cell counts
        count_inserted = 0
        for _, row in df.iterrows():
            sample_id = row['sample']
            
            for pop_name in self.CELL_POPULATIONS:
                pop_id = pop_map[pop_name]
                count = row[pop_name]
                
                cursor.execute("""
                    INSERT OR REPLACE INTO cell_counts 
                    (sample_id, population_id, count)
                    VALUES (?, ?, ?)
                """, (sample_id, pop_id, int(count)))
                
                count_inserted += 1
        
        self.conn.commit()
        print(f"   ✓ Loaded {count_inserted} cell count measurements")
    
    def _run_quality_checks(self) -> None:
        """Run post-load validation queries."""
        cursor = self.conn.cursor()
        
        # Check 1: No negative cell counts
        cursor.execute("SELECT COUNT(*) FROM cell_counts WHERE count < 0")
        if cursor.fetchone()[0] > 0:
            raise ValueError("Quality check failed: Found negative cell counts")
        print("   ✓ No negative cell counts")
        
        # Check 2: All samples have metadata
        cursor.execute("""
            SELECT COUNT(DISTINCT cc.sample_id) 
            FROM cell_counts cc 
            LEFT JOIN samples s ON cc.sample_id = s.sample_id 
            WHERE s.sample_id IS NULL
        """)
        if cursor.fetchone()[0] > 0:
            raise ValueError("Quality check failed: Cell counts without sample metadata")
        print("   ✓ All cell counts have sample metadata")
        
        # Check 3: All samples have 5 populations
        cursor.execute("""
            SELECT sample_id, COUNT(*) as pop_count 
            FROM cell_counts 
            GROUP BY sample_id 
            HAVING pop_count != 5
        """)
        incomplete = cursor.fetchall()
        if incomplete:
            raise ValueError(f"Quality check failed: Samples with incomplete populations: {incomplete}")
        print("   ✓ All samples have 5 cell populations")
    
    def _generate_load_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary report of data loading."""
        cursor = self.conn.cursor()
        
        # Get counts from database
        cursor.execute("SELECT COUNT(*) FROM samples")
        db_samples = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT subject) FROM samples")
        db_subjects = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM cell_counts")
        db_measurements = cursor.fetchone()[0]
        
        summary = {
            'csv_rows': len(df),
            'db_samples': db_samples,
            'db_subjects': db_subjects,
            'db_measurements': db_measurements,
            'indications': df['condition'].value_counts().to_dict(),
            'treatments': df['treatment'].value_counts().to_dict(),
            'response_distribution': df['response'].value_counts().to_dict(),
            'sample_types': df['sample_type'].value_counts().to_dict()
        }
        
        # Print summary
        print("\nLOAD SUMMARY:")
        print(f"  Total CSV rows: {summary['csv_rows']}")
        print(f"  Samples in DB: {summary['db_samples']}")
        print(f"  Unique subjects: {summary['db_subjects']}")
        print(f"  Total measurements: {summary['db_measurements']}")
        print(f"\n  Indications: {summary['indications']}")
        print(f"  Treatments: {summary['treatments']}")
        print(f"  Response: {summary['response_distribution']}")
        print(f"  Sample types: {summary['sample_types']}")
        
        return summary


if __name__ == "__main__":
    # Test data loading
    from schema import create_database
    
    print("Testing data loader...")
    
    # Create test database
    populations = ['b_cell', 'cd8_t_cell', 'cd4_t_cell', 'nk_cell', 'monocyte']
    conn = create_database("data/processed/test.db", populations)
    
    # Load data
    loader = DataLoader(conn)
    summary = loader.load_csv_to_database("data/raw/cell-count.csv")
    
    conn.close()
    print("\n✓ Loader test complete!")