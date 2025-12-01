"""
Database schema definitions for Loblaw Bio clinical trial data.

This module creates and initializes the SQLite database with three tables:
1. samples - Patient/sample metadata
2. cell_populations - Reference table for cell types
3. cell_counts - Measurement data
"""

import sqlite3
from pathlib import Path
from typing import List


class DatabaseSchema:
    """Handles database schema creation and initialization."""
    
    def __init__(self, db_path: str):
        """
        Initialize database schema handler.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def create_schema(self) -> sqlite3.Connection:
        """
        Create database schema with all tables.
        
        Returns:
            SQLite connection object
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create samples table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS samples (
                sample_id TEXT PRIMARY KEY,
                project TEXT NOT NULL,
                subject TEXT NOT NULL,
                indication TEXT NOT NULL,
                age INTEGER,
                gender TEXT CHECK(gender IN ('M', 'F', 'other')),
                treatment TEXT NOT NULL,
                response TEXT CHECK(response IN ('yes', 'no', 'not_applicable')),
                sample_type TEXT NOT NULL,
                time_from_treatment_start INTEGER NOT NULL
            )
        """)
        
        # Create cell_populations reference table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cell_populations (
                population_id INTEGER PRIMARY KEY AUTOINCREMENT,
                population_name TEXT UNIQUE NOT NULL
            )
        """)
        
        # Create cell_counts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cell_counts (
                count_id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id TEXT NOT NULL,
                population_id INTEGER NOT NULL,
                count INTEGER NOT NULL CHECK(count >= 0),
                FOREIGN KEY (sample_id) REFERENCES samples(sample_id),
                FOREIGN KEY (population_id) REFERENCES cell_populations(population_id),
                UNIQUE(sample_id, population_id)
            )
        """)
        
        # Create indexes for better query performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_samples_indication 
            ON samples(indication)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_samples_treatment 
            ON samples(treatment)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_samples_response 
            ON samples(response)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cell_counts_sample 
            ON cell_counts(sample_id)
        """)
        
        conn.commit()
        print(f"✓ Database schema created: {self.db_path}")
        
        return conn
    
    def initialize_cell_populations(
        self, 
        conn: sqlite3.Connection,
        populations: List[str]
    ) -> None:
        """
        Populate cell_populations reference table.
        
        Args:
            conn: SQLite connection
            populations: List of cell population names
        """
        cursor = conn.cursor()
        
        for pop_name in populations:
            cursor.execute("""
                INSERT OR IGNORE INTO cell_populations (population_name)
                VALUES (?)
            """, (pop_name,))
        
        conn.commit()
        
        # Verify
        cursor.execute("SELECT COUNT(*) FROM cell_populations")
        count = cursor.fetchone()[0]
        print(f"✓ Initialized {count} cell populations")
    
    def drop_all_tables(self, conn: sqlite3.Connection) -> None:
        """
        Drop all tables (use with caution!).
        
        Args:
            conn: SQLite connection
        """
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS cell_counts")
        cursor.execute("DROP TABLE IF EXISTS cell_populations")
        cursor.execute("DROP TABLE IF EXISTS samples")
        conn.commit()
        print("✓ All tables dropped")
    
    def get_database_info(self, conn: sqlite3.Connection) -> dict:
        """
        Get summary information about the database.
        
        Args:
            conn: SQLite connection
            
        Returns:
            Dictionary with database statistics
        """
        cursor = conn.cursor()
        
        # Count samples
        cursor.execute("SELECT COUNT(*) FROM samples")
        num_samples = cursor.fetchone()[0]
        
        # Count populations
        cursor.execute("SELECT COUNT(*) FROM cell_populations")
        num_populations = cursor.fetchone()[0]
        
        # Count measurements
        cursor.execute("SELECT COUNT(*) FROM cell_counts")
        num_measurements = cursor.fetchone()[0]
        
        # Get unique subjects
        cursor.execute("SELECT COUNT(DISTINCT subject) FROM samples")
        num_subjects = cursor.fetchone()[0]
        
        # Get unique projects
        cursor.execute("SELECT COUNT(DISTINCT project) FROM samples")
        num_projects = cursor.fetchone()[0]
        
        return {
            'database_path': str(self.db_path),
            'num_samples': num_samples,
            'num_subjects': num_subjects,
            'num_projects': num_projects,
            'num_populations': num_populations,
            'num_measurements': num_measurements,
            'expected_measurements': num_samples * num_populations
        }


def create_database(db_path: str, cell_populations: List[str]) -> sqlite3.Connection:
    """
    Convenience function to create and initialize database.
    
    Args:
        db_path: Path to database file
        cell_populations: List of cell population names
        
    Returns:
        SQLite connection object
    """
    schema = DatabaseSchema(db_path)
    conn = schema.create_schema()
    schema.initialize_cell_populations(conn, cell_populations)
    
    return conn


if __name__ == "__main__":
    # Test database creation
    print("Testing database schema creation...")
    
    test_populations = [
        'b_cell',
        'cd8_t_cell',
        'cd4_t_cell',
        'nk_cell',
        'monocyte'
    ]
    
    conn = create_database("data/processed/test.db", test_populations)
    
    schema = DatabaseSchema("data/processed/test.db")
    info = schema.get_database_info(conn)
    
    print("\nDatabase Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    conn.close()
    print("\n✓ Schema test complete!")