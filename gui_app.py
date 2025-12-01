"""
Simple GUI Application for Loblaw Bio Clinical Trial Analysis

A user-friendly interface for running the complete analysis pipeline without using the terminal.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import subprocess
import sys
from pathlib import Path
import yaml
import sqlite3

# Import project modules
from src.database.schema import create_database
from src.database.loader import DataLoader
from src.analysis.summary_stats import FrequencyAnalyzer, export_frequency_table
from src.analysis.statistical_tests import ResponseAnalyzer
from src.analysis.filtering import CohortFilter
from src.visualization.plots import TrialVisualizer
import matplotlib.pyplot as plt


class LoblawBioGUI:
    """Main GUI application class."""
    
    def __init__(self, root):
        """Initialize the GUI."""
        self.root = root
        self.root.title("Loblaw Bio - Clinical Trial Analysis")
        self.root.geometry("700x600")
        self.root.resizable(True, True)
        
        # Variables
        self.data_file_path = tk.StringVar()
        self.status_text = tk.StringVar(value="Ready to start")
        self.config = self.load_config()
        
        # Create UI
        self.create_widgets()
        
        # Check if database exists
        self.check_database_status()
    
    def load_config(self):
        """Load configuration from YAML."""
        try:
            with open('config/config.yaml', 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            messagebox.showerror("Config Error", f"Could not load config: {e}")
            return {}
    
    def create_widgets(self):
        """Create all GUI widgets."""
        
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="🧬 Loblaw Bio - Clinical Trial Analysis",
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, pady=(0, 10), sticky=tk.W)
        
        # Subtitle
        subtitle_label = ttk.Label(
            main_frame,
            text="Analyze immune cell population data from clinical trials",
            font=("Arial", 10)
        )
        subtitle_label.grid(row=1, column=0, pady=(0, 20), sticky=tk.W)
        
        # File selection section
        file_frame = ttk.LabelFrame(main_frame, text="Step 1: Select Data File", padding="10")
        file_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(0, weight=1)
        
        # File path display
        file_entry = ttk.Entry(file_frame, textvariable=self.data_file_path, state="readonly")
        file_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # Browse button
        browse_btn = ttk.Button(
            file_frame,
            text="📁 Browse...",
            command=self.browse_file
        )
        browse_btn.grid(row=0, column=1)
        
        # Analysis section
        analysis_frame = ttk.LabelFrame(main_frame, text="Step 2: Run Analysis", padding="10")
        analysis_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        analysis_frame.columnconfigure(0, weight=1)
        
        # Run button
        self.run_btn = ttk.Button(
            analysis_frame,
            text="▶️ Run Complete Analysis",
            command=self.run_analysis,
            style="Accent.TButton"
        )
        self.run_btn.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Progress bar
        self.progress = ttk.Progressbar(analysis_frame, mode='indeterminate')
        self.progress.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Status label
        status_label = ttk.Label(analysis_frame, textvariable=self.status_text, font=("Arial", 9))
        status_label.grid(row=2, column=0, sticky=tk.W)
        
        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Step 3: View Results", padding="10")
        results_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        
        # View dashboard button
        self.dashboard_btn = ttk.Button(
            results_frame,
            text="📊 View Interactive Dashboard",
            command=self.open_dashboard,
            state="disabled"
        )
        self.dashboard_btn.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Open outputs folder button
        self.outputs_btn = ttk.Button(
            results_frame,
            text="📁 Open Output Files",
            command=self.open_outputs_folder,
            state="disabled"
        )
        self.outputs_btn.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Log output section
        log_frame = ttk.LabelFrame(main_frame, text="Analysis Log", padding="10")
        log_frame.grid(row=5, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(5, weight=1)
        
        # Scrolled text widget for log
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=10,
            wrap=tk.WORD,
            font=("Courier", 9),
            state="disabled"
        )
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Bottom buttons
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=6, column=0, sticky=(tk.W, tk.E))
        bottom_frame.columnconfigure(0, weight=1)
        
        # Help button
        help_btn = ttk.Button(bottom_frame, text="❓ Help", command=self.show_help)
        help_btn.grid(row=0, column=0, sticky=tk.W)
        
        # Exit button
        exit_btn = ttk.Button(bottom_frame, text="❌ Exit", command=self.root.quit)
        exit_btn.grid(row=0, column=1, sticky=tk.E)
    
    def browse_file(self):
        """Open file dialog to select data file."""
        filename = filedialog.askopenfilename(
            title="Select Clinical Trial Data File",
            initialdir=Path("data/raw"),
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            self.data_file_path.set(filename)
            self.log_message(f"Selected file: {Path(filename).name}")
            self.status_text.set("Data file selected")
    
    def check_database_status(self):
        """Check if database exists and has data."""
        db_path = Path(self.config['database']['path'])
        
        if db_path.exists():
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM samples")
                count = cursor.fetchone()[0]
                conn.close()
                
                if count > 0:
                    self.log_message(f"✓ Database found with {count} samples")
                    self.dashboard_btn.config(state="normal")
                    self.outputs_btn.config(state="normal")
            except:
                pass
    
    def log_message(self, message):
        """Add message to log window."""
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")
    
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        # Validate file selection
        if not self.data_file_path.get():
            messagebox.showwarning("No File Selected", "Please select a data file first.")
            return
        
        if not Path(self.data_file_path.get()).exists():
            messagebox.showerror("File Not Found", "The selected file does not exist.")
            return
        
        # Disable buttons during analysis
        self.run_btn.config(state="disabled")
        self.dashboard_btn.config(state="disabled")
        self.outputs_btn.config(state="disabled")
        
        # Start progress bar
        self.progress.start(10)
        
        # Run analysis in separate thread to keep GUI responsive
        thread = threading.Thread(target=self._run_analysis_thread, daemon=True)
        thread.start()
    
    def _run_analysis_thread(self):
        """Run analysis in background thread."""
        try:
            self.status_text.set("Initializing database...")
            self.log_message("\n" + "="*50)
            self.log_message("STARTING ANALYSIS PIPELINE")
            self.log_message("="*50)
            
            # Step 1: Initialize database
            self.log_message("\n1. Initializing database...")
            db_path = self.config['database']['path']
            populations = self.config['cell_populations']
            
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = create_database(db_path, populations)
            self.log_message("   ✓ Database initialized")
            
            # Step 2: Load data
            self.status_text.set("Loading data...")
            self.log_message("\n2. Loading data from CSV...")
            loader = DataLoader(conn)
            summary = loader.load_csv_to_database(self.data_file_path.get())
            self.log_message(f"   ✓ Loaded {summary['db_samples']} samples")
            self.log_message(f"   ✓ {summary['db_measurements']} measurements")
            
            # Step 3: Frequency analysis
            self.status_text.set("Running frequency analysis...")
            self.log_message("\n3. Running frequency analysis...")
            analyzer = FrequencyAnalyzer(conn)
            freq_df = analyzer.calculate_frequency_table()
            output_path = "outputs/frequency_table.csv"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            freq_df.to_csv(output_path, index=False)
            self.log_message(f"   ✓ Saved to {output_path}")
            
            # Step 4: Response analysis
            self.status_text.set("Running response analysis...")
            self.log_message("\n4. Running response analysis...")
            params = self.config['analysis']['part3']
            response_analyzer = ResponseAnalyzer(conn)
            
            data = response_analyzer.get_response_data(
                indication=params['indication'],
                treatment=params['treatment'],
                sample_type=params['sample_type']
            )
            
            results = response_analyzer.analyze_all_populations(
                indication=params['indication'],
                treatment=params['treatment'],
                sample_type=params['sample_type']
            )
            
            results.to_csv("outputs/response_analysis_results.csv", index=False)
            self.log_message("   ✓ Saved to outputs/response_analysis_results.csv")
            
            # Create visualization
            self.log_message("   ✓ Creating boxplot...")
            viz = TrialVisualizer()
            fig = viz.create_response_boxplot_with_stats(
                data,
                results,
                save_path="outputs/response_boxplot.png"
            )
            plt.close()
            self.log_message("   ✓ Saved to outputs/response_boxplot.png")
            
            # Step 5: Cohort analysis
            self.status_text.set("Running cohort analysis...")
            self.log_message("\n5. Running cohort analysis...")
            cohort = CohortFilter(conn)
            baseline_df = cohort.get_baseline_samples(
                indication=params['indication'],
                treatment=params['treatment'],
                sample_type=params['sample_type']
            )
            
            baseline_df.to_csv("outputs/baseline_cohort.csv", index=False)
            self.log_message(f"   ✓ Saved {len(baseline_df)} baseline samples")
            
            conn.close()
            
            # Success!
            self.log_message("\n" + "="*50)
            self.log_message("ANALYSIS COMPLETE!")
            self.log_message("="*50)
            self.log_message("\nGenerated files in outputs/:")
            self.log_message("  - frequency_table.csv")
            self.log_message("  - response_analysis_results.csv")
            self.log_message("  - response_boxplot.png")
            self.log_message("  - baseline_cohort.csv")
            
            # Update UI on main thread
            self.root.after(0, self._analysis_complete_success)
            
        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            self.log_message(f"\n❌ {error_msg}")
            self.root.after(0, lambda: self._analysis_complete_error(error_msg))
    
    def _analysis_complete_success(self):
        """Update UI after successful analysis."""
        self.progress.stop()
        self.status_text.set("Analysis complete! Ready to view results.")
        self.run_btn.config(state="normal")
        self.dashboard_btn.config(state="normal")
        self.outputs_btn.config(state="normal")
        
        messagebox.showinfo(
            "Analysis Complete",
            "Analysis completed successfully!\n\n"
            "You can now:\n"
            "• View the interactive dashboard\n"
            "• Check the output files in the outputs/ folder"
        )
    
    def _analysis_complete_error(self, error_msg):
        """Update UI after analysis error."""
        self.progress.stop()
        self.status_text.set("Analysis failed - see log for details")
        self.run_btn.config(state="normal")
        
        messagebox.showerror("Analysis Error", error_msg)
    
    def open_dashboard(self):
        """Launch the Streamlit dashboard."""
        self.log_message("\nLaunching dashboard...")
        self.status_text.set("Opening dashboard in browser...")
        
        try:
            # Launch Streamlit in a separate process
            subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py"],
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
            )
            
            self.log_message("✓ Dashboard opened at http://localhost:8501")
            self.status_text.set("Dashboard running - check your browser")
            
            messagebox.showinfo(
                "Dashboard Launched",
                "The interactive dashboard is now running!\n\n"
                "URL: http://localhost:8501\n\n"
                "If it didn't open automatically, copy the URL above into your browser.\n\n"
                "Close the dashboard terminal window when finished."
            )
        except Exception as e:
            self.log_message(f"❌ Failed to launch dashboard: {e}")
            messagebox.showerror("Dashboard Error", f"Could not launch dashboard: {e}")
    
    def open_outputs_folder(self):
        """Open the outputs folder in file explorer."""
        outputs_path = Path("outputs").absolute()
        
        if not outputs_path.exists():
            messagebox.showwarning("No Outputs", "No output files found. Run analysis first.")
            return
        
        try:
            if sys.platform == 'win32':
                subprocess.run(['explorer', str(outputs_path)])
            elif sys.platform == 'darwin':  # macOS
                subprocess.run(['open', str(outputs_path)])
            else:  # Linux
                subprocess.run(['xdg-open', str(outputs_path)])
            
            self.log_message(f"✓ Opened outputs folder: {outputs_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder: {e}")
    
    def show_help(self):
        """Show help dialog."""
        help_text = """
Loblaw Bio - Clinical Trial Analysis Tool

HOW TO USE:

1. SELECT DATA FILE
   Click 'Browse' and select your clinical trial CSV file.
   The file should contain cell count data with required columns.

2. RUN ANALYSIS
   Click 'Run Complete Analysis' to process your data.
   This will:
   • Initialize the database
   • Load and validate your data
   • Calculate cell population frequencies
   • Perform statistical tests
   • Generate visualizations
   
   Progress will be shown in the log window.

3. VIEW RESULTS
   After analysis completes:
   • Click 'View Interactive Dashboard' to explore data
   • Click 'Open Output Files' to see generated CSV/PNG files

DATA FORMAT:
Your CSV file should include columns:
project, subject, condition, age, sex, treatment, response,
sample, time_from_b_cell, b_cell, cd8_t_cell, cd4_t_cell,
nk_cell, monocyte

For more information, see README.md
        """
        
        help_window = tk.Toplevel(self.root)
        help_window.title("Help")
        help_window.geometry("600x500")
        
        text_widget = scrolledtext.ScrolledText(help_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, help_text)
        text_widget.config(state="disabled")


def main():
    """Main entry point for GUI application."""
    root = tk.Tk()
    
    # Set style
    style = ttk.Style()
    style.theme_use('clam')  # Modern theme
    
    # Create app
    app = LoblawBioGUI(root)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    # Run
    root.mainloop()


if __name__ == "__main__":
    main()