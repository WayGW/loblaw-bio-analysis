import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys
from pathlib import Path
import csv
from PIL import Image, ImageTk


def get_outputs_path():
    """Get the correct path to outputs directory"""
    if getattr(sys, 'frozen', False):
        # Running from .exe - outputs should be relative to exe location
        base_path = os.path.dirname(sys.executable)
    else:
        # Running from Python
        base_path = os.getcwd()
    return os.path.join(base_path, 'outputs')


class SimpleResultsViewer:
    """Simple viewer for displaying analysis results with tabs"""
    
    def __init__(self, callback=None):
        """Initialize the simple results viewer
        
        Args:
            callback: Optional function to call when window closes
        """
        self.window = tk.Tk()
        self.window.title("Loblaw Bio - Analysis Results")
        self.window.geometry("1000x700")
        self.callback = callback
        
        # CRITICAL FIX: Pre-initialize Tkinter's image system for frozen environments
        # This prevents "pyimage doesn't exist" errors in PyInstaller builds
        try:
            # Create a dummy PhotoImage to initialize the Tk image system
            dummy = tk.PhotoImage(width=1, height=1)
            dummy_label = tk.Label(self.window, image=dummy)
            dummy_label.image = dummy
            # Don't grid it, just create it to initialize the system
        except Exception:
            pass  # If this fails, continue anyway
        
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Create main container
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create tabs
        self.create_overview_tab()
        self.create_frequency_tab()
        self.create_response_tab()
        self.create_cohort_tab()
        self.create_ml_tab()
        
    def create_overview_tab(self):
        """Create overview tab"""
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="📋 Overview")
        
        # Title
        title = ttk.Label(frame, text="Analysis Results Overview", 
                         font=('Arial', 16, 'bold'))
        title.grid(row=0, column=0, pady=10, sticky=tk.W)
        
        # Summary text
        summary = tk.Text(frame, wrap=tk.WORD, height=30, width=80)
        summary.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=summary.yview)
        scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        summary.configure(yscrollcommand=scrollbar.set)
        
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        
        # Load summary
        summary_text = """
╔═══════════════════════════════════════════════════════════════╗
║         LOBLAW BIO - CLINICAL TRIAL ANALYSIS RESULTS          ║
╚═══════════════════════════════════════════════════════════════╝

Analysis Complete! 

ANALYSIS SCOPE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Indication:  Melanoma
Treatment:   Miraclib  
Sample Type: PBMC

NOTE: Statistical analyses (Response, ML) use FILTERED data with 
the above criteria. Frequency analysis shows ALL samples.

TABS AVAILABLE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Frequency Analysis
   - Cell population frequencies across ALL samples
   - Summary statistics (mean, median, std, min, max)

🔬 Response Analysis  
   - Statistical comparison of responders vs non-responders
   - FILTERED: Melanoma + Miraclib + PBMC only
   - Mann-Whitney U tests with Bonferroni correction
   - Boxplot visualization

👥 Baseline Cohort
   - Filtered baseline samples (time=0)
   - FILTERED: Melanoma + Miraclib + PBMC only
   - Demographics and cohort characteristics

🤖 ML Analysis
   - Machine learning model performance
   - FILTERED: Melanoma + Miraclib + PBMC only
   - Feature importance rankings
   - ROC curves and confusion matrices

GENERATED FILES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ outputs/frequency_table.csv (ALL samples)
✓ outputs/response_analysis_results.csv (FILTERED)
✓ outputs/response_boxplot.png (FILTERED)
✓ outputs/baseline_cohort.csv (FILTERED)
✓ outputs/ml_model_comparison.csv (FILTERED)
✓ outputs/ml_feature_importance.csv (FILTERED)
✓ outputs/ml_analysis_results.png (FILTERED)

Navigate through the tabs to explore different aspects of the analysis.

"""
        summary.insert('1.0', summary_text)
        summary.configure(state='disabled')
        
    def create_frequency_tab(self):
        """Create frequency analysis tab"""
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="📊 Frequency")
        
        # Title
        title = ttk.Label(frame, text="Cell Population Frequencies", 
                         font=('Arial', 14, 'bold'))
        title.grid(row=0, column=0, pady=10, sticky=tk.W)
        
        # Check if file exists
        outputs_dir = get_outputs_path()
        freq_file = os.path.join(outputs_dir, "frequency_table.csv")
        if not os.path.exists(freq_file):
            error_label = ttk.Label(frame, text="⚠️ Frequency table not found", 
                                   foreground='red')
            error_label.grid(row=1, column=0, pady=20)
            return
        
        # Create treeview
        tree_frame = ttk.Frame(frame)
        tree_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbars
        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal")
        
        tree = ttk.Treeview(tree_frame, 
                           yscrollcommand=vsb.set,
                           xscrollcommand=hsb.set)
        
        vsb.configure(command=tree.yview)
        hsb.configure(command=tree.xview)
        
        # Grid layout
        tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        vsb.grid(row=0, column=1, sticky=(tk.N, tk.S))
        hsb.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        
        # Load data
        with open(freq_file, 'r') as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames
            
            # Configure columns
            tree['columns'] = columns
            tree['show'] = 'headings'
            
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=100)
            
            # Insert data
            for row in reader:
                values = [row[col] for col in columns]
                tree.insert('', 'end', values=values)
    
    def create_response_tab(self):
        """Create response analysis tab"""
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="🔬 Response")
        
        # Title
        title = ttk.Label(frame, text="Responders vs Non-Responders", 
                         font=('Arial', 14, 'bold'))
        title.grid(row=0, column=0, pady=10, sticky=tk.W, columnspan=2)
        
        # Filter info
        filter_info = ttk.Label(frame, 
            text="📋 Filtered Data: Melanoma + Miraclib + PBMC samples only", 
            font=('Arial', 9, 'italic'),
            foreground='blue')
        filter_info.grid(row=1, column=0, pady=(0, 10), sticky=tk.W, columnspan=2)
        
        # Check if files exist
        outputs_dir = get_outputs_path()
        results_file = os.path.join(outputs_dir, "response_analysis_results.csv")
        plot_file = os.path.join(outputs_dir, "response_boxplot.png")
        
        if not os.path.exists(results_file):
            error_label = ttk.Label(frame, text="⚠️ Results file not found", 
                                   foreground='red')
            error_label.grid(row=2, column=0, pady=20)
            return
        
        # Create results table
        results_label = ttk.Label(frame, text="Statistical Test Results:", 
                                 font=('Arial', 11, 'bold'))
        results_label.grid(row=2, column=0, pady=5, sticky=tk.W)
        
        tree_frame = ttk.Frame(frame)
        tree_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        tree = ttk.Treeview(tree_frame, yscrollcommand=vsb.set, height=6)
        vsb.configure(command=tree.yview)
        
        tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        vsb.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        
        # Load results
        with open(results_file, 'r') as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames
            
            tree['columns'] = columns
            tree['show'] = 'headings'
            
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=120)
            
            for row in reader:
                values = [row[col] for col in columns]
                # Highlight significant results
                if 'significant' in row and row['significant'].lower() == 'yes':
                    tree.insert('', 'end', values=values, tags=('significant',))
                else:
                    tree.insert('', 'end', values=values)
        
        tree.tag_configure('significant', background='#90EE90')  # Light green
        
        # Display plot if available
        plot_file_abs = os.path.abspath(plot_file)
        if os.path.exists(plot_file_abs):
            plot_label = ttk.Label(frame, text="Visualization:", 
                                  font=('Arial', 11, 'bold'))
            plot_label.grid(row=4, column=0, pady=(20, 5), sticky=tk.W)
            
            try:
                # Load and display image with proper reference handling
                img = Image.open(plot_file_abs)
                # Resize to fit
                img.thumbnail((800, 400), Image.Resampling.LANCZOS)
                
                # CRITICAL: Use master parameter to tie PhotoImage to correct Tk instance
                photo = ImageTk.PhotoImage(img, master=self.window)
                
                img_label = ttk.Label(frame, image=photo)
                img_label.image = photo  # Keep a reference
                img_label.grid(row=5, column=0, pady=5)
                
                # Also store at frame level to prevent garbage collection
                if not hasattr(frame, '_images'):
                    frame._images = []
                frame._images.append(photo)
                
            except Exception as e:
                error_label = ttk.Label(frame, 
                    text=f"⚠️ Could not load image\nPath: {plot_file_abs}\nError: {str(e)}", 
                    foreground='red', justify=tk.LEFT)
                error_label.grid(row=5, column=0, pady=5)
        
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(3, weight=1)
    
    def create_cohort_tab(self):
        """Create cohort analysis tab"""
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="👥 Cohort")
        
        # Title
        title = ttk.Label(frame, text="Baseline Cohort", 
                         font=('Arial', 14, 'bold'))
        title.grid(row=0, column=0, pady=10, sticky=tk.W)
        
        # Check if file exists
        outputs_dir = get_outputs_path()
        cohort_file = os.path.join(outputs_dir, "baseline_cohort.csv")
        if not os.path.exists(cohort_file):
            error_label = ttk.Label(frame, text="⚠️ Cohort file not found", 
                                   foreground='red')
            error_label.grid(row=1, column=0, pady=20)
            return
        
        # Summary section
        summary_frame = ttk.LabelFrame(frame, text="Cohort Summary", padding="10")
        summary_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # Load data to get summary
        with open(cohort_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            total = len(rows)
            responders = sum(1 for r in rows if r.get('response', '').lower() == 'yes')
            non_responders = sum(1 for r in rows if r.get('response', '').lower() == 'no')
            males = sum(1 for r in rows if r.get('gender', '').lower() == 'm')
            females = sum(1 for r in rows if r.get('gender', '').lower() == 'f')
            
            summary_text = f"""
Total Samples: {total}
Responders: {responders}
Non-responders: {non_responders}
Males: {males}
Females: {females}
"""
            
            summary_label = ttk.Label(summary_frame, text=summary_text, 
                                     font=('Courier', 10))
            summary_label.grid(row=0, column=0, sticky=tk.W)
        
        # Data table
        data_label = ttk.Label(frame, text="Sample Data (first 100 rows):", 
                              font=('Arial', 11, 'bold'))
        data_label.grid(row=2, column=0, pady=5, sticky=tk.W)
        
        tree_frame = ttk.Frame(frame)
        tree_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal")
        
        tree = ttk.Treeview(tree_frame,
                           yscrollcommand=vsb.set,
                           xscrollcommand=hsb.set)
        
        vsb.configure(command=tree.yview)
        hsb.configure(command=tree.xview)
        
        tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        vsb.grid(row=0, column=1, sticky=(tk.N, tk.S))
        hsb.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(3, weight=1)
        
        # Load data
        with open(cohort_file, 'r') as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames
            
            tree['columns'] = columns
            tree['show'] = 'headings'
            
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=100)
            
            # Insert first 100 rows
            for i, row in enumerate(reader):
                if i >= 100:
                    break
                values = [row[col] for col in columns]
                tree.insert('', 'end', values=values)
    
    def create_ml_tab(self):
        """Create ML analysis tab"""
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="🤖 ML Analysis")
        
        # Title
        title = ttk.Label(frame, text="Machine Learning Analysis", 
                         font=('Arial', 14, 'bold'))
        title.grid(row=0, column=0, pady=10, sticky=tk.W, columnspan=2)
        
        # Filter info
        filter_info = ttk.Label(frame, 
            text="📋 Filtered Data: Melanoma + Miraclib + PBMC samples only", 
            font=('Arial', 9, 'italic'),
            foreground='blue')
        filter_info.grid(row=1, column=0, pady=(0, 10), sticky=tk.W, columnspan=2)
        
        # Check if files exist
        outputs_dir = get_outputs_path()
        comparison_file = os.path.join(outputs_dir, "ml_model_comparison.csv")
        importance_file = os.path.join(outputs_dir, "ml_feature_importance.csv")
        viz_file = os.path.join(outputs_dir, "ml_analysis_results.png")
        
        if not os.path.exists(comparison_file):
            info_label = ttk.Label(frame, 
                text="ℹ️ ML analysis results not available\n\nFor ML features, use: python main.py --run-all",
                foreground='blue', justify=tk.LEFT)
            info_label.grid(row=2, column=0, pady=20, padx=20)
            return
        
        # Model performance section
        perf_label = ttk.Label(frame, text="Model Performance:", 
                              font=('Arial', 11, 'bold'))
        perf_label.grid(row=2, column=0, pady=5, sticky=tk.W)
        
        # Load and display model comparison
        perf_frame = ttk.Frame(frame)
        perf_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        
        with open(comparison_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            perf_text = ""
            for row in rows:
                model_name = row.get('Model', 'Unknown')
                perf_text += f"\n{model_name}:\n"
                perf_text += "─" * 40 + "\n"
                
                for key, value in row.items():
                    if key != 'Model':
                        try:
                            val = float(value)
                            perf_text += f"  {key}: {val:.4f}\n"
                        except:
                            perf_text += f"  {key}: {value}\n"
                perf_text += "\n"
            
            perf_display = tk.Text(perf_frame, height=12, width=50, wrap=tk.NONE)
            perf_display.insert('1.0', perf_text)
            perf_display.configure(state='disabled', font=('Courier', 9))
            perf_display.grid(row=0, column=0, sticky=(tk.W, tk.E))
            
            scrollbar = ttk.Scrollbar(perf_frame, orient=tk.VERTICAL, 
                                     command=perf_display.yview)
            scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
            perf_display.configure(yscrollcommand=scrollbar.set)
        
        # Feature importance
        if os.path.exists(importance_file):
            imp_label = ttk.Label(frame, text="Feature Importance:", 
                                 font=('Arial', 11, 'bold'))
            imp_label.grid(row=4, column=0, pady=(15, 5), sticky=tk.W)
            
            tree_frame = ttk.Frame(frame)
            tree_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=5)
            
            tree = ttk.Treeview(tree_frame, height=6)
            vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=vsb.set)
            
            tree.grid(row=0, column=0, sticky=(tk.W, tk.E))
            vsb.grid(row=0, column=1, sticky=(tk.N, tk.S))
            
            with open(importance_file, 'r') as f:
                reader = csv.DictReader(f)
                columns = reader.fieldnames
                
                tree['columns'] = columns
                tree['show'] = 'headings'
                
                for col in columns:
                    tree.heading(col, text=col)
                    tree.column(col, width=150)
                
                for row in reader:
                    values = [row[col] for col in columns]
                    tree.insert('', 'end', values=values)
        
        # Visualization
        viz_file_abs = os.path.abspath(viz_file)
        
        if os.path.exists(viz_file_abs):
            viz_label = ttk.Label(frame, text="Visualizations:", 
                                 font=('Arial', 11, 'bold'))
            viz_label.grid(row=6, column=0, pady=(15, 5), sticky=tk.W)
            
            try:
                img = Image.open(viz_file_abs)
                img.thumbnail((900, 500), Image.Resampling.LANCZOS)
                
                # CRITICAL: Use master parameter to tie PhotoImage to correct Tk instance
                photo = ImageTk.PhotoImage(img, master=self.window)
                
                img_label = ttk.Label(frame, image=photo)
                img_label.image = photo
                img_label.grid(row=7, column=0, pady=5)
                
                # Also store at frame level to prevent garbage collection
                if not hasattr(frame, '_images'):
                    frame._images = []
                frame._images.append(photo)
                
            except Exception as e:
                error_label = ttk.Label(frame, 
                    text=f"⚠️ Could not load image\nPath: {viz_file_abs}\nError: {str(e)}", 
                    foreground='red', justify=tk.LEFT)
                error_label.grid(row=7, column=0, pady=5)
        else:
            # Show what path we're looking for
            outputs_dir = get_outputs_path()
            error_text = f"⚠️ Image file not found\n\nLooking for:\n{viz_file_abs}\n\nOutputs directory:\n{outputs_dir}"
            
            # List what files ARE in outputs
            if os.path.exists(outputs_dir):
                files = os.listdir(outputs_dir)
                if files:
                    error_text += f"\n\nFiles found in outputs:\n" + "\n".join(f"  • {f}" for f in files[:10])
                else:
                    error_text += "\n\n(Outputs directory is empty)"
            else:
                error_text += "\n\n(Outputs directory does not exist)"
            
            error_label = ttk.Label(frame, text=error_text, 
                                   foreground='red', justify=tk.LEFT, font=('Courier', 8))
            error_label.grid(row=6, column=0, pady=5, sticky=tk.W)
        
        frame.columnconfigure(0, weight=1)
    
    def on_closing(self):
        """Handle window close event"""
        if hasattr(self, 'callback') and self.callback:
            self.callback()
        if hasattr(self, 'window'):
            self.window.destroy()
    
    def run(self):
        """Start the viewer main loop"""
        self.window.mainloop()


if __name__ == '__main__':
    viewer = SimpleResultsViewer()
    viewer.run()