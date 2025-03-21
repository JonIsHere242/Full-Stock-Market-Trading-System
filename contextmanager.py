import os
import re
import glob
import pandas as pd
import json
import traceback
from datetime import datetime
import concurrent.futures
from pathlib import Path
import argparse
import random
from collections import defaultdict


import inspect
import os
import re
import glob
import json
import ast
import nbformat
from collections import defaultdict
from pathlib import Path

class EnhancedTradingSystemAnalyzer:
    """
    An enhanced trading system analyzer that focuses on discovering file structure patterns
    and data organization within a trading system codebase.
    
    Improvements:
    - Better directory structure visualization
    - Pattern detection in directories and files
    - Representative sample extraction from data files
    - Schema evolution tracking across the pipeline
    """
    
    def __init__(self, root_dir='.', max_file_size_mb=10, max_workers=8, ignore_git=True):
        self.root_dir = os.path.abspath(root_dir)
        self.max_file_size_mb = max_file_size_mb
        self.max_workers = max_workers
        self.ignore_git = ignore_git  # New flag to control Git filtering

        # Core system tracking
        self.components = {}
        self.file_structure = {}
        self.data_patterns = {}

        # Pattern detection
        self.ticker_pattern = re.compile(r'^[A-Z0-9]{1,5}\.parquet$')
        self.enhanced_ticker_pattern = re.compile(r'^[A-Z0-9]{1,5}_DAILY_ENHANCED\.parquet$')
        self.dated_pattern = re.compile(r'.*_\d{8}\.parquet$')

        # Sample storage
        self.data_samples = {}
        self.schema_evolution = {}

        # Patterns for files to ignore
        self.ignore_patterns = [
            r'\.git.*',            # Git directories and files
            r'\.gitignore',        # Git ignore file
            r'\.gitattributes',    # Git attributes file
            r'\.github.*'          # GitHub specific files and directories
        ]
        if self.ignore_git:
            self.ignore_regex = re.compile('|'.join(self.ignore_patterns))
        else:
            self.ignore_regex = None
        


    def _should_ignore(self, path):
        """Determine if a file or directory should be ignored"""
        if not self.ignore_git:
            return False

        # Get the basename
        basename = os.path.basename(path)

        # Check against our ignore patterns
        if self.ignore_regex and self.ignore_regex.match(basename):
            return True

        # Special case for .git directory
        if basename == '.git':
            return True

        return False



    def analyze(self):
        """Run a complete analysis of the trading system structure"""
        print(f"Starting enhanced analysis of {self.root_dir}")
        
        # First analyze main scripts to establish component structure
        self._analyze_components()
        
        # Then analyze directory structure with pattern recognition
        self._analyze_directory_structure()
        
        # Extract representative samples from different data categories
        self._extract_data_samples()
        
        # Track schema evolution across the pipeline
        self._track_schema_evolution()
        
        print("Analysis complete")
        
    def _analyze_components(self):
        """Find and analyze the trading system components"""
        # Find numbered script components (0__app.py, 1__TickerDownloader.py, etc.)
        script_pattern = re.compile(r'(\d+)__([A-Za-z]+)\.py$')
        
        for item in os.listdir(self.root_dir):
            if item.endswith('.py'):
                match = script_pattern.match(item)
                if match:
                    number = int(match.group(1))
                    name = match.group(2)
                    filepath = os.path.join(self.root_dir, item)
                    
                    # Extract basic info about the component
                    purpose = self._extract_component_purpose(filepath)
                    self.components[number] = {
                        'name': name,
                        'purpose': purpose,
                        'file': item
                    }
    
    def _extract_component_purpose(self, filepath):
        """Extract the purpose of a component from its docstring or comments"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read(5000)  # Read first 5000 chars
                
                # Try to extract a docstring
                doc_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
                if doc_match:
                    return doc_match.group(1).strip()
                
                # Try to get comments at the top
                lines = content.split('\n')
                comment_lines = []
                for line in lines[:20]:
                    if line.strip().startswith('#'):
                        comment_lines.append(line.strip()[1:].strip())
                    elif comment_lines and not line.strip():
                        break
                
                if comment_lines:
                    return ' '.join(comment_lines)
                
                return "No description available"
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def _analyze_directory_structure(self):
        """Build a hierarchical map of the directory structure with pattern detection"""
        print(f"Analyzing directory structure...")
        
        # Start with mapping the high-level directory structure
        self.file_structure = self._map_directory(self.root_dir)
        




        
    def _map_directory(self, directory, max_depth=5, current_depth=0):
        """Map a directory recursively, detecting patterns and sampling files"""
        if current_depth >= max_depth:
            return {"_note": "Max depth reached"}

        # Get just the basename for the directory
        dir_basename = os.path.basename(directory) or os.path.basename(self.root_dir)

        # Skip Git-related directories and files
        if dir_basename == '.git' or dir_basename.startswith('.git'):
            return None

        result = {
            "name": dir_basename,
            "type": "directory",
            "children": [],
            "patterns": {}
        }

        try:
            items = os.listdir(directory)

            # Separate files and directories
            subdirs = []
            files = []

            for item in items:
                # Skip Git-related files and directories
                if item == '.git' or item.startswith('.git'):
                    continue

                full_path = os.path.join(directory, item)
                if os.path.isdir(full_path):
                    subdirs.append(full_path)
                elif os.path.isfile(full_path):
                    # Skip Git-related files like .gitignore
                    if item == '.gitignore' or item == '.gitattributes':
                        continue
                    files.append(full_path)
            
            # Check for ticker directory patterns
            ticker_dirs = [d for d in subdirs if self._is_ticker_name(os.path.basename(d))]
            if ticker_dirs and len(ticker_dirs) > 5:
                # We have a pattern of ticker directories
                result["patterns"]["ticker_dirs"] = {
                    "count": len(ticker_dirs),
                    "samples": [os.path.basename(d) for d in random.sample(ticker_dirs, min(3, len(ticker_dirs)))]
                }

                # Sample one ticker directory to see what's inside
                if ticker_dirs:
                    sample_dir = ticker_dirs[0]
                    sample_files = [os.path.basename(f) for f in glob.glob(os.path.join(sample_dir, "*")) 
                                  if os.path.isfile(f)]
                    result["patterns"]["ticker_dirs"]["sample_contents"] = sample_files[:5]

                # Remove these from normal processing
                for d in ticker_dirs:
                    if d in subdirs:
                        subdirs.remove(d)

            # Check for file patterns
            self._detect_file_patterns(files, result["patterns"])

            # Process remaining files (limit to a reasonable number)
            remaining_files = [f for f in files if not self._is_in_pattern(f, result["patterns"])]
            for file_path in remaining_files[:10]:  # Limit to 10 non-pattern files
                file_info = {
                    "name": os.path.basename(file_path),
                    "type": "file",
                    "size_mb": os.path.getsize(file_path) / (1024 * 1024)
                }

                # Get sample content for important files
                if file_info["size_mb"] < self.max_file_size_mb:
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext == '.parquet':
                        file_info["sample"] = self._sample_parquet(file_path)
                    elif ext in ['.csv', '.txt', '.md', '.py']:
                        file_info["sample"] = self._sample_text_file(file_path)

                result["children"].append(file_info)

            # Process subdirectories
            for subdir in subdirs:
                # Skip if it's in a pattern we already detected
                if not any(os.path.basename(subdir) in pattern.get("samples", []) 
                         for pattern in result["patterns"].values()):
                    child_result = self._map_directory(subdir, max_depth, current_depth + 1)
                    if child_result is not None:  # Only add if not None (not Git-related)
                        result["children"].append(child_result)

            return result

        except Exception as e:
            print(f"Error mapping directory {directory}: {str(e)}")
            return {
                "name": dir_basename,
                "type": "directory",
                "error": str(e)
            }
    





    def _is_ticker_name(self, name):
        """Check if a name looks like a stock ticker"""
        return re.match(r'^[A-Z0-9]{1,5}$', name) is not None
    
    def _detect_file_patterns(self, files, patterns_dict):
        """Detect patterns in file names"""
        # Check for ticker parquet files
        ticker_files = [f for f in files if self.ticker_pattern.match(os.path.basename(f))]
        if ticker_files and len(ticker_files) > 5:
            patterns_dict["ticker_parquet"] = {
                "pattern": "{TICKER}.parquet",
                "count": len(ticker_files),
                "samples": [os.path.basename(f) for f in random.sample(ticker_files, min(3, len(ticker_files)))]
            }
            
            # Sample one file for structure
            if ticker_files:
                patterns_dict["ticker_parquet"]["sample_structure"] = self._sample_parquet(ticker_files[0])
        
        # Check for enhanced daily files
        enhanced_files = [f for f in files if self.enhanced_ticker_pattern.match(os.path.basename(f))]
        if enhanced_files and len(enhanced_files) > 5:
            patterns_dict["enhanced_ticker"] = {
                "pattern": "{TICKER}_DAILY_ENHANCED.parquet",
                "count": len(enhanced_files),
                "samples": [os.path.basename(f) for f in random.sample(enhanced_files, min(3, len(enhanced_files)))]
            }
            
            # Sample one file for structure
            if enhanced_files:
                patterns_dict["enhanced_ticker"]["sample_structure"] = self._sample_parquet(enhanced_files[0])
        
        # Check for dated files
        dated_files = [f for f in files if self.dated_pattern.match(os.path.basename(f))]
        if dated_files and len(dated_files) > 3:
            patterns_dict["dated_files"] = {
                "pattern": "{NAME}_{YYYYMMDD}.parquet",
                "count": len(dated_files),
                "samples": [os.path.basename(f) for f in random.sample(dated_files, min(3, len(dated_files)))]
            }
    
    def _is_in_pattern(self, file_path, patterns_dict):
        """Check if a file is already covered by a detected pattern"""
        filename = os.path.basename(file_path)
        
        for pattern_info in patterns_dict.values():
            if "samples" in pattern_info and filename in pattern_info["samples"]:
                return True
                
            pattern_str = pattern_info.get("pattern", "")
            if pattern_str == "{TICKER}.parquet" and self.ticker_pattern.match(filename):
                return True
            elif pattern_str == "{TICKER}_DAILY_ENHANCED.parquet" and self.enhanced_ticker_pattern.match(filename):
                return True
            elif pattern_str == "{NAME}_{YYYYMMDD}.parquet" and self.dated_pattern.match(filename):
                return True
                
        return False
    
    def _sample_parquet(self, file_path):
        """Extract a sample from a parquet file"""
        try:
            # Check file size
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                return {
                    "error": f"File too large to sample: {file_size_mb:.2f} MB"
                }
            
            df = pd.read_parquet(file_path)
            return {
                "shape": df.shape,
                "columns": list(df.columns),
                "sample_row": df.iloc[0].to_dict() if len(df) > 0 else None
            }
        except Exception as e:
            return {
                "error": f"Error sampling parquet: {str(e)}"
            }
    
    def _sample_text_file(self, file_path):
        """Extract a sample from a text file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read(1000)  # Read first 1000 chars
                
                # Get first 5 non-empty lines
                lines = [line for line in content.split('\n') if line.strip()][:5]
                
                return {
                    "lines": lines,
                    "is_truncated": len(content) >= 1000
                }
        except Exception as e:
            return {
                "error": f"Error sampling text file: {str(e)}"
            }
    
    def _extract_data_samples(self):
        """Extract representative samples from key data directories"""
        print("Extracting data samples...")
        
        # Look for the Data directory
        data_dir = os.path.join(self.root_dir, 'Data')
        if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
            print("No Data directory found")
            return
            
        # Get all subdirectories of Data
        subdirs = [d for d in os.listdir(data_dir) 
                 if os.path.isdir(os.path.join(data_dir, d))]
                 
        # Process each data directory to extract samples
        for subdir in subdirs:
            subdir_path = os.path.join(data_dir, subdir)
            self.data_samples[subdir] = self._extract_directory_samples(subdir_path)
    
    def _extract_directory_samples(self, dir_path):
        """Extract samples from a specific data directory"""
        result = {
            "file_count": 0,
            "sample_files": []
        }
        
        try:
            # Count parquet files
            parquet_files = glob.glob(os.path.join(dir_path, "*.parquet"))
            result["file_count"] = len(parquet_files)
            
            # If there are subdirectories with ticker names, check those too
            ticker_dirs = [d for d in os.listdir(dir_path) 
                         if os.path.isdir(os.path.join(dir_path, d)) and 
                         self._is_ticker_name(d)]
                         
            if ticker_dirs:
                result["ticker_dirs"] = {
                    "count": len(ticker_dirs),
                    "samples": ticker_dirs[:3]
                }
                
                # Sample files from a ticker directory
                if ticker_dirs:
                    sample_dir = os.path.join(dir_path, ticker_dirs[0])
                    sample_files = glob.glob(os.path.join(sample_dir, "*.parquet"))
                    if sample_files:
                        sample_file = sample_files[0]
                        result["ticker_dirs"]["sample_file"] = os.path.basename(sample_file)
                        result["ticker_dirs"]["sample_structure"] = self._sample_parquet(sample_file)
            
            # Get samples from main directory
            if parquet_files:
                # Sample up to 3 files
                sample_files = random.sample(parquet_files, min(3, len(parquet_files)))
                
                for file_path in sample_files:
                    sample = {
                        "file": os.path.basename(file_path),
                        "structure": self._sample_parquet(file_path)
                    }
                    result["sample_files"].append(sample)
            
            return result
            
        except Exception as e:
            print(f"Error extracting samples from {dir_path}: {str(e)}")
            return {
                "error": str(e)
            }
    
    def _track_schema_evolution(self):
        """Track how data schemas evolve through the pipeline"""
        print("Tracking schema evolution...")
        
        # Only process if we have data samples
        if not self.data_samples:
            return
            
        # Map schema changes across pipeline stages
        stages = ['TickerCikData', 'PriceData', 'IndicatorData', 'RFpredictions']
        
        # Get columns from each stage
        for stage in stages:
            if stage in self.data_samples:
                sample_files = self.data_samples[stage].get("sample_files", [])
                
                if sample_files:
                    for sample in sample_files:
                        structure = sample.get("structure", {})
                        if "columns" in structure:
                            if stage not in self.schema_evolution:
                                self.schema_evolution[stage] = {
                                    "columns": structure["columns"],
                                    "count": len(structure["columns"])
                                }
                            break  # Just need one sample per stage
    
    def generate_report(self, output_file="Data/Context/TRADING_SYSTEM_STRUCTURE.md"):

        """Generate a comprehensive report on the trading system structure"""
        print(f"Generating report to {output_file}...")
        
        report = []
        
        # Title
        report.append("# Trading System Structure Analysis")
        report.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
        report.append("")
        
        # Component overview
        report.append("## System Components")
        report.append("")
        
        if self.components:
            # Create a workflow diagram
            workflow = " → ".join([f"[{num}] {info['name']}" for num, info in sorted(self.components.items())])
            report.append(f"`{workflow}`")
            report.append("")
            
            # List individual components
            for num, info in sorted(self.components.items()):
                report.append(f"**[{num}] {info['name']}**")
                report.append(f"- File: `{info['file']}`")
                report.append(f"- Purpose: {info['purpose'][:150]}{'...' if len(info['purpose']) > 150 else ''}")
                report.append("")
        else:
            report.append("No system components identified.")
            report.append("")
        
        # Data directories and patterns
        report.append("## Data Structure")
        report.append("")
        
        # Data directory analysis
        if self.data_samples:
            # Table header
            report.append("| Directory | Files | Pattern | Sample Columns |")
            report.append("|-----------|-------|---------|----------------|")
            
            for dir_name, samples in self.data_samples.items():
                file_count = samples.get("file_count", 0)
                
                pattern = ""
                if "ticker_dirs" in samples:
                    ticker_count = samples["ticker_dirs"].get("count", 0)
                    pattern = f"{ticker_count} ticker subdirectories"
                elif "sample_files" in samples and samples["sample_files"]:
                    sample_file = samples["sample_files"][0]["file"]
                    if self.ticker_pattern.match(sample_file):
                        pattern = "{TICKER}.parquet"
                    elif self.enhanced_ticker_pattern.match(sample_file):
                        pattern = "{TICKER}_DAILY_ENHANCED.parquet"
                
                # Sample columns
                columns = ""
                if "sample_files" in samples and samples["sample_files"]:
                    structure = samples["sample_files"][0].get("structure", {})
                    if "columns" in structure:
                        columns = ", ".join(structure["columns"][:3]) + "..."
                
                report.append(f"| {dir_name} | {file_count} | {pattern} | {columns} |")
                
            report.append("")
        
        # Schema evolution
        report.append("## Data Pipeline Evolution")
        report.append("")
        
        if self.schema_evolution:
            report.append("As data flows through the system, the schema evolves:")
            report.append("")
            
            stages = ['TickerCikData', 'PriceData', 'IndicatorData', 'RFpredictions']
            for stage in stages:
                if stage in self.schema_evolution:
                    info = self.schema_evolution[stage]
                    report.append(f"**{stage}**: {info['count']} columns")
                    report.append(f"- Sample columns: {', '.join(info['columns'][:10])}" + 
                                ("..." if len(info['columns']) > 10 else ""))
                    report.append("")
        
        # Directory structure visualization
        report.append("## Directory Structure")
        report.append("")
        report.append("```")
        report.append(self._format_directory_structure(self.file_structure))
        report.append("```")
        
        # Write the report
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report))
            
        print(f"Report generated: {output_file}")
        return output_file
    
    def _format_directory_structure(self, structure, indent=0, max_depth=3, current_depth=0):
        """Format directory structure for display"""
        if current_depth > max_depth:
            return " " * indent + "..."
            
        result = []
        
        name = structure.get("name", "Unknown")
        result.append(" " * indent + f"📁 {name}")
        
        # Add pattern information
        patterns = structure.get("patterns", {})
        for pattern_name, pattern_info in patterns.items():
            if pattern_name == "ticker_dirs":
                count = pattern_info.get("count", 0)
                samples = pattern_info.get("samples", [])
                result.append(" " * (indent + 2) + f"📂 {count} ticker directories: {', '.join(samples)} ...")
                
                # Add sample content
                if "sample_contents" in pattern_info:
                    sample_files = pattern_info["sample_contents"]
                    result.append(" " * (indent + 4) + f"Example contents: {', '.join(sample_files)}")
            else:
                count = pattern_info.get("count", 0)
                pattern = pattern_info.get("pattern", "")
                samples = pattern_info.get("samples", [])
                result.append(" " * (indent + 2) + f"📄 {count} files matching {pattern}: {', '.join(samples)} ...")
                
                # Add sample structure for first file
                if "sample_structure" in pattern_info:
                    structure = pattern_info["sample_structure"]
                    if "shape" in structure:
                        shape = structure["shape"]
                        result.append(" " * (indent + 4) + f"Shape: {shape}")
                    if "columns" in structure:
                        columns = structure["columns"][:5]
                        result.append(" " * (indent + 4) + f"Columns: {', '.join(columns)}...")
        
        # Process children
        children = structure.get("children", [])
        for child in children:
            if child.get("type") == "directory":
                child_result = self._format_directory_structure(child, indent + 2, 
                                                              max_depth, current_depth + 1)
                result.append(child_result)
            else:
                file_name = child.get("name", "Unknown")
                file_size = child.get("size_mb", 0)
                result.append(" " * (indent + 2) + f"📄 {file_name} ({file_size:.2f} MB)")
                
                # Add sample info for parquet
                if "sample" in child:
                    sample = child["sample"]
                    if "shape" in sample:
                        shape = sample["shape"]
                        result.append(" " * (indent + 4) + f"Shape: {shape}")
                    if "columns" in sample:
                        columns = sample["columns"][:5]
                        result.append(" " * (indent + 4) + f"Columns: {', '.join(columns)}...")
        
        return "\n".join(result)

    def generate_visual_report(self, output_file="Data/Context/TRADING_SYSTEM_VISUAL.html"):
        """Generate a visual HTML report for the trading system structure"""
        print(f"Generating visual report to {output_file}...")
        
        html = []
        
        # HTML head
        html.append("<!DOCTYPE html>")
        html.append("<html lang='en'>")
        html.append("<head>")
        html.append("  <meta charset='UTF-8'>")
        html.append("  <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
        html.append("  <title>Trading System Structure</title>")
        html.append("  <style>")
        html.append("    body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 1200px; margin: 0 auto; }")
        html.append("    h1, h2, h3 { color: #2c3e50; }")
        html.append("    .workflow { padding: 15px; background: #f8f9fa; border-radius: 5px; margin-bottom: 20px; overflow-x: auto; }")
        html.append("    .file-structure { font-family: monospace; white-space: pre; background: #f8f9fa; padding: 15px; border-radius: 5px; overflow: auto; }")
        html.append("    .component { border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin-bottom: 15px; }")
        html.append("    .component-header { font-weight: bold; }")
        html.append("    .data-flow { display: flex; overflow-x: auto; padding: 20px 0; }")
        html.append("    .data-stage { flex: 1; min-width: 180px; padding: 15px; border: 1px solid #ddd; margin-right: 10px; border-radius: 5px; }")
        html.append("    .data-stage h3 { margin-top: 0; }")
        html.append("    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }")
        html.append("    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html.append("    th { background-color: #f2f2f2; }")
        html.append("    .pattern { background-color: #f8f9fa; padding: 8px; border-radius: 5px; margin-bottom: 5px; }")
        html.append("    .pattern-header { font-weight: bold; }")
        html.append("    .arrow { display: flex; align-items: center; margin: 0 5px; font-size: 24px; color: #7f8c8d; }")
        html.append("    .collapsible { cursor: pointer; }")
        html.append("    .content { max-height: 0; overflow: hidden; transition: max-height 0.2s ease-out; }")
        html.append("    .active + .content { max-height: 500px; }")
        html.append("  </style>")
        html.append("</head>")
        html.append("<body>")
        
        # Title
        html.append("<h1>Trading System Structure Analysis</h1>")
        html.append(f"<p><em>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</em></p>")
        
        # Components
        html.append("<h2>System Components</h2>")
        
        if self.components:
            # Create a workflow diagram
            html.append("<div class='workflow'>")
            workflow = " → ".join([f"[{num}] {info['name']}" for num, info in sorted(self.components.items())])
            html.append(f"<code>{workflow}</code>")
            html.append("</div>")
            
            # Component details
            for num, info in sorted(self.components.items()):
                html.append("<div class='component'>")
                html.append(f"  <div class='component-header'>[{num}] {info['name']}</div>")
                html.append(f"  <p><strong>File:</strong> {info['file']}</p>")
                html.append(f"  <p><strong>Purpose:</strong> {info['purpose'][:150]}{'...' if len(info['purpose']) > 150 else ''}</p>")
                html.append("</div>")
        else:
            html.append("<p>No system components identified.</p>")
        
        # Data Evolution
        html.append("<h2>Data Pipeline</h2>")
        
        if self.schema_evolution:
            html.append("<div class='data-flow'>")
            
            # Generate stages based on schema evolution
            stages = ['TickerCikData', 'PriceData', 'IndicatorData', 'RFpredictions']
            for i, stage in enumerate(stages):
                if stage in self.schema_evolution:
                    info = self.schema_evolution[stage]
                    
                    html.append("<div class='data-stage'>")
                    html.append(f"  <h3>{stage}</h3>")
                    html.append(f"  <p><strong>{info['count']} columns</strong></p>")
                    html.append("  <p><strong>Sample columns:</strong></p>")
                    html.append("  <ul>")
                    for col in info['columns'][:5]:
                        html.append(f"    <li>{col}</li>")
                    html.append("  </ul>")
                    html.append("</div>")
                    
                    # Add arrow between stages
                    if i < len(stages) - 1 and stages[i+1] in self.schema_evolution:
                        html.append("<div class='arrow'>→</div>")
                        
            html.append("</div>")
        
        # Data Directory Structure
        html.append("<h2>Data Directories</h2>")
        
        if self.data_samples:
            html.append("<table>")
            html.append("<tr><th>Directory</th><th>Files</th><th>Pattern</th><th>Sample</th></tr>")
            
            for dir_name, samples in self.data_samples.items():
                file_count = samples.get("file_count", 0)
                
                pattern = "N/A"
                if "ticker_dirs" in samples:
                    ticker_count = samples["ticker_dirs"].get("count", 0)
                    pattern = f"{ticker_count} ticker subdirectories"
                elif "sample_files" in samples and samples["sample_files"]:
                    sample_file = samples["sample_files"][0]["file"]
                    if self.ticker_pattern.match(sample_file):
                        pattern = "{TICKER}.parquet"
                    elif self.enhanced_ticker_pattern.match(sample_file):
                        pattern = "{TICKER}_DAILY_ENHANCED.parquet"
                
                # Sample columns
                sample_html = ""
                if "sample_files" in samples and samples["sample_files"]:
                    structure = samples["sample_files"][0].get("structure", {})
                    if "shape" in structure:
                        shape = structure["shape"]
                        sample_html += f"<div><strong>Shape:</strong> {shape}</div>"
                    if "columns" in structure and structure["columns"]:
                        columns = structure["columns"][:5]
                        sample_html += f"<div><strong>Columns:</strong> {', '.join(columns)}...</div>"
                    if "sample_row" in structure and structure["sample_row"]:
                        sample_html += "<details><summary>Sample Row</summary><pre>"
                        # Format first 3 items of sample row
                        row_items = list(structure["sample_row"].items())[:3]
                        for k, v in row_items:
                            sample_html += f"{k}: {v}\n"
                        sample_html += "...</pre></details>"
                
                html.append(f"<tr><td>{dir_name}</td><td>{file_count}</td><td>{pattern}</td><td>{sample_html}</td></tr>")
            
            html.append("</table>")
        
        # File Structure Tree
        html.append("<h2>File Structure Tree</h2>")
        html.append("<p>Click on folders to expand/collapse</p>")
        html.append("<div class='file-structure'>")
        html.append(self._generate_html_file_tree(self.file_structure))
        html.append("</div>")
        
        # JavaScript for interactivity
        html.append("<script>")
        html.append("var coll = document.getElementsByClassName('collapsible');")
        html.append("for (var i = 0; i < coll.length; i++) {")
        html.append("  coll[i].addEventListener('click', function() {")
        html.append("    this.classList.toggle('active');")
        html.append("  });")
        html.append("}")
        html.append("</script>")
        
        # Close HTML tags
        html.append("</body>")
        html.append("</html>")
        
        # Write the HTML report
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(html))
            
        print(f"Visual report generated: {output_file}")
        return output_file
    
    def _generate_html_file_tree(self, structure, indent=0):
        """Generate HTML representation of file structure tree with collapsible folders"""
        result = []
        
        name = structure.get("name", "Unknown")
        folder_id = f"folder_{random.randint(1000, 9999)}"
        
        # Create folder entry with collapsible content
        result.append(f"<div style='margin-left: {indent}px; margin-bottom: 5px;'>")
        result.append(f"  <span class='collapsible'>📁 {name}</span>")
        result.append(f"  <div class='content'>")
        
        # Add pattern information
        patterns = structure.get("patterns", {})
        for pattern_name, pattern_info in patterns.items():
            result.append(f"<div style='margin-left: {indent+20}px;' class='pattern'>")
            
            if pattern_name == "ticker_dirs":
                count = pattern_info.get("count", 0)
                samples = pattern_info.get("samples", [])
                result.append(f"<div class='pattern-header'>📂 {count} ticker directories</div>")
                result.append(f"<div>Samples: {', '.join(samples)} ...</div>")
                
                # Add sample content
                if "sample_contents" in pattern_info:
                    sample_files = pattern_info["sample_contents"]
                    result.append(f"<div>Example contents: {', '.join(sample_files)}</div>")
            else:
                count = pattern_info.get("count", 0)
                pattern = pattern_info.get("pattern", "")
                samples = pattern_info.get("samples", [])
                result.append(f"<div class='pattern-header'>📄 {count} files matching {pattern}</div>")
                result.append(f"<div>Samples: {', '.join(samples)} ...</div>")
                
                # Add sample structure for first file
                if "sample_structure" in pattern_info:
                    structure = pattern_info["sample_structure"]
                    if "shape" in structure:
                        shape = structure["shape"]
                        result.append(f"<div>Shape: {shape}</div>")
                    if "columns" in structure:
                        columns = structure["columns"][:5]
                        result.append(f"<div>Columns: {', '.join(columns)}...</div>")
            
            result.append("</div>")
        
        # Process children
        children = structure.get("children", [])
        for child in children:
            if child.get("type") == "directory":
                child_html = self._generate_html_file_tree(child, indent + 20)
                result.append(child_html)
            else:
                file_name = child.get("name", "Unknown")
                file_size = child.get("size_mb", 0)
                
                result.append(f"<div style='margin-left: {indent+20}px;'>")
                result.append(f"  📄 {file_name} ({file_size:.2f} MB)")
                
                # Add sample info for parquet
                if "sample" in child:
                    sample = child["sample"]
                    if "shape" in sample:
                        shape = sample["shape"]
                        result.append(f"  <div style='margin-left: {indent+40}px;'>Shape: {shape}</div>")
                    if "columns" in sample:
                        columns = sample["columns"][:5]
                        result.append(f"  <div style='margin-left: {indent+40}px;'>Columns: {', '.join(columns)}...</div>")
                
                result.append("</div>")
        
        result.append("  </div>") # Close content div
        result.append("</div>")   # Close folder div
        
        return "\n".join(result)







class RequirementsGenerator:
    """
    A class to scan Python files and Jupyter notebooks in a project directory
    and generate a requirements.txt file with all the imported packages.
    """
    
    def __init__(self, root_dir='.', output_path='Data/Context/requirements.txt'):
        self.root_dir = os.path.abspath(root_dir)
        self.output_path = output_path
        self.imports = set()
        self.common_builtin_modules = {
            'os', 'sys', 'time', 're', 'datetime', 'json', 'logging', 'traceback',
            'collections', 'functools', 'random', 'math', 'io', 'contextlib',
            'uuid', 'pathlib', 'glob', 'argparse', 'warnings', 'inspect', 'concurrent', 
            'zoneinfo', 'ast', 'multiprocessing', 'typing'
        }
        
    def scan_project(self):
        """Scan all Python files and Jupyter notebooks in the project directory"""
        print(f"Scanning project directory: {self.root_dir}")
        
        # Find all Python files
        py_files = glob.glob(f"{self.root_dir}/**/*.py", recursive=True)
        
        # Find all Jupyter notebooks
        ipynb_files = glob.glob(f"{self.root_dir}/**/*.ipynb", recursive=True)
        
        # Process Python files
        for py_file in py_files:
            self._process_py_file(py_file)
            
        # Process Jupyter notebooks
        for ipynb_file in ipynb_files:
            self._process_ipynb_file(ipynb_file)
            
        print(f"Found {len(self.imports)} unique package imports")
        
    def _process_py_file(self, file_path):
        """Extract imports from a Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                
            self._extract_imports_from_code(content)
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            
    def _process_ipynb_file(self, file_path):
        """Extract imports from a Jupyter notebook"""
        try:
            # Load the notebook
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                notebook = nbformat.read(f, as_version=4)
                
            # Process each code cell
            for cell in notebook.cells:
                if cell.cell_type == 'code':
                    self._extract_imports_from_code(cell.source)
                    
        except Exception as e:
            print(f"Error processing notebook {file_path}: {str(e)}")
            
    def _extract_imports_from_code(self, code):
        """Extract imports from a code string using AST parsing and regex"""
        # Use AST parsing for accurate import statements detection
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        module_name = name.name.split('.')[0]
                        if module_name not in self.common_builtin_modules:
                            self.imports.add(module_name)
                            
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module.split('.')[0]
                        if module_name not in self.common_builtin_modules:
                            self.imports.add(module_name)
        except SyntaxError:
            # If AST parsing fails, fall back to regex
            self._extract_imports_with_regex(code)
            
    def _extract_imports_with_regex(self, code):
        """Extract imports using regex as a fallback method"""
        # Match 'import package' and 'from package import ...'
        import_regex = r'^\s*import\s+([a-zA-Z0-9_]+)'
        from_regex = r'^\s*from\s+([a-zA-Z0-9_]+)'
        
        # Find all imports
        for line in code.splitlines():
            for regex in [import_regex, from_regex]:
                matches = re.findall(regex, line)
                for match in matches:
                    if match not in self.common_builtin_modules:
                        self.imports.add(match)
    
    def _map_to_package_name(self, import_name):
        """Map import name to PyPI package name"""
        # Common mappings between import name and package name
        mappings = {
            'sklearn': 'scikit-learn',
            'PIL': 'pillow',
            'cv2': 'opencv-python',
            'bs4': 'beautifulsoup4',
            'yaml': 'pyyaml',
            'plt': 'matplotlib',
            'sklearn_pandas': 'sklearn-pandas',
            'pytz': 'pytz',
            'bt': 'backtrader',
            'ibi': 'ib_insync',
            'matplotlib.pyplot': 'matplotlib',
            'seaborn': 'seaborn',
            'sns': 'seaborn',
            'pq': 'pyarrow',
            'ec': 'exchange_calendars',
            'mcal': 'pandas_market_calendars',
        }
        
        return mappings.get(import_name, import_name)
    
    def generate_requirements(self):
        """Generate a requirements.txt file"""
        print(f"Generating requirements.txt file at {self.output_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Normalize package names and sort them
        packages = sorted([self._map_to_package_name(name) for name in self.imports])
        
        # Add common version requirements for known packages
        package_versions = {
            'pandas': '>=1.3.0',
            'numpy': '>=1.20.0',
            'requests': '>=2.25.0',
            'scipy': '>=1.6.0',
            'tqdm': '>=4.60.0',
            'pykalman': '>=0.9.5',
            'numba': '>=0.53.0',
            'matplotlib': '>=3.4.0',
            'xgboost': '>=1.4.0',
            'scikit-learn': '>=0.24.0',
            'joblib': '>=1.0.0',
            'backtrader': '>=1.9.76',
            'pandas-market-calendars': '>=3.2',
            'exchange-calendars': '>=3.0',
            'ib_async': '>=1.0.3',
            'ib_insync': '>=0.9.70',
            'pyarrow': '>=5.0.0',
            'seaborn': '>=0.11.0',
        }
        
        # Write to requirements.txt
        with open(self.output_path, 'w') as f:
            for package in packages:
                version = package_versions.get(package, '')
                f.write(f"{package}{version}\n")
                
        print(f"Generated requirements.txt with {len(packages)} packages")
        return self.output_path

def add_requirements_generator_to_analyzer(analyzer_code):
    """Add requirements generation functionality to the existing analyzer code"""
    # Find the class definition
    class_match = re.search(r'class EnhancedTradingSystemAnalyzer:', analyzer_code)
    if not class_match:
        return analyzer_code
    
    # Find the generate_report method to position our new method
    method_match = re.search(r'def generate_report\(self', analyzer_code)
    if not method_match:
        return analyzer_code
    
    # Add our new method before generate_report
    new_method = """
    def generate_requirements(self, output_file="Data/Context/requirements.txt"):
        \"\"\"Generate a requirements.txt file by scanning Python files and notebooks for imports\"\"\"
        print(f"Generating requirements.txt file to {output_file}...")
        
        # Create the requirements generator
        req_gen = RequirementsGenerator(root_dir=self.root_dir, output_path=output_file)
        
        # Scan the project and generate requirements
        req_gen.scan_project()
        req_gen.generate_requirements()
        
        print(f"Requirements file generated: {output_file}")
        return output_file
    
    """
    
    # Insert the new method before generate_report
    position = method_match.start()
    modified_code = analyzer_code[:position] + new_method + analyzer_code[position:]
    
    # Now find the end of the if __name__ == "__main__" block to add our RequirementsGenerator class
    main_match = re.search(r'if __name__ == "__main__":', modified_code)
    if not main_match:
        # If no main block, add the class at the end
        modified_code += "\n\n# Requirements generator class\n"
        modified_code += inspect.getsource(RequirementsGenerator)
    else:
        # Find the end of the file
        modified_code += "\n\n# Requirements generator class\n"
        modified_code += inspect.getsource(RequirementsGenerator)
    
    # Add the requirement generation to the main function
    args_match = re.search(r'args = parser\.parse_args\(\)', modified_code)
    if args_match:
        # Add the argument to parser
        parser_match = re.search(r'parser = argparse\.ArgumentParser\(', modified_code)
        if parser_match:
            parser_end = modified_code.find(')', parser_match.end())
            add_arg = "    parser.add_argument('--generate_requirements', action='store_true', help='Generate requirements.txt file')\n"
            
            # Find the line after parser definition
            parser_block_end = modified_code.find('\n', parser_end)
            modified_code = modified_code[:parser_block_end+1] + add_arg + modified_code[parser_block_end+1:]
        
        # Add the conditional to call generate_requirements
        add_condition = "\n    if args.generate_requirements:\n        analyzer.generate_requirements()\n"
        
        # Find a good position to add the condition
        position = modified_code.find('analyzer.generate_report(args.output)', args_match.end())
        if position != -1:
            modified_code = modified_code[:position] + add_condition + modified_code[position:]
    
    return modified_code





# Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced Trading System Structure Analyzer and Requirements Generator')
    
    # General arguments
    parser.add_argument('--dir', default='.', help='Root directory to analyze')
    
    # Analyzer specific arguments
    parser.add_argument('--output', default='Data/Context/TRADING_SYSTEM_STRUCTURE.md', help='Output markdown file for system structure')
    parser.add_argument('--visual', default='Data/Context/TRADING_SYSTEM_VISUAL.html', help='Output HTML file for system structure')
    parser.add_argument('--max_file_size', type=int, default=10, help='Maximum file size to analyze in MB')
    parser.add_argument('--max_workers', type=int, default=8, help='Number of worker threads')
    
    # RequirementsGenerator specific arguments
    parser.add_argument('--requirements', action='store_true', help='Generate requirements.txt file')
    parser.add_argument('--req_output', default='Data/Context/requirements.txt', help='Output file for requirements.txt')
    
    # Mode selection
    parser.add_argument('--mode', choices=['analyze', 'requirements', 'both'], default='both',
                        help='Mode to run: analyze (structure only), requirements (deps only), or both')
    
    args = parser.parse_args()
    
    # Check if Data/Context directory exists, create if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Create analyzer instance
    analyzer = EnhancedTradingSystemAnalyzer(
        root_dir=args.dir,
        max_file_size_mb=args.max_file_size,
        max_workers=args.max_workers
    )
    
    # Run based on mode
    if args.mode in ['analyze', 'both']:
        print(f"Running system structure analysis on {args.dir}...")
        analyzer.analyze()
        analyzer.generate_report(args.output)
        analyzer.generate_visual_report(args.visual)
        print(f"Analysis complete. Reports generated at {args.output} and {args.visual}")
    
    if args.mode in ['requirements', 'both'] or args.requirements:
        print(f"Generating requirements.txt for {args.dir}...")
        req_gen = RequirementsGenerator(root_dir=args.dir, output_path=args.req_output)
        req_gen.scan_project()
        req_gen.generate_requirements()
        print(f"Requirements.txt generated at {args.req_output}")
    
    print("All tasks completed successfully!")