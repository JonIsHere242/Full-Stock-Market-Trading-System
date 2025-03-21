#!/usr/bin/env python3
"""
Enhanced SEC EDGAR Financial Data Processor
-------------------------------------------
A complete solution for processing large volumes of SEC filing data,
extracting financial metrics, and deriving missing values.

This script is designed to handle 20GB+ of JSON data efficiently with
parallel processing, smart tag mapping, and metric derivation.
"""

import os
import json
import re
import gc
import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import difflib
##import all util functions to set up the logging 
from Util import get_logger



logger = get_logger(script_name="Z-Macro")

# Define paths - adjust these to match your environment
BASE_DIR = Path("Data")
SEC_DIR = BASE_DIR / "SEC_Data"
FILINGS_DIR = BASE_DIR / "Filings"
OUTPUT_DIR = SEC_DIR / "Enhanced"
METRICS_DIR = SEC_DIR / "Metrics"

# Create output directories if they don't exist
for directory in [OUTPUT_DIR, METRICS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Primary and fallback tag mappings for financial metrics
METRIC_TAG_MAPPING = {
    # Income Statement metrics
    "Revenue": {
        "primary": [
            "us-gaap:Revenues", 
            "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
            "us-gaap:SalesRevenueNet"
        ],
        "fallback": [
            "us-gaap:SalesRevenueGoodsNet",
            "us-gaap:RegulatedAndUnregulatedOperatingRevenue"
        ]
    },
    "Revenue (Excluding Tax)": {
        "primary": [
            "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
        ],
        "fallback": [
            "us-gaap:Revenues",
            "us-gaap:SalesRevenueNet"
        ]
    },
    "Net Income": {
        "primary": [
            "us-gaap:NetIncomeLoss",
            "us-gaap:ProfitLoss"
        ],
        "fallback": [
            "us-gaap:IncomeLossFromContinuingOperations",
            "us-gaap:IncomeLossAttributableToParent"
        ]
    },
    "Operating Income": {
        "primary": [
            "us-gaap:OperatingIncomeLoss"
        ],
        "fallback": [
            "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
            "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxes"
        ]
    },
    "Income Before Tax": {
        "primary": [
            "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
            "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxes"
        ],
        "fallback": [
            "us-gaap:IncomeLossBeforeIncomeTaxes"
        ]
    },
    "COGS": {
        "primary": [
            "us-gaap:CostOfGoodsAndServicesSold"
        ],
        "fallback": [
            "us-gaap:CostOfRevenue",
            "us-gaap:CostOfServices",
            "us-gaap:CostOfGoodsSold"
        ]
    },
    "Gross Profit": {
        "primary": [
            "us-gaap:GrossProfit"
        ],
        "fallback": []
    },
    "EPS (Basic)": {
        "primary": [
            "us-gaap:EarningsPerShareBasic"
        ],
        "fallback": [
            "us-gaap:IncomeLossFromContinuingOperationsPerBasicShare"
        ]
    },
    "EPS (Diluted)": {
        "primary": [
            "us-gaap:EarningsPerShareDiluted"
        ],
        "fallback": [
            "us-gaap:IncomeLossFromContinuingOperationsPerDilutedShare"
        ]
    },
    "Shares Outstanding (Basic)": {
        "primary": [
            "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic"
        ],
        "fallback": [
            "us-gaap:CommonStockSharesOutstanding"
        ]
    },
    "Shares Outstanding (Diluted)": {
        "primary": [
            "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding"
        ],
        "fallback": [
            "us-gaap:WeightedAverageNumberOfSharesOutstandingDiluted"
        ]
    },
    "R&D Expense": {
        "primary": [
            "us-gaap:ResearchAndDevelopmentExpense"
        ],
        "fallback": []
    },
    "Interest Expense": {
        "primary": [
            "us-gaap:InterestExpense"
        ],
        "fallback": [
            "us-gaap:InterestAndDebtExpense"
        ]
    },
    
    # Cash Flow metrics
    "Operating Cash Flow": {
        "primary": [
            "us-gaap:NetCashProvidedByUsedInOperatingActivities"
        ],
        "fallback": []
    },
    "Investing Cash Flow": {
        "primary": [
            "us-gaap:NetCashProvidedByUsedInInvestingActivities"
        ],
        "fallback": []
    },
    "Financing Cash Flow": {
        "primary": [
            "us-gaap:NetCashProvidedByUsedInFinancingActivities"
        ],
        "fallback": []
    },
    "Net Change in Cash": {
        "primary": [
            "us-gaap:CashAndCashEquivalentsPeriodIncreaseDecrease"
        ],
        "fallback": []
    },
    
    # Other metrics
    "Share-based Compensation": {
        "primary": [
            "us-gaap:ShareBasedCompensation"
        ],
        "fallback": [
            "us-gaap:StockBasedCompensation"
        ]
    },
    "Depreciation & Amortization": {
        "primary": [
            "us-gaap:DepreciationDepletionAndAmortization"
        ],
        "fallback": [
            "us-gaap:DepreciationAndAmortization",
            "us-gaap:Depreciation"
        ]
    }
}

# Ratio metrics to calculate
RATIO_METRICS = {
    "GrossMargin": {
        "formula": lambda metrics: metrics.get("Gross Profit") / metrics.get("Revenue") if metrics.get("Revenue", 0) != 0 else None,
        "required_metrics": ["Gross Profit", "Revenue"]
    },
    "OperatingMargin": {
        "formula": lambda metrics: metrics.get("Operating Income") / metrics.get("Revenue") if metrics.get("Revenue", 0) != 0 else None,
        "required_metrics": ["Operating Income", "Revenue"]
    },
    "NetMargin": {
        "formula": lambda metrics: metrics.get("Net Income") / metrics.get("Revenue") if metrics.get("Revenue", 0) != 0 else None,
        "required_metrics": ["Net Income", "Revenue"]
    }
}

# Highly correlated metrics for derivation
METRIC_CORRELATIONS = {
    "EPS (Basic)": {"EPS (Diluted)": 1.000, "Net Income": 0.999},
    "EPS (Diluted)": {"EPS (Basic)": 1.000, "Net Income": 0.999},
    "Gross Profit": {"Revenue": 1.000, "Revenue (Excluding Tax)": 0.999},
    "Operating Income": {"Income Before Tax": 1.000},
    "Shares Outstanding (Diluted)": {"Shares Outstanding (Basic)": 0.999},
    "Shares Outstanding (Basic)": {"Shares Outstanding (Diluted)": 0.999},
    "Net Income": {"Income Before Tax": 0.998},
    "Income Before Tax": {"Net Income": 0.998}
}

# Industry-specific tag mappings
INDUSTRY_TAG_MAPPINGS = {
    # Finance/Banking industry (SIC codes 6000-6799)
    "FINANCIAL": {
        "Revenue": {
            "primary": [
                "us-gaap:InterestAndDividendIncomeOperating",
                "us-gaap:NoninterestIncome",
                "us-gaap:InterestIncome"
            ],
            "fallback": [
                "us-gaap:RevenuesNetOfInterestExpense"
            ]
        },
        "Net Income": {
            "primary": [
                "us-gaap:NetIncomeLoss",
                "us-gaap:NetIncomeLossAvailableToCommonStockholdersBasic"
            ],
            "fallback": []
        }
    },
    
    # Oil & Gas industry (SIC codes 1300-1399, 2900-2999)
    "OIL_GAS": {
        "Revenue": {
            "primary": [
                "us-gaap:OilAndGasRevenue",
                "us-gaap:RevenueFromSaleOfCrudeOil",
                "us-gaap:RevenueFromSaleOfNaturalGas"
            ],
            "fallback": [
                "us-gaap:Revenues"
            ]
        }
    },
    
    # Technology industry (SIC codes 7370-7379, 3570-3579)
    "TECHNOLOGY": {
        "Revenue": {
            "primary": [
                "us-gaap:Revenues",
                "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
            ],
            "fallback": []
        }
    }
}

# SIC code to industry type mapping
SIC_TO_INDUSTRY = {
    # Financial industry
    **{str(sic): "FINANCIAL" for sic in range(6000, 6800)},
    
    # Oil & Gas industry
    **{str(sic): "OIL_GAS" for sic in list(range(1300, 1400)) + list(range(2900, 3000))},
    
    # Technology industry
    **{str(sic): "TECHNOLOGY" for sic in list(range(7370, 7380)) + list(range(3570, 3580))}
}

class SECDataProcessor:
    """
    Enhanced SEC EDGAR data processor that implements sophisticated strategies
    for extracting financial metrics from SEC filings.
    """
    
    def __init__(self, base_dir=BASE_DIR, filings_dir=FILINGS_DIR, output_dir=OUTPUT_DIR, max_workers=32):
        self.base_dir = Path(base_dir)
        self.filings_dir = Path(filings_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.max_workers = max_workers
        self.cik_to_ticker = {}
        self.ticker_to_cik = {}
        self.cik_to_name = {}
        self.cik_to_industry = {}
        self.company_data = {}  # Store processed company data
        
        # Track statistics
        self.stats = {
            "total_files": 0,
            "processed_files": 0,
            "error_files": 0,
            "metrics_found": defaultdict(int),
            "companies_processed": set(),
            "tag_usage": defaultdict(Counter)
        }
    
    def load_ticker_mapping(self, ticker_file):
        """Load CIK to ticker mappings from a parquet file"""
        try:
            logger.info(f"Loading ticker mapping from {ticker_file}")
            ticker_df = pd.read_parquet(ticker_file)
            
            for _, row in ticker_df.iterrows():
                cik = str(row['cik']).zfill(10)
                ticker = row['ticker']
                name = row.get('name', '')
                
                self.cik_to_ticker[cik] = ticker
                self.ticker_to_cik[ticker] = cik
                
                if name:
                    self.cik_to_name[cik] = name
            
            logger.info(f"Loaded {len(self.cik_to_ticker)} ticker-CIK mappings")
        except Exception as e:
            logger.error(f"Error loading ticker mapping: {e}")
            raise
    


    def extract_metadata(self, file_content, file_path):
        """
        Extract key metadata from the filing content to determine processing strategy.
        Enhanced with more flexible pattern matching.
        """
        metadata = {}

        # Extract CIK - more flexible pattern matching
        cik_patterns = [
            r'"cik"\s*:\s*"?(\d+)"?',
            r'"CIK"\s*:\s*"?(\d+)"?',
            r'CIK=(\d+)',
            r'CIK-(\d+)'
        ]

        for pattern in cik_patterns:
            cik_match = re.search(pattern, file_content)
            if cik_match:
                cik = cik_match.group(1).zfill(10)
                metadata['cik'] = cik
                metadata['ticker'] = self.cik_to_ticker.get(cik, "")
                metadata['name'] = self.cik_to_name.get(cik, "")
                break
            
        if 'cik' not in metadata:
            # Try to extract from filename
            cik_match = re.search(r'CIK(\d+)\.json', str(file_path))
            if cik_match:
                cik = cik_match.group(1).zfill(10)
                metadata['cik'] = cik
                metadata['ticker'] = self.cik_to_ticker.get(cik, "")
                metadata['name'] = self.cik_to_name.get(cik, "")

        # Extract SIC code - more flexible pattern matching
        sic_patterns = [
            r'"sic"\s*:\s*"?(\d+)"?', 
            r'"SIC"\s*:\s*"?(\d+)"?',
            r'SIC=(\d+)',
            r'SIC-(\d+)'
        ]

        for pattern in sic_patterns:
            sic_match = re.search(pattern, file_content)
            if sic_match:
                sic = sic_match.group(1)
                metadata['sic'] = sic
                metadata['industry'] = SIC_TO_INDUSTRY.get(sic, "OTHER")
                break
            
        # Extract form type - more flexible pattern matching
        form_patterns = [
            r'"form"\s*:\s*"([^"]+)"',
            r'"FORM"\s*:\s*"([^"]+)"',
            r'FORM=([^\s&]+)',
            r'FORM-([^\s&]+)'
        ]

        for pattern in form_patterns:
            form_match = re.search(pattern, file_content)
            if form_match:
                metadata['form'] = form_match.group(1)

                # Determine if 10-K or 10-Q
                form = form_match.group(1).upper()
                if "10-K" in form:
                    metadata['filing_type'] = "10-K"
                elif "10-Q" in form:
                    metadata['filing_type'] = "10-Q"
                else:
                    metadata['filing_type'] = form
                break
            
        # Extract period - more flexible pattern matching
        period_patterns = [
            r'"period"\s*:\s*"?(\d{8})"?',
            r'"PERIOD"\s*:\s*"?(\d{8})"?',
            r'PERIOD=(\d{8})',
            r'PERIOD-(\d{8})'
        ]

        for pattern in period_patterns:
            period_match = re.search(pattern, file_content)
            if period_match:
                period = period_match.group(1)
                metadata['period'] = period

                # Convert YYYYMMDD to YYYY-MM-DD format
                try:
                    year = period[:4]
                    month = period[4:6]
                    day = period[6:8]
                    metadata['period_formatted'] = f"{year}-{month}-{day}"
                except:
                    metadata['period_formatted'] = period
                break
            
        # If we still don't have a period, look for other date formats
        if 'period' not in metadata:
            # Try to find a date in YYYY-MM-DD format
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', file_content)
            if date_match:
                date_str = date_match.group(1)
                metadata['period_formatted'] = date_str
                # Convert to YYYYMMDD format
                try:
                    metadata['period'] = date_str.replace('-', '')
                except:
                    pass
                
        # Extract fiscal year
        fy_patterns = [
            r'"fy"\s*:\s*"?(\d{4})"?',
            r'"FY"\s*:\s*"?(\d{4})"?',
            r'FY=(\d{4})',
            r'FY-(\d{4})'
        ]

        for pattern in fy_patterns:
            fy_match = re.search(pattern, file_content)
            if fy_match:
                metadata['fiscal_year'] = fy_match.group(1)
                break
            
        # If no fiscal year found, try to extract from period
        if 'fiscal_year' not in metadata and 'period' in metadata:
            try:
                metadata['fiscal_year'] = metadata['period'][:4]
            except:
                pass
            
        # Extract fiscal period focus (Q1, Q2, Q3, Q4, FY)
        fp_patterns = [
            r'"fp"\s*:\s*"([^"]+)"',
            r'"FP"\s*:\s*"([^"]+)"',
            r'FP=([^\s&]+)',
            r'FP-([^\s&]+)'
        ]

        for pattern in fp_patterns:
            fp_match = re.search(pattern, file_content)
            if fp_match:
                metadata['fiscal_period'] = fp_match.group(1)
                break
            
        # If fiscal period not found, try to determine from form type
        if 'fiscal_period' not in metadata and 'form' in metadata:
            form = metadata['form'].upper()
            if '10-K' in form:
                metadata['fiscal_period'] = 'FY'
            elif '10-Q' in form:
                # Can't determine which quarter without more info
                metadata['fiscal_period'] = 'QX'

        # Generate period_id if possible
        if 'fiscal_year' in metadata:
            if metadata.get('fiscal_period') in ('Q1', 'Q2', 'Q3', 'Q4', 'QX'):
                metadata['period_id'] = f"{metadata['fiscal_year']}-{metadata['fiscal_period']}"
            elif metadata.get('fiscal_period') == 'FY':
                metadata['period_id'] = metadata['fiscal_year']
        elif 'period' in metadata:
            # Use the period as a fallback for period_id
            metadata['period_id'] = metadata['period']

        # Special case: If we have both period and filing_type but no period_id
        if 'period_id' not in metadata and 'period' in metadata and 'filing_type' in metadata:
            metadata['period_id'] = metadata['period']

        # If we've gotten this far and still don't have filing_type and period, use defaults
        # This is a fallback to ensure processing continues
        if not metadata.get('filing_type') and metadata.get('form'):
            # Default to the form value if a specific type couldn't be determined
            metadata['filing_type'] = metadata['form']

        if not metadata.get('period') and metadata.get('period_formatted'):
            # Use the formatted period if the raw period isn't available
            metadata['period'] = metadata['period_formatted'].replace('-', '')

        return metadata
    
    def determine_tag_mapping(self, metadata):
        """
        Determine the appropriate tag mapping based on metadata.
        """
        tag_mapping = {}
        
        # Start with standard tag mappings
        for metric, mapping in METRIC_TAG_MAPPING.items():
            tag_mapping[metric] = {
                "primary": mapping["primary"].copy(),
                "fallback": mapping["fallback"].copy()
            }
        
        # Apply industry-specific mappings if available
        industry = metadata.get('industry')
        if industry and industry in INDUSTRY_TAG_MAPPINGS:
            industry_mappings = INDUSTRY_TAG_MAPPINGS[industry]
            
            for metric, mapping in industry_mappings.items():
                if metric in tag_mapping:
                    # Add industry-specific primary tags to the front of the list
                    tag_mapping[metric]["primary"] = mapping["primary"] + tag_mapping[metric]["primary"]
                    tag_mapping[metric]["fallback"] = mapping["fallback"] + tag_mapping[metric]["fallback"]
                else:
                    tag_mapping[metric] = mapping
        
        return tag_mapping
    
    def find_tag_recursive(self, data, tag_pattern, results=None):
        """
        Recursively search for tags matching a pattern in nested data structures.
        """
        if results is None:
            results = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                if re.match(tag_pattern, key):
                    results.append((key, value))
                if isinstance(value, (dict, list)):
                    self.find_tag_recursive(value, tag_pattern, results)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    self.find_tag_recursive(item, tag_pattern, results)
        
        return results
    
    def extract_value_for_tag(self, json_data, tag, fallback_tags=None):
        """
        Extract value for a specific tag with fallback options.
        """
        # Try exact match first
        if tag in json_data:
            return json_data[tag], tag
        
        # Try to find it recursively
        tag_results = self.find_tag_recursive(json_data, f"^{re.escape(tag)}$")
        if tag_results:
            return tag_results[0][1], tag_results[0][0]
        
        # Try fallback tags if provided
        if fallback_tags:
            for fallback_tag in fallback_tags:
                if fallback_tag in json_data:
                    return json_data[fallback_tag], fallback_tag
                
                # Try to find fallback recursively
                fallback_results = self.find_tag_recursive(json_data, f"^{re.escape(fallback_tag)}$")
                if fallback_results:
                    return fallback_results[0][1], fallback_results[0][0]
        
        # Try semantic matching as last resort
        potential_matches = self.find_semantically_similar_tags(json_data, tag, 0.85)
        if potential_matches:
            best_match_tag = potential_matches[0][0]
            return json_data.get(best_match_tag), best_match_tag
        
        return None, None
    
    def find_semantically_similar_tags(self, json_data, target_tag, similarity_threshold=0.7):
        """
        Find tags in the data that are semantically similar to the target tag.
        """
        all_tags = []
        for key in json_data.keys():
            if isinstance(key, str):
                all_tags.append(key)
        
        # Also search for nested tags
        for tag_tuple in self.find_tag_recursive(json_data, r".*"):
            if isinstance(tag_tuple[0], str):
                all_tags.append(tag_tuple[0])
        
        potential_matches = []
        target_tag_simple = target_tag.split(':')[-1].lower()
        
        for tag in all_tags:
            tag_simple = tag.split(':')[-1].lower()
            similarity = difflib.SequenceMatcher(None, target_tag_simple, tag_simple).ratio()
            if similarity >= similarity_threshold:
                potential_matches.append((tag, similarity))
        
        return sorted(potential_matches, key=lambda x: x[1], reverse=True)
    
    def extract_value_from_fact(self, fact, filing_type):
        """
        Extract the appropriate value from a fact structure.
        """
        if not fact:
            return None
        
        # Handle various fact structures
        if isinstance(fact, dict):
            # Direct value access
            if 'val' in fact:
                return fact['val']
            
            # Handle units section
            if 'units' in fact:
                units = fact['units']
                
                # Try USD for monetary values
                if 'USD' in units and units['USD']:
                    values = units['USD']
                    for value_obj in values:
                        # Filter for the right form type
                        if 'form' in value_obj and value_obj['form'] in (filing_type, '10-K', '10-Q'):
                            if 'val' in value_obj:
                                return value_obj['val']
                    
                    # If no match by form, take the first value
                    if values and 'val' in values[0]:
                        return values[0]['val']
                
                # Try shares for share counts
                if 'shares' in units and units['shares']:
                    values = units['shares']
                    for value_obj in values:
                        if 'form' in value_obj and value_obj['form'] in (filing_type, '10-K', '10-Q'):
                            if 'val' in value_obj:
                                return value_obj['val']
                    
                    if values and 'val' in values[0]:
                        return values[0]['val']
                
                # Try USD/shares for per-share values
                if 'USD/shares' in units and units['USD/shares']:
                    values = units['USD/shares']
                    for value_obj in values:
                        if 'form' in value_obj and value_obj['form'] in (filing_type, '10-K', '10-Q'):
                            if 'val' in value_obj:
                                return value_obj['val']
                    
                    if values and 'val' in values[0]:
                        return values[0]['val']
                
                # Try pure/xbrl for ratios
                if 'pure' in units and units['pure']:
                    values = units['pure']
                    for value_obj in values:
                        if 'form' in value_obj and value_obj['form'] in (filing_type, '10-K', '10-Q'):
                            if 'val' in value_obj:
                                return value_obj['val']
                    
                    if values and 'val' in values[0]:
                        return values[0]['val']
        
        # Handle simple value
        return fact
    
    def extract_metrics(self, json_data, metadata, tag_mapping):
        """
        Extract financial metrics from JSON data using the provided tag mapping.
        """
        extracted_metrics = {}
        tag_sources = {}
        
        # Process US GAAP facts first
        facts = json_data.get('facts', {})
        us_gaap = facts.get('us-gaap', {})
        
        # Extract each metric using appropriate tags
        for metric, tags in tag_mapping.items():
            primary_tags = tags['primary']
            fallback_tags = tags['fallback']
            
            for tag in primary_tags:
                tag_base = tag.split(':')[-1]  # Remove namespace prefix
                
                if tag_base in us_gaap:
                    value = self.extract_value_from_fact(us_gaap[tag_base], metadata.get('filing_type', ''))
                    
                    # Convert to float if possible
                    try:
                        if value is not None:
                            extracted_metrics[metric] = float(value)
                            tag_sources[metric] = tag
                            
                            # Update tag usage statistics
                            self.stats["tag_usage"][metric][tag] += 1
                            self.stats["metrics_found"][metric] += 1
                            break
                    except (ValueError, TypeError):
                        continue
            
            # If not found with primary tags, try fallback tags
            if metric not in extracted_metrics and fallback_tags:
                for tag in fallback_tags:
                    tag_base = tag.split(':')[-1]
                    
                    if tag_base in us_gaap:
                        value = self.extract_value_from_fact(us_gaap[tag_base], metadata.get('filing_type', ''))
                        
                        try:
                            if value is not None:
                                extracted_metrics[metric] = float(value)
                                tag_sources[metric] = tag
                                
                                # Update tag usage statistics
                                self.stats["tag_usage"][metric][tag] += 1
                                self.stats["metrics_found"][metric] += 1
                                break
                        except (ValueError, TypeError):
                            continue
        
        # Calculate derived ratio metrics
        self.derive_ratio_metrics(extracted_metrics)
        
        return extracted_metrics, tag_sources
    
    def derive_ratio_metrics(self, metrics):
        """
        Calculate derived ratio metrics from available metrics.
        """
        for ratio, details in RATIO_METRICS.items():
            # Check if all required metrics are available
            if all(m in metrics for m in details["required_metrics"]):
                try:
                    ratio_value = details["formula"](metrics)
                    if ratio_value is not None:
                        metrics[ratio] = ratio_value
                except Exception as e:
                    logger.debug(f"Error calculating {ratio}: {e}")
    
    def derive_missing_metrics(self, metrics):
        """
        Attempt to derive missing metrics using correlations.
        """
        derived_metrics = {}
        
        # Try to derive each missing metric using correlations
        for target_metric, correlations in METRIC_CORRELATIONS.items():
            # Skip if target already exists
            if target_metric in metrics:
                continue
                
            # Try each correlated metric
            for source_metric, correlation in correlations.items():
                if source_metric in metrics:
                    derived_metrics[target_metric] = {
                        "value": metrics[source_metric],
                        "source": source_metric,
                        "correlation": correlation,
                        "confidence": correlation
                    }
                    break
        
        # Special case calculations
        if "Gross Profit" not in metrics and "COGS" in metrics and "Revenue" in metrics:
            derived_metrics["Gross Profit"] = {
                "value": metrics["Revenue"] - metrics["COGS"],
                "source": "calculated",
                "correlation": 0.95,
                "confidence": 0.95
            }
        
        if "Net Change in Cash" not in metrics and all(m in metrics for m in ["Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow"]):
            derived_metrics["Net Change in Cash"] = {
                "value": metrics["Operating Cash Flow"] + metrics["Investing Cash Flow"] + metrics["Financing Cash Flow"],
                "source": "calculated",
                "correlation": 0.95,
                "confidence": 0.95
            }
        
        # Add derived metrics to the original set
        for metric, details in derived_metrics.items():
            metrics[metric] = details["value"]
        
        return metrics, derived_metrics
    

    def process_filing(self, file_path):
        """
        Process a single SEC filing file.
        """
        try:
            # Extract CIK from filename to check if we should process this file
            cik_match = re.search(r'CIK(\d+)\.json', str(file_path))
            if not cik_match:
                logger.debug(f"Skipping {file_path}: No CIK in filename")
                return None

            cik = cik_match.group(1).zfill(10)
            if cik not in self.cik_to_ticker:
                logger.debug(f"Skipping {file_path}: CIK {cik} not in ticker mapping")
                return None

            # First pass - read a sample to extract metadata
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    sample_content = f.read(50000)  # Read enough to extract metadata
            except Exception as e:
                logger.debug(f"Error reading file {file_path}: {e}")
                return None

            # Extract metadata
            metadata = self.extract_metadata(sample_content, file_path)
            logger.debug(f"Extracted metadata from {file_path}: {metadata}")

            # Skip if we couldn't determine key metadata
            if not metadata.get('filing_type'):
                logger.debug(f"Skipping {file_path}: No filing_type in metadata")
                return None

            if not metadata.get('period'):
                logger.debug(f"Skipping {file_path}: No period in metadata")
                return None
            
            # Determine appropriate tag mapping
            tag_mapping = self.determine_tag_mapping(metadata)
            
            # Second pass - load full data for processing
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        full_data = json.load(f)
                    except json.JSONDecodeError as e:
                        logger.debug(f"JSON decode error: {e}. Trying alternate approach...")
                        # If file too large or has issues, use a different approach
                        f.seek(0)  # Go back to beginning of file
                        content = f.read()
                        
                        # Look for facts section using more flexible pattern
                        facts_patterns = [
                            r'"facts"\s*:\s*({.*})',
                            r'"FACTS"\s*:\s*({.*})',
                            r'"Facts"\s*:\s*({.*})'
                        ]
                        
                        facts_found = False
                        for pattern in facts_patterns:
                            facts_match = re.search(pattern, content, re.DOTALL)
                            if facts_match:
                                facts_str = facts_match.group(1)
                                try:
                                    facts = json.loads(facts_str)
                                    full_data = {"facts": facts}
                                    facts_found = True
                                    break
                                except json.JSONDecodeError:
                                    continue
                                
                        if not facts_found:
                            # If still can't extract facts, try to find US-GAAP data directly
                            us_gaap_match = re.search(r'"us-gaap"\s*:\s*({.*?})', content, re.DOTALL)
                            if us_gaap_match:
                                try:
                                    us_gaap_str = us_gaap_match.group(1)
                                    us_gaap_data = json.loads(us_gaap_str)
                                    full_data = {"facts": {"us-gaap": us_gaap_data}}
                                except:
                                    logger.debug(f"Failed to parse us-gaap section from {file_path}")
                                    return {
                                        "cik": cik,
                                        "ticker": metadata.get('ticker', ''),
                                        "name": metadata.get('name', ''),
                                        "metadata": metadata,
                                        "metrics": {},
                                        "error": "Failed to parse us-gaap section"
                                    }
                            else:
                                logger.debug(f"No facts or us-gaap section found in {file_path}")
                                return {
                                    "cik": cik,
                                    "ticker": metadata.get('ticker', ''),
                                    "name": metadata.get('name', ''),
                                    "metadata": metadata,
                                    "metrics": {},
                                    "error": "No facts section found"
                                }
            except Exception as e:
                logger.error(f"Error reading or parsing {file_path}: {e}")
                return {
                    "cik": cik,
                    "ticker": metadata.get('ticker', ''),
                    "name": metadata.get('name', ''),
                    "metadata": metadata,
                    "metrics": {},
                    "error": f"File reading error: {str(e)}"
                }
            

            metrics, tag_sources = self.extract_metrics(full_data, metadata, tag_mapping)
            
            # Attempt to derive missing metrics
            metrics, derived_metrics = self.derive_missing_metrics(metrics)
            
            # Add metadata to metrics
            if 'period_id' in metadata:
                period_id = metadata['period_id']
            elif 'period_formatted' in metadata:
                period_id = metadata['period_formatted']
            else:
                period_id = metadata['period']
            
            # Format output
            result = {
                "cik": cik,
                "ticker": metadata.get('ticker', ''),
                "name": metadata.get('name', ''),
                "metadata": metadata,
                "period_id": period_id,
                "metrics": metrics,
                "tag_sources": tag_sources,
                "derived_metrics": derived_metrics
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return {
                "cik": cik if 'cik' in locals() else None,
                "error": str(e),
                "file_path": str(file_path)
            }
    
    def update_company_data(self, filing_result):
        """
        Update company data with results from a processed filing.
        """
        if not filing_result or "error" in filing_result and not filing_result.get("metrics"):
            return False
            
        cik = filing_result["cik"]
        ticker = filing_result["ticker"]
        period_id = filing_result["period_id"]
        
        # Initialize company data if not exists
        if cik not in self.company_data:
            self.company_data[cik] = {
                "cik": cik,
                "ticker": ticker,
                "name": filing_result["name"],
                "periods": set(),
                "metrics": defaultdict(dict),
                "filing_dates": {},
                "data_sources": defaultdict(dict)
            }
        
        # Add period
        self.company_data[cik]["periods"].add(period_id)
        
        # Add filing date
        if "filing_date" in filing_result["metadata"]:
            self.company_data[cik]["filing_dates"][period_id] = filing_result["metadata"]["filing_date"]
        
        # Add metrics
        for metric, value in filing_result["metrics"].items():
            self.company_data[cik]["metrics"][metric][period_id] = value
            
            # Add data source information
            if "tag_sources" in filing_result and metric in filing_result["tag_sources"]:
                self.company_data[cik]["data_sources"][metric][period_id] = {
                    "source": "direct",
                    "tag": filing_result["tag_sources"][metric]
                }
            elif "derived_metrics" in filing_result and metric in filing_result["derived_metrics"]:
                derived_info = filing_result["derived_metrics"][metric]
                self.company_data[cik]["data_sources"][metric][period_id] = {
                    "source": "derived",
                    "method": derived_info.get("source", "unknown"),
                    "confidence": derived_info.get("confidence", 0.5)
                }
        
        # Track statistics
        self.stats["companies_processed"].add(cik)
        
        return True
    







    def process_filings_in_batches(self, batch_size=1000):
        """
        Process all filing files in batches with improved error handling.
        """
        logger.info("Starting batch processing of filings")

        # Get list of all filing files
        all_files = []
        for file_path in self.filings_dir.glob("CIK*.json"):
            all_files.append(file_path)

        self.stats["total_files"] = len(all_files)
        logger.info(f"Found {len(all_files)} filing files to process")

        # Add debugging for a sample file to understand structure
        if all_files:
            sample_file = all_files[0]
            logger.info(f"Examining sample file: {sample_file}")
            try:
                with open(sample_file, 'r', encoding='utf-8') as f:
                    sample_content = f.read(1000)  # Read first 1000 chars
                logger.info(f"Sample file content (first 1000 chars): {sample_content[:1000]}")
            except Exception as e:
                logger.error(f"Error reading sample file: {e}")

        # Process in batches
        total_batches = (len(all_files) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(all_files))
            batch_files = all_files[start_idx:end_idx]

            logger.info(f"Processing batch {batch_idx+1}/{total_batches} ({len(batch_files)} files)")

            # Process batch in parallel
            results = []
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {executor.submit(self.process_filing, str(file_path)): file_path 
                                 for file_path in batch_files}

                for future in tqdm(as_completed(future_to_file), total=len(batch_files), 
                                  desc=f"Batch {batch_idx+1}/{total_batches}", unit="file"):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        if result:
                            if "error" in result and not result.get("metrics"):
                                self.stats["error_files"] += 1
                            else:
                                self.stats["processed_files"] += 1
                                results.append(result)
                    except Exception as e:
                        self.stats["error_files"] += 1
                        logger.error(f"Error processing {file_path}: {e}")

            # Update company data with batch results
            for result in results:
                self.update_company_data(result)

            # Free up memory
            results = None
            gc.collect()

            logger.info(f"Completed batch {batch_idx+1}/{total_batches}. "
                       f"Processed: {self.stats['processed_files']}, "
                       f"Errors: {self.stats['error_files']}")

            # Stop after first batch if no files were processed
            if batch_idx == 0 and self.stats['processed_files'] == 0:
                logger.warning("No files processed in first batch. Checking file structure issues...")
                # Try to process one file directly to debug
                if batch_files:
                    debug_file = str(batch_files[0])
                    logger.info(f"Debugging file: {debug_file}")
                    try:
                        with open(debug_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        logger.info(f"File size: {len(content)} bytes")
                        logger.info(f"First 500 characters: {content[:500]}")

                        # Check for key patterns
                        cik_match = re.search(r'"cik"\s*:\s*"?(\d+)"?', content)
                        form_match = re.search(r'"form"\s*:\s*"([^"]+)"', content)
                        period_match = re.search(r'"period"\s*:\s*"?(\d{8})"?', content)

                        logger.info(f"CIK match: {cik_match.group(1) if cik_match else 'Not found'}")
                        logger.info(f"Form match: {form_match.group(1) if form_match else 'Not found'}")
                        logger.info(f"Period match: {period_match.group(1) if period_match else 'Not found'}")

                        # Try to load as JSON to check structure
                        try:
                            data = json.loads(content)
                            logger.info(f"Top-level keys: {list(data.keys())}")
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON parsing error: {e}")
                    except Exception as e:
                        logger.error(f"Debug file error: {e}")

        logger.info(f"Completed all batches. Total files processed: {self.stats['processed_files']}")
        return self.stats["processed_files"]
    
    def save_company_data(self):
        """
        Save processed company data to parquet files.
        """
        logger.info(f"Saving data for {len(self.company_data)} companies")
        
        companies_saved = 0
        
        for cik, company in tqdm(self.company_data.items(), desc="Saving company data"):
            try:
                ticker = company["ticker"]
                if not ticker:
                    continue
                
                # Convert company data to DataFrame
                rows = []
                periods = sorted(list(company["periods"]))
                
                for period in periods:
                    row = {
                        "Period": period,
                        "FilingDate": company["filing_dates"].get(period, "")
                    }
                    
                    # Add metrics
                    for metric, values in company["metrics"].items():
                        if period in values:
                            row[metric] = values[period]
                    
                    rows.append(row)
                
                # Create DataFrame and save
                if rows:
                    df = pd.DataFrame(rows)
                    output_file = self.output_dir / f"{ticker}.parquet"
                    df.to_parquet(output_file)
                    companies_saved += 1
            except Exception as e:
                logger.error(f"Error saving data for {company['ticker']}: {e}")
        
        logger.info(f"Saved data for {companies_saved} companies")
        return companies_saved
    
    def create_consolidated_metrics(self):
        """
        Create consolidated metrics files from all company data.
        """
        logger.info("Creating consolidated metrics files")
        
        # Annual metrics (most recent year for each company)
        annual_metrics = []
        quarterly_metrics = []
        
        for cik, company in self.company_data.items():
            ticker = company["ticker"]
            name = company["name"]
            periods = company["periods"]
            
            # Separate annual and quarterly periods
            annual_periods = sorted([p for p in periods if '-Q' not in p and len(p) == 4])
            quarterly_periods = sorted([p for p in periods if '-Q' in p])
            
            # Get most recent annual data
            if annual_periods:
                latest_year = annual_periods[-1]
                annual_row = {
                    "Ticker": ticker,
                    "Name": name,
                    "CIK": cik,
                    "Period": latest_year,
                    "FilingDate": company["filing_dates"].get(latest_year, "")
                }
                
                # Add all metrics for this year
                for metric, values in company["metrics"].items():
                    if latest_year in values:
                        annual_row[metric] = values[latest_year]
                
                annual_metrics.append(annual_row)
            
            # Get most recent quarterly data
            if quarterly_periods:
                latest_quarter = quarterly_periods[-1]
                quarterly_row = {
                    "Ticker": ticker,
                    "Name": name,
                    "CIK": cik,
                    "Period": latest_quarter,
                    "FilingDate": company["filing_dates"].get(latest_quarter, "")
                }
                
                # Add all metrics for this quarter
                for metric, values in company["metrics"].items():
                    if latest_quarter in values:
                        quarterly_row[metric] = values[latest_quarter]
                
                quarterly_metrics.append(quarterly_row)
        
        # Save consolidated metrics
        if annual_metrics:
            annual_df = pd.DataFrame(annual_metrics)
            annual_df.to_parquet(METRICS_DIR / "annual_metrics.parquet")
            logger.info(f"Saved annual metrics for {len(annual_metrics)} companies")
        
        if quarterly_metrics:
            quarterly_df = pd.DataFrame(quarterly_metrics)
            quarterly_df.to_parquet(METRICS_DIR / "quarterly_metrics.parquet")
            logger.info(f"Saved quarterly metrics for {len(quarterly_metrics)} companies")
        
        # Save metrics availability statistics
        self.save_metrics_statistics()
        
        return len(annual_metrics), len(quarterly_metrics)
    
    def save_metrics_statistics(self):
        """
        Save statistics about metrics availability to a JSON file.
        """
        # Calculate metrics availability
        metrics_stats = {}
        total_companies = len(self.stats["companies_processed"])
        
        for metric, count in self.stats["metrics_found"].items():
            metrics_stats[metric] = {
                "count": count,
                "availability": count / total_companies if total_companies > 0 else 0,
                "top_tags": self.stats["tag_usage"][metric].most_common(5)
            }
        
        # Save to JSON
        with open(METRICS_DIR / "metrics_statistics.json", "w") as f:
            json.dump({
                "total_files": self.stats["total_files"],
                "processed_files": self.stats["processed_files"],
                "error_files": self.stats["error_files"],
                "companies_processed": len(self.stats["companies_processed"]),
                "metrics_stats": metrics_stats
            }, f, indent=2)
        
        logger.info("Saved metrics statistics")
    
    def process_all_data(self, ticker_file, batch_size=1000):
        """
        Complete pipeline to process all SEC filing data.
        """
        start_time = time.time()
        logger.info("Starting SEC data processing pipeline")
        
        # Load ticker mapping
        self.load_ticker_mapping(ticker_file)
        
        # Process filings in batches
        files_processed = self.process_filings_in_batches(batch_size)
        
        # Save company data
        companies_saved = self.save_company_data()
        
        # Create consolidated metrics
        annual_count, quarterly_count = self.create_consolidated_metrics()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Completed SEC data processing in {elapsed_time:.2f} seconds")
        logger.info(f"Files processed: {files_processed}")
        logger.info(f"Companies saved: {companies_saved}")
        logger.info(f"Annual metrics: {annual_count}, Quarterly metrics: {quarterly_count}")
        
        # Output processing summary
        print("\n===== SEC Data Processing Summary =====")
        print(f"Total files found: {self.stats['total_files']}")
        print(f"Files processed successfully: {self.stats['processed_files']} ({self.stats['processed_files']/self.stats['total_files']*100:.1f}%)")
        print(f"Files with errors: {self.stats['error_files']} ({self.stats['error_files']/self.stats['total_files']*100:.1f}%)")
        print(f"Companies processed: {len(self.stats['companies_processed'])}")
        print(f"Companies saved: {companies_saved}")
        print("")
        print("Top 10 most available metrics:")
        
        # Sort metrics by availability
        sorted_metrics = sorted(
            [(m, c) for m, c in self.stats["metrics_found"].items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        for metric, count in sorted_metrics[:10]:
            availability = count / len(self.stats["companies_processed"]) if self.stats["companies_processed"] else 0
            print(f"  {metric}: {count} companies ({availability*100:.1f}%)")
        
        print(f"\nProcessing time: {elapsed_time:.2f} seconds")
        print("=======================================\n")
        
        return {
            "files_processed": files_processed,
            "companies_saved": companies_saved,
            "annual_metrics": annual_count,
            "quarterly_metrics": quarterly_count,
            "processing_time": elapsed_time
        }




def main():
    """
    Main entry point for SEC data processing with improved debugging.
    """
    # Configure more verbose logging for debugging
    logging.basicConfig(level=logging.DEBUG)
    
    # Log important directory information
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Data directory exists: {os.path.exists('Data')}")
    logger.info(f"Filings directory exists: {os.path.exists('Data/Filings')}")
    
    # Check ticker file exists
    ticker_file = "Data/TickerCikData/TickerCIKs_20250319.parquet"
    logger.info(f"Ticker file exists: {os.path.exists(ticker_file)}")
    
    # Check for file counts
    try:
        filings_dir = Path("Data/Filings")
        file_count = len(list(filings_dir.glob("CIK*.json")))
        logger.info(f"Found {file_count} CIK*.json files in Filings directory")
        
        # Log first few filenames to check format
        files = list(filings_dir.glob("CIK*.json"))[:5]
        for f in files:
            logger.info(f"Sample filename: {f}")
    except Exception as e:
        logger.error(f"Error checking files: {e}")
    
    # Create processor instance
    processor = SECDataProcessor(
        base_dir="Data",
        filings_dir="Data/Filings",
        output_dir="Data/SEC_Data/Enhanced",
        max_workers=32  # Adjust based on your CPU
    )
    
    # Process all data
    result = processor.process_all_data(
        ticker_file=ticker_file,
        batch_size=1000
    )
    
    return result

if __name__ == "__main__":
    main()