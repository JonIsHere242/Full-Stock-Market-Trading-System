#!/usr/bin/env python3

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import Counter, defaultdict

# NLP libraries
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import DBSCAN
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False






from Util import get_logger

logger = get_logger(script_name="Z-NLP")

DATA_DIR = Path("Data")
NEWS_DIR = DATA_DIR / "News"
OUTPUT_DIR = NEWS_DIR / "Analysis"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)



class NewsAnalyzer:

    def __init__(self, ticker_file=None, portfolio_file=None):
        self.tickers = self._load_ticker_data(ticker_file)
        self.ticker_patterns = {}
        self.portfolio = self._load_portfolio(portfolio_file)

        self._setup_risk_terms()

        self.nlp_ready = False

        if NLTK_AVAILABLE:
            try:
                # Download all required NLTK resources
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)


                nltk.download('punkt', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
                # Add the missing punkt model specifically
                import ssl
                try:
                    _create_unverified_https_context = ssl._create_unverified_context
                except AttributeError:
                    pass
                else:
                    ssl._create_default_https_context = _create_unverified_https_context

                # Initialize NLTK components after downloads
                self.stop_words = set(stopwords.words('english'))
                self.lemmatizer = WordNetLemmatizer()
                self.nlp_ready = True
            except Exception as e:
                logger.error(f"Error initializing NLTK: {e}")
                self.nlp_ready = False
        else:
            logger.warning("NLTK not available, text processing will be limited")




    def _load_ticker_data(self, ticker_file=None):
        if ticker_file is not None and Path(ticker_file).exists():
            return pd.read_parquet(ticker_file)
        
        ticker_dir = DATA_DIR / "TickerCikData"
        if ticker_dir.exists():
            ticker_files = list(ticker_dir.glob("TickerCIKs_*.parquet"))
            if ticker_files:
                latest_file = max(ticker_files, key=lambda x: x.stat().st_mtime)
                logger.info(f"Using ticker file: {latest_file}")
                return pd.read_parquet(latest_file)
            
            backup_file = ticker_dir / "BackupCIK.parquet"
            if backup_file.exists():
                return pd.read_parquet(backup_file)
        
        logger.warning("No ticker data file found")
        return pd.DataFrame(columns=["ticker", "name"])
    
    def _load_portfolio(self, portfolio_file=None):
        portfolio = set()
        
        if portfolio_file is not None and Path(portfolio_file).exists():
            try:
                df = pd.read_parquet(portfolio_file)
                if 'Ticker' in df.columns:
                    portfolio = set(df['Ticker'].unique())
                    return portfolio
            except Exception as e:
                logger.error(f"Error loading portfolio file: {e}")
        
        positions_file = Path("_Live_trades.parquet")
        if positions_file.exists():
            try:
                df = pd.read_parquet(positions_file)
                if 'Ticker' in df.columns:
                    portfolio = set(df['Ticker'].unique())
                    return portfolio
            except Exception as e:
                logger.error(f"Error loading live trades file: {e}")
        
        return portfolio
    


    def _setup_risk_terms(self):
        # Sentiment terms (keep these separate)
        self.positive_terms = set([
            "growth", "profit", "revenue", "earnings", "outperform", "beat", "upgrade",
            "bullish", "exceeded", "positive", "strong", "surge", "rally", "gain",
            "upside", "opportunity", "record", "momentum", "expand", "boost"
        ])

        self.negative_terms = set([
            "loss", "debt", "downgrade", "missed", "bearish", "decline", "fall", "drop",
            "weak", "risk", "concern", "downside", "sell-off", "underperform", "bankruptcy",
            "investigation", "lawsuit", "warning", "layoff", "cut"
        ])

        # Initialize risk structures
        self.risk_terms = set()
        self.risk_category_map = {}
        self._setup_enhanced_risk_categories()
    
    def _setup_enhanced_risk_categories(self):
        """Centralized configuration for all risk categories"""
        category_definitions = [
            (
                "Financial Crisis", 
                [
                    "default", "bankruptcy", "liquidity crisis", "credit crunch", "bank run",
                    "financial collapse", "market crash", "bear market", "margin call", "debt crisis",
                    "collapse", "crash", "crisis", "shock", "burst", "bubble"
                ]
            ),
            (
                "Geopolitical Risk", 
                [
                    "war", "attack", "invasion", "military conflict", "coup", "terrorism", "sanctions",
                    "embargo", "trade war", "cyberattack", "assassination", "political unrest", "riot",
                    "revolution", "border clash", "nuclear threat", "espionage", "diplomatic crisis"
                ]
            ),
            (
                "Natural Disaster", 
                [
                    "earthquake", "tsunami", "hurricane", "typhoon", "tornado", "flood",
                    "wildfire", "drought", "volcanic eruption", "landslide", "blizzard",
                    "disaster", "catastrophe", "extreme weather", "storm surge", "avalanche"
                ]
            ),
            (
                "Health Crisis", 
                [
                    "pandemic", "epidemic", "outbreak", "virus", "contagion", "quarantine",
                    "lockdown", "health emergency", "disease spread", "infection rate",
                    "biosecurity threat", "vaccine shortage", "mutation", "superbug"
                ]
            ),
            (
                "Supply Chain Disruption", 
                [
                    "supply chain disruption", "logistics breakdown", "shipping delay", "port congestion",
                    "container shortage", "production halt", "factory shutdown", "inventory shortage",
                    "labor shortage", "raw material shortage", "export ban", "import restriction"
                ]
            ),
            (
                "Energy Crisis", 
                [
                    "energy crisis", "oil shock", "fuel shortage", "power outage", "blackout",
                    "energy rationing", "grid failure", "gas shortage", "pipeline disruption",
                    "OPEC cut", "refinery fire", "energy sanctions", "carbon tax", "energy price cap"
                ]
            ),
            (
                "Regulatory/Legal Risk", 
                [
                    "regulation", "legislation", "lawsuit", "legal action", "investigation",
                    "fine", "penalty", "compliance", "antitrust", "class action", "fraud",
                    "indictment", "subpoena", "whistleblower", "regulatory overhaul"
                ]
            ),
            (
                "Technology Risk",
                [
                    "data breach", "ransomware", "system outage", "cloud failure", "API outage",
                    "encryption flaw", "zero-day exploit", "IT infrastructure failure",
                    "technology obsolescence", "semiconductor shortage", "patent litigation"
                ]
            ),
            (
                "Market Structure Risk",
                [
                    "flash crash", "liquidity drought", "high-frequency trading glitch",
                    "circuit breaker", "market manipulation", "dark pool issue",
                    "settlement failure", "short squeeze", "naked short selling"
                ]
            )
        ]
        

        self.risk_category_map = {}
        # Create category attributes and build structures
        for category_name, terms in category_definitions:
            # Store terms as instance variable (optional)
            attr_name = f"{category_name.lower().replace(' ', '_').replace('/', '_')}_terms"
            setattr(self, attr_name, set(terms))

            # Update risk terms and category map
            self.risk_terms.update(terms)
            self.risk_category_map.update({term: category_name for term in terms})
    




    def load_news_data(self, days_back=7):
        latest_file = NEWS_DIR / "latest_financial_news.parquet"
        if latest_file.exists():
            try:
                df = pd.read_parquet(latest_file)
                logger.info(f"Loaded {len(df)} articles from latest news file")
                return df
            except Exception as e:
                logger.error(f"Error loading latest news file: {e}")
        
        news_files = list(NEWS_DIR.glob("financial_news_*.parquet"))
        if not news_files:
            db_dir = Path("database")
            if db_dir.exists():
                parquet_files = list(db_dir.glob("articles_*.parquet"))
                if parquet_files:
                    news_files = parquet_files
                else:
                    csv_files = list(db_dir.glob("articles_*.csv"))
                    if csv_files:
                        latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
                        try:
                            df = pd.read_csv(latest_csv)
                            return df
                        except Exception as e:
                            logger.error(f"Error loading CSV file: {e}")
        
        if not news_files:
            logger.error("No news files found")
            return pd.DataFrame()
        
        latest_file = max(news_files, key=lambda x: x.stat().st_mtime)
        
        try:
            df = pd.read_parquet(latest_file)
            return df
        except Exception as e:
            logger.error(f"Error loading news file: {e}")
            json_files = list(NEWS_DIR.glob("financial_news_*.json"))
            if json_files:
                latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
                try:
                    with open(latest_json, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    df = pd.DataFrame(data)
                    return df
                except Exception as e:
                    logger.error(f"Error loading JSON file: {e}")
        
        return pd.DataFrame()
    


    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""

        # If NLP is not available or we've had issues, use this simplified approach
        if not self.nlp_ready:
            return text.lower()

        try:
            # Custom basic tokenization that doesn't rely on punkt_tab
            simple_tokens = text.lower().split()
            # Still filter stopwords and apply lemmatization if available
            filtered_tokens = [word for word in simple_tokens if word.isalpha() and word not in self.stop_words]
            lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in filtered_tokens]
            return " ".join(lemmatized_tokens)
        except Exception as e:
            logger.warning(f"Error in text preprocessing: {e}")
            # Very basic fallback - just lowercase and split
            return text.lower()
    




    def calculate_sentiment(self, news_df):
        if news_df.empty:
            return news_df
        
        news_df['full_text'] = news_df['title'].fillna('') + ' ' + news_df['description'].fillna('')
        sentiments = []
        
        for _, row in news_df.iterrows():
            text = row['full_text'].lower()
            
            pos_count = sum(1 for term in self.positive_terms if term in text)
            neg_count = sum(1 for term in self.negative_terms if term in text)
            risk_count = sum(1 for term in self.risk_terms if term in text)
            
            total = pos_count + neg_count
            sentiment = (pos_count - neg_count) / total if total > 0 else 0
            
            sentiments.append({
                'positive_count': pos_count,
                'negative_count': neg_count,
                'risk_count': risk_count,
                'sentiment_score': sentiment,
                'risk_level': min(risk_count / 2, 1.0) if risk_count > 0 else 0
            })
        
        sentiment_df = pd.DataFrame(sentiments)
        result_df = pd.concat([news_df, sentiment_df], axis=1)
        
        return result_df
    
    def ensure_datetime_columns(self, df):
        """Ensure the dataframe has properly formatted datetime columns"""
        if df.empty:
            return df
            
        # Parse dates if needed
        if 'parsed_date' not in df.columns:
            # Try different date columns in order of preference
            if 'pub_date_iso' in df.columns:
                df['parsed_date'] = pd.to_datetime(df['pub_date_iso'], errors='coerce')
            elif 'pub_date' in df.columns:
                df['parsed_date'] = pd.to_datetime(df['pub_date'], errors='coerce')
            else:
                df['parsed_date'] = pd.to_datetime(df['fetch_time'], errors='coerce')
        elif not pd.api.types.is_datetime64_any_dtype(df['parsed_date']):
            # If it exists but isn't datetime type
            df['parsed_date'] = pd.to_datetime(df['parsed_date'], errors='coerce')
        
        # Remove timezone info for consistent comparison (if it's a datetime column)
        if pd.api.types.is_datetime64_any_dtype(df['parsed_date']):
            # Check if we have timezone info before trying to remove it
            if df['parsed_date'].dt.tz is not None:
                df['parsed_date'] = df['parsed_date'].dt.tz_localize(None)
                
        # Drop rows with invalid dates
        df = df.dropna(subset=['parsed_date'])
        
        return df
    

    def fix_datetime_comparison(self, news_df):
        """
        Helper function to ensure all datetime comparisons are compatible
        by making all datetimes naive (removing timezone info)
        """
        # Create a copy to avoid SettingWithCopyWarning
        news_df = news_df.copy()

        # First, ensure parsed_date is a datetime column
        if 'parsed_date' not in news_df.columns:
            if 'pub_date_iso' in news_df.columns:
                news_df['parsed_date'] = pd.to_datetime(news_df['pub_date_iso'], errors='coerce')
            elif 'pub_date' in news_df.columns:
                news_df['parsed_date'] = pd.to_datetime(news_df['pub_date'], errors='coerce')
            else:
                news_df['parsed_date'] = pd.to_datetime(news_df['fetch_time'], errors='coerce')

        # Make sure parsed_date is a datetime column
        if not pd.api.types.is_datetime64_any_dtype(news_df['parsed_date']):
            news_df['parsed_date'] = pd.to_datetime(news_df['parsed_date'], errors='coerce')
            news_df = news_df.dropna(subset=['parsed_date'])

        # Convert timezone-aware datetimes to timezone-naive using string conversion
        # This is the most reliable method to strip timezone information
        news_df['parsed_date'] = pd.to_datetime(
            news_df['parsed_date'].dt.strftime('%Y-%m-%d %H:%M:%S'),
            errors='coerce'
        )

        # Drop rows with invalid dates
        news_df = news_df.dropna(subset=['parsed_date'])

        return news_df



    
    def classify_risk_type(self, text):
        """
        Classify text into risk categories using the risk term mappings
        """
        text_lower = text.lower()
        
        # Initialize risk categories with all defined categories
        risk_categories = {category: 0 for category in set(self.risk_category_map.values())}
        
        # Count mentions of risk terms by category
        for term, category in self.risk_category_map.items():
            if term in text_lower:
                risk_categories[category] += 1
        
        total_mentions = sum(risk_categories.values())
        
        if total_mentions > 0:
            # Normalize scores
            for category in risk_categories:
                risk_categories[category] = risk_categories[category] / total_mentions
            
            primary_category = max(risk_categories, key=risk_categories.get)
            risk_categories["primary_category"] = primary_category
        else:
            risk_categories["primary_category"] = "None"
        
        return risk_categories
    


    def detect_anomalous_clusters(self, news_df, timeframe_hours=48, min_cluster_size=3):
        if news_df.empty or not ML_AVAILABLE:
            return []

        # Create a copy and fix datetime issues 
        news_df = self.fix_datetime_comparison(news_df)

        if news_df.empty:
            logger.warning("No valid dates found in news data")
            return []

        # Create a timezone-naive threshold
        recent_threshold = datetime.now().replace(tzinfo=None) - timedelta(hours=timeframe_hours)

        # Now do the comparison with both sides being timezone-naive
        try:
            recent_news = news_df[news_df['parsed_date'] > recent_threshold].copy()
        except Exception as e:
            logger.error(f"Error in date comparison: {e}")
            return []

        if len(recent_news) < min_cluster_size * 2:
            logger.info(f"Not enough recent news articles for clustering (found {len(recent_news)})")
            return []

        # Rest of function remains the same
        recent_news['full_text'] = recent_news['title'].fillna('') + ' ' + recent_news['description'].fillna('')

        if self.nlp_ready:
            recent_news['processed_text'] = recent_news['full_text'].apply(self.preprocess_text)
        else:
            recent_news['processed_text'] = recent_news['full_text'].apply(lambda x: x.lower())

        try:
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(recent_news['processed_text'])

            dbscan = DBSCAN(eps=0.7, min_samples=min_cluster_size, metric='cosine')
            recent_news['cluster'] = dbscan.fit_predict(tfidf_matrix)

            clustered_news = recent_news[recent_news['cluster'] >= 0]

            cluster_counts = clustered_news['cluster'].value_counts().to_dict()

            significant_clusters = {k: v for k, v in cluster_counts.items() if v >= min_cluster_size}

            anomalous_clusters = []

            for cluster_id, count in significant_clusters.items():
                cluster_articles = clustered_news[clustered_news['cluster'] == cluster_id]

                avg_sentiment = cluster_articles['sentiment_score'].mean() if 'sentiment_score' in cluster_articles.columns else 0
                avg_risk = cluster_articles['risk_level'].mean() if 'risk_level' in cluster_articles.columns else 0

                cluster_text = ' '.join(cluster_articles['processed_text'])

                risk_terms_present = [term for term in self.risk_terms if term in cluster_text.lower()]

                novelty_score = (len(risk_terms_present) / max(1, len(cluster_articles))) * (count / len(recent_news))

                risk_classification = self.classify_risk_type(cluster_text)

                if len(risk_terms_present) > 0 or avg_sentiment < -0.1 or avg_risk > 0.2:
                    headline = cluster_articles.iloc[cluster_articles['parsed_date'].argmax()]['title']

                    anomalous_clusters.append({
                        'cluster_id': int(cluster_id),
                        'size': int(count),
                        'headline': headline,
                        'avg_sentiment': float(avg_sentiment),
                        'avg_risk': float(avg_risk),
                        'risk_terms': risk_terms_present[:10],
                        'novelty_score': float(novelty_score),
                        'primary_risk_category': risk_classification['primary_category'],
                        'risk_categories': {k: float(v) for k, v in risk_classification.items() if k != 'primary_category'},
                        'articles': cluster_articles[['title', 'link', 'source_name', 'parsed_date']].head(5).to_dict('records')
                    })

            anomalous_clusters.sort(key=lambda x: x['novelty_score'], reverse=True)

            return anomalous_clusters

        except Exception as e:
            logger.error(f"Error in anomalous cluster detection: {e}")
            return []






    def detect_rapid_sentiment_shifts(self, news_df, window_hours=12, threshold=0.15):
        if news_df.empty:
            return []
        
        if 'sentiment_score' not in news_df.columns:
            news_df = self.calculate_sentiment(news_df)
        
        # Handle date columns safely
        news_df = self.ensure_datetime_columns(news_df)
        
        if news_df.empty:
            logger.warning("No valid dates found in news data")
            return []
            
        news_df = news_df.sort_values('parsed_date')
        
        if len(news_df) < 10:
            return []
        
        try:
            shifts = []
            
            latest_time = news_df['parsed_date'].max()
            window_end = latest_time
            
            while len(news_df[news_df['parsed_date'] <= window_end]) > 0:
                window_start = window_end - timedelta(hours=window_hours)
                
                window_news = news_df[(news_df['parsed_date'] > window_start) & 
                                    (news_df['parsed_date'] <= window_end)]
                
                prev_window_start = window_start - timedelta(hours=window_hours)
                prev_window_news = news_df[(news_df['parsed_date'] > prev_window_start) & 
                                        (news_df['parsed_date'] <= window_start)]
                
                if len(window_news) >= 5 and len(prev_window_news) >= 5:
                    window_sentiment = window_news['sentiment_score'].mean()
                    prev_window_sentiment = prev_window_news['sentiment_score'].mean()
                    
                    sentiment_shift = window_sentiment - prev_window_sentiment
                    
                    window_risk = window_news['risk_level'].mean()
                    prev_window_risk = prev_window_news['risk_level'].mean()
                    
                    risk_shift = window_risk - prev_window_risk
                    
                    if abs(sentiment_shift) > threshold or risk_shift > threshold:
                        window_text = ' '.join(window_news['full_text'])
                        
                        risk_classification = self.classify_risk_type(window_text)
                        
                        headline = window_news.iloc[window_news['parsed_date'].argmax()]['title']
                        
                        shifts.append({
                            'window_start': window_start.isoformat(),
                            'window_end': window_end.isoformat(),
                            'sentiment_shift': float(sentiment_shift),
                            'risk_shift': float(risk_shift),
                            'window_sentiment': float(window_sentiment),
                            'window_risk': float(window_risk),
                            'headline': headline,
                            'article_count': len(window_news),
                            'primary_risk_category': risk_classification['primary_category'],
                            'articles': window_news[['title', 'link', 'source_name', 'parsed_date']].head(5).to_dict('records')
                        })
                
                window_end = window_start
                
                if window_end < news_df['parsed_date'].min():
                    break
            
            shifts.sort(key=lambda x: abs(x['sentiment_shift']), reverse=True)
            
            return shifts
            
        except Exception as e:
            logger.error(f"Error in sentiment shift detection: {e}")
            return []
    
    def generate_weekend_risk_report(self):
        report = {
            'generated_at': datetime.now().isoformat(),
            'weekend_alert': False,
            'alert_reason': "",
            'risk_level': 0.0,
            'anomalous_clusters': [],
            'sentiment_shifts': [],
            'sector_risks': {},
            'key_tickers_at_risk': [],
            'market_mood': "Neutral"
        }
        
        news_df = self.load_news_data(days_back=3)
        
        if news_df.empty:
            report['error'] = "No news data available"
            return report
        
        news_df = self.calculate_sentiment(news_df)
        
        # Handle date columns safely
        news_df = self.ensure_datetime_columns(news_df)
        
        anomalous_clusters = self.detect_anomalous_clusters(news_df, timeframe_hours=72)
        report['anomalous_clusters'] = anomalous_clusters
        
        sentiment_shifts = self.detect_rapid_sentiment_shifts(news_df, window_hours=24)
        report['sentiment_shifts'] = sentiment_shifts
        
        cluster_risk = max([c['novelty_score'] for c in anomalous_clusters]) if anomalous_clusters else 0
        shift_risk = max([abs(s['sentiment_shift']) for s in sentiment_shifts]) if sentiment_shifts else 0
        
        if 'parsed_date' in news_df.columns:
            recent_threshold = datetime.now() - timedelta(hours=24)
            recent_news = news_df[news_df['parsed_date'] > recent_threshold]
            if not recent_news.empty:
                recent_sentiment = recent_news['sentiment_score'].mean()
                recent_risk = recent_news['risk_level'].mean()
            else:
                recent_sentiment = 0
                recent_risk = 0
        else:
            recent_sentiment = news_df['sentiment_score'].mean()
            recent_risk = news_df['risk_level'].mean()
        
        report['risk_level'] = (0.4 * cluster_risk + 0.3 * shift_risk + 0.3 * recent_risk)
        
        if recent_sentiment < -0.2:
            report['market_mood'] = "Bearish"
        elif recent_sentiment < -0.05:
            report['market_mood'] = "Slightly Bearish"
        elif recent_sentiment > 0.2:
            report['market_mood'] = "Bullish"
        elif recent_sentiment > 0.05:
            report['market_mood'] = "Slightly Bullish"
        else:
            report['market_mood'] = "Neutral"
        
        if report['risk_level'] > 0.6:
            report['weekend_alert'] = True
            report['alert_reason'] = "Critical risk level detected. "
        elif report['risk_level'] > 0.4:
            report['weekend_alert'] = True
            report['alert_reason'] = "Elevated risk level detected. "
        
        high_novelty_clusters = [c for c in anomalous_clusters if c['novelty_score'] > 0.5]
        if high_novelty_clusters:
            report['weekend_alert'] = True
            report['alert_reason'] += f"Unusual news clusters detected: {len(high_novelty_clusters)}. "
        
        major_shifts = [s for s in sentiment_shifts if abs(s['sentiment_shift']) > 0.3]
        if major_shifts:
            report['weekend_alert'] = True
            report['alert_reason'] += f"Major sentiment shifts detected: {len(major_shifts)}. "
        
        sector_risks = {}
        
        for cluster in anomalous_clusters:
            category = cluster['primary_risk_category']
            if category != "None":
                sector_risks[category] = sector_risks.get(category, 0) + cluster['novelty_score']
        
        if sector_risks:
            max_risk = max(sector_risks.values())
            for category in sector_risks:
                sector_risks[category] = sector_risks[category] / max_risk
        
        report['sector_risks'] = sector_risks
        
        sector_ticker_map = {
            "Financial Crisis": ["XLF", "KBE", "KRE"],
            "Geopolitical Risk": ["XLE", "GLD", "USO"],
            "Natural Disaster": ["XLU", "MRO", "XOM"],
            "Health Crisis": ["XLV", "JNJ", "PFE"],
            "Supply Chain Disruption": ["XLI", "FDX", "UPS"],
            "Energy Crisis": ["XLE", "USO", "UNG"],
            "Regulatory/Legal Risk": ["XLK", "META", "GOOGL"]
        }
        
        tickers_at_risk = []
        for category, risk_level in sector_risks.items():
            if risk_level > 0.5 and category in sector_ticker_map:
                for ticker in sector_ticker_map[category]:
                    tickers_at_risk.append({
                        "ticker": ticker,
                        "risk_category": category,
                        "risk_level": risk_level
                    })
        
        unique_tickers = {}
        for item in tickers_at_risk:
            ticker = item['ticker']
            if ticker not in unique_tickers or item['risk_level'] > unique_tickers[ticker]['risk_level']:
                unique_tickers[ticker] = item
        
        report['key_tickers_at_risk'] = list(unique_tickers.values())
        
        report_path = OUTPUT_DIR / f"weekend_risk_report_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=json_serializable)
        
        return report


def json_serializable(obj):
    """
    Convert objects to JSON serializable format
    """
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")




def run_nlp_analysis(days_back=3, ticker_file=None, portfolio_file=None):
    analyzer = NewsAnalyzer(ticker_file=ticker_file, portfolio_file=portfolio_file)
    
    logger.info("Loading news data...")
    news_df = analyzer.load_news_data(days_back=days_back)
    
    if news_df.empty:
        logger.error("No news data available for analysis")
        return None
    
    logger.info(f"Analyzing {len(news_df)} news articles...")
    
    news_df = analyzer.calculate_sentiment(news_df)
    
    logger.info("Looking for anomalous news clusters...")
    anomalous_clusters = analyzer.detect_anomalous_clusters(news_df)
    
    logger.info("Detecting sentiment shifts...")
    sentiment_shifts = analyzer.detect_rapid_sentiment_shifts(news_df)
    
    logger.info("Generating weekend risk report...")
    risk_report = analyzer.generate_weekend_risk_report()
    
    report_path = OUTPUT_DIR / f"nlp_analysis_{datetime.now().strftime('%Y%m%d')}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            'risk_report': risk_report,
            'anomalous_clusters_count': len(anomalous_clusters),
            'sentiment_shifts_count': len(sentiment_shifts),
            'analysis_timestamp': datetime.now().isoformat()
        }, f, indent=2, default=json_serializable)
    
    logger.info(f"Analysis complete. Results saved to {report_path}")
    return risk_report

if __name__ == "__main__":
    risk_report = run_nlp_analysis()
    
    if risk_report:
        print(f"Risk Level: {risk_report['risk_level']:.2f}")
        print(f"Market Mood: {risk_report['market_mood']}")
        print(f"Alert Status: {'🚨 ALERT!' if risk_report['weekend_alert'] else 'Normal'}")
        
        if risk_report['weekend_alert']:
            print(f"Alert Reason: {risk_report['alert_reason']}")
        
        if risk_report['key_tickers_at_risk']:
            print("Key Tickers at Risk:", ", ".join([t['ticker'] for t in risk_report['key_tickers_at_risk']]))