import os
import random
import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from joblib import dump, load
import argparse
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm
from joblib import parallel_backend
from contextlib import redirect_stdout, redirect_stderr
import io
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def setup_logging(config):
    """Set up logging configuration."""
    log_dir = os.path.dirname(config['log_file'])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        filename=config['log_file'],
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Test logging to ensure it's working
    logging.info("Logging system initialized.")
    print(f"Logging configured. Log file: {config['log_file']}")


argparser = argparse.ArgumentParser()
argparser.add_argument("--runpercent", type=int, default=50, help="Percentage of files to process.")
argparser.add_argument("--clear", action='store_true', help="Flag to clear the model and data directories.")
argparser.add_argument("--predict", action='store_true', help="Flag to predict new data.")
argparser.add_argument("--reuse", action='store_true', help="Flag to reuse existing training data if available.")
argparser.add_argument("--threshold_pos", type=float, default=0.70, help="Confidence threshold for positive predictions.")
argparser.add_argument("--threshold_neg", type=float, default=0.70, help="Confidence threshold for negative predictions.")
argparser.add_argument("--eda", action='store_true', help="Flag to perform exploratory data analysis on predictions.")
args = argparser.parse_args()


config = {
    "input_directory": "Data/IndicatorData",
    "model_output_directory": "Data/ModelData",
    "data_output_directory": "Data/ModelData/TrainingData",
    "prediction_output_directory": "Data/DualRFpredictions",
    "feature_importance_output": "Data/ModelData/FeatureImportances/feature_importance.parquet",
    "log_file": "data/logging/DualPredictor.log",
    "file_selection_percentage": args.runpercent,
    "target_column": "percent_change_Close",
    "date_column": "Date",

    # Model parameters for positive classifier
    "pos_model": {
        "n_estimators": 64,
        "criterion": "entropy",
        "max_depth": 8,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "min_weight_fraction_leaf": 0,
        "max_features": 0.10,
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0,
        "bootstrap": True,
        "oob_score": True,
        "random_state": 3301,
        "verbose": 2,
        "warm_start": False,
        "class_weight": {0: 0.9, 1: 2.0},
        "ccp_alpha": 0,
        "max_samples": None
    },
    
    # Model parameters for negative classifier
    "neg_model": {
        "n_estimators": 64,
        "criterion": "entropy",
        "max_depth": 8,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "min_weight_fraction_leaf": 0,
        "max_features": 0.10,
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0,
        "bootstrap": True,
        "oob_score": True,
        "random_state": 4402,  # Different seed for diversity
        "verbose": 2,
        "warm_start": False,
        "class_weight": {0: 0.9, 1: 2.0},
        "ccp_alpha": 0,
        "max_samples": None
    }
}


def drop_string_columns(df, date_column, target_column):
    """
    Drops any column (except date_column and target_column) that contains string data.
    """
    # Identify columns (excluding date and target) where the dtype is object
    columns_to_drop = [col for col in df.columns 
                       if col not in [date_column, target_column] and df[col].dtype == 'object']
    if columns_to_drop:
        logging.info(f"Dropping columns due to string data: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)
    return df


def prepare_training_data(input_directory, output_directory, file_selection_percentage, target_column, reuse, date_column):
    output_file = os.path.join(output_directory, 'training_data.parquet')
    if reuse and os.path.exists(output_file):
        logging.info("Reusing existing training data.")
        print("Reusing existing training data.")
        return pd.read_parquet(output_file)
    
    logging.info("Preparing new training data.")
    all_files = [f for f in os.listdir(input_directory) if f.endswith('.parquet')]
    selected_files = random.sample(all_files, int(len(all_files) * file_selection_percentage / 100))
    
    if os.path.exists(output_file):
        os.remove(output_file)
    pbar = tqdm(total=len(selected_files), desc="Processing files")
    
    all_data = []
    
    for file in selected_files:
        df = pd.read_parquet(os.path.join(input_directory, file))
        # Ensure the df is not empty and has at least the target and 50 rows 
        if df.shape[0] > 50 and target_column in df.columns and date_column in df.columns:
            # Ensure the date column is in datetime format
            df[date_column] = pd.to_datetime(df[date_column])
            
            # Drop columns that contain string data (except date and target)
            df = drop_string_columns(df, date_column, target_column)
            
            # Shift the target column by 1 (for future prediction)
            df[target_column] = df[target_column].shift(-1)
            # Remove the first 2 rows and the last 2 rows to avoid boundary issues after shifting
            df = df.iloc[2:-2]
            # Drop rows with NaN in the target column
            df = df.dropna(subset=[target_column])
            # Filter out target values outside a reasonable range
            df = df[(df[target_column] <= 10000) & (df[target_column] >= -10000)]
            
            # Only add if there is any remaining data
            if not df.empty:
                all_data.append(df)
        pbar.update(1)
    pbar.close()
    
    if len(all_data) == 0:
        logging.error("No valid training data found after processing files. Check your data provider and cleaning logic.")
        raise ValueError("No valid training data found after processing files.")
    
    # Concatenate the dataframes together
    combined_df = pd.concat(all_data)
    # Group by date, shuffle within each group, and then concatenate
    grouped = combined_df.groupby(date_column)
    shuffled_groups = [group.sample(frac=1).reset_index(drop=True) for _, group in grouped]
    
    if len(shuffled_groups) == 0:
        logging.error("No groups available after grouping by date. Check your data's date values.")
        raise ValueError("No groups available after grouping by date.")
    
    final_df = pd.concat(shuffled_groups).reset_index(drop=True)
    # Save the dataframe to a parquet file
    final_df.to_parquet(output_file, index=False)
    return final_df


def train_dual_random_forest(training_data, config, confidence_threshold_pos=0.70, confidence_threshold_neg=0.70):
    logging.info("Training Dual Random Forest models (positive and negative extremes).")
    
    # Remove the old model files
    pos_model_path = os.path.join(config['model_output_directory'], 'rf_model_positive.joblib')
    neg_model_path = os.path.join(config['model_output_directory'], 'rf_model_negative.joblib')
    
    for path in [pos_model_path, neg_model_path]:
        if os.path.exists(path):
            os.remove(path)

    # Sort data by date to ensure temporal order
    training_data = training_data.sort_values(config['date_column'])
    
    # Create two copies of the data with different target definitions
    # For positive extremes model: y = 1 if target > 0 else 0
    # For negative extremes model: y = 1 if target < 0 else 0
    
    # Separate features from target for positive model
    X = training_data.drop(columns=[config['target_column']])
    y_pos = training_data[config['target_column']].apply(lambda x: 1 if x > 0 else 0)
    y_neg = training_data[config['target_column']].apply(lambda x: 1 if x < 0 else 0)
    
    # Get datetime columns before removing them from X
    datetime_columns = X.select_dtypes(include=['datetime64']).columns
    
    # Calculate the split point based on dates
    split_date = X[config['date_column']].quantile(0.8)  # Using last 20% of dates for testing
    logging.info(f"Split date: {split_date}")
    
    # Split based on date
    train_mask = X[config['date_column']] < split_date
    test_mask = X[config['date_column']] >= split_date
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    
    y_pos_train = y_pos[train_mask]
    y_pos_test = y_pos[test_mask]
    
    y_neg_train = y_neg[train_mask]
    y_neg_test = y_neg[test_mask]
    
    # Store the test indices and dates for later use in evaluation
    test_dates = X_test[config['date_column']].copy()
    test_indices = X_test.index.copy()
    
    # Now remove datetime columns from both train and test
    X_train = X_train.drop(columns=datetime_columns)
    X_test = X_test.drop(columns=datetime_columns)

    # Train the Positive Extreme model
    logging.info("Training POSITIVE model...")
    pos_clf = RandomForestClassifier(
        n_estimators=config['pos_model']['n_estimators'],
        criterion=config['pos_model']['criterion'],
        max_depth=config['pos_model']['max_depth'],
        min_samples_split=config['pos_model']['min_samples_split'],
        min_samples_leaf=config['pos_model']['min_samples_leaf'],
        min_weight_fraction_leaf=config['pos_model']['min_weight_fraction_leaf'],
        max_features=config['pos_model']['max_features'],
        max_leaf_nodes=config['pos_model']['max_leaf_nodes'],
        min_impurity_decrease=config['pos_model']['min_impurity_decrease'],
        bootstrap=config['pos_model']['bootstrap'],
        oob_score=config['pos_model']['oob_score'],
        random_state=config['pos_model']['random_state'],
        verbose=config['pos_model']['verbose'],
        warm_start=config['pos_model']['warm_start'],
        class_weight=config['pos_model']['class_weight'],
        ccp_alpha=config['pos_model']['ccp_alpha'],
        max_samples=config['pos_model']['max_samples'],
        n_jobs=-1  # Use all available processors
    )
    
    pos_clf.fit(X_train, y_pos_train)
    
    # Train the Negative Extreme model
    logging.info("Training NEGATIVE model...")
    neg_clf = RandomForestClassifier(
        n_estimators=config['neg_model']['n_estimators'],
        criterion=config['neg_model']['criterion'],
        max_depth=config['neg_model']['max_depth'],
        min_samples_split=config['neg_model']['min_samples_split'],
        min_samples_leaf=config['neg_model']['min_samples_leaf'],
        min_weight_fraction_leaf=config['neg_model']['min_weight_fraction_leaf'],
        max_features=config['neg_model']['max_features'],
        max_leaf_nodes=config['neg_model']['max_leaf_nodes'],
        min_impurity_decrease=config['neg_model']['min_impurity_decrease'],
        bootstrap=config['neg_model']['bootstrap'],
        oob_score=config['neg_model']['oob_score'],
        random_state=config['neg_model']['random_state'],
        verbose=config['neg_model']['verbose'],
        warm_start=config['neg_model']['warm_start'],
        class_weight=config['neg_model']['class_weight'],
        ccp_alpha=config['neg_model']['ccp_alpha'],
        max_samples=config['neg_model']['max_samples'],
        n_jobs=-1  # Use all available processors
    )
    
    neg_clf.fit(X_train, y_neg_train)
    
    # Evaluate both models
    # Positive model evaluation
    y_pos_pred_proba = pos_clf.predict_proba(X_test)
    y_pos_pred = np.where(
        y_pos_pred_proba[:, 1] >= confidence_threshold_pos, 1,
        np.where(y_pos_pred_proba[:, 0] >= confidence_threshold_neg, 0, -1)
    )
    
    # Negative model evaluation
    y_neg_pred_proba = neg_clf.predict_proba(X_test)
    y_neg_pred = np.where(
        y_neg_pred_proba[:, 1] >= confidence_threshold_pos, 1,
        np.where(y_neg_pred_proba[:, 0] >= confidence_threshold_neg, 0, -1)
    )
    
    # Filter out undecided predictions for each model
    pos_mask = y_pos_pred != -1
    pos_count = np.sum(pos_mask)
    logging.info(f"Positive model makes {pos_count} confident predictions out of {len(y_pos_test)} ({pos_count/len(y_pos_test)*100:.2f}%)")
    
    neg_mask = y_neg_pred != -1
    neg_count = np.sum(neg_mask)
    logging.info(f"Negative model makes {neg_count} confident predictions out of {len(y_neg_test)} ({neg_count/len(y_neg_test)*100:.2f}%)")
    
    # Create evaluation metrics for both models
    for model_name, y_pred, y_test, mask in [
        ("Positive", y_pos_pred, y_pos_test, pos_mask),
        ("Negative", y_neg_pred, y_neg_test, neg_mask)
    ]:
        if np.sum(mask) > 0:
            y_test_filtered = y_test[mask]
            y_pred_filtered = y_pred[mask]
            
            accuracy = accuracy_score(y_test_filtered, y_pred_filtered)
            f1 = f1_score(y_test_filtered, y_pred_filtered, average='weighted')
            precision = precision_score(y_test_filtered, y_pred_filtered, average='weighted')
            recall = recall_score(y_test_filtered, y_pred_filtered, average='weighted')
            
            logging.info(f"{model_name} Model Metrics:")
            logging.info(f"  Accuracy: {accuracy:.4f}")
            logging.info(f"  F1 Score: {f1:.4f}")
            logging.info(f"  Precision: {precision:.4f}")
            logging.info(f"  Recall: {recall:.4f}")
            
            print(f"\n{model_name} Model Metrics:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            
            if len(y_test_filtered) > 0 and len(np.unique(y_test_filtered)) > 1:
                print(f"\n{model_name} Classification Report:")
                print(classification_report(y_test_filtered, y_pred_filtered, zero_division=0))
        else:
            logging.warning(f"{model_name} model has no confident predictions after filtering.")
            print(f"\nWarning: {model_name} model has no confident predictions after filtering.")
    
    # Create a combined prediction for analysis purposes
    # Combine predictions and store them with test dates
    results_df = pd.DataFrame({
        'date': test_dates,
        'true_target': training_data.loc[test_indices, config['target_column']],
        'pos_model_prob': y_pos_pred_proba[:, 1],
        'neg_model_prob': y_neg_pred_proba[:, 1],
        'pos_model_pred': y_pos_pred,
        'neg_model_pred': y_neg_pred
    })
    
    # Calculate the combined signal
    # When pos_model predicts 1 (up) with high confidence: +1
    # When neg_model predicts 1 (down) with high confidence: -1
    # When both models predict their respective 1s or both predict 0s or disagree: 0
    
    results_df['combined_signal'] = 0
    # Positive signal when pos_model predicts 1 (going up) with high confidence
    results_df.loc[(results_df['pos_model_pred'] == 1), 'combined_signal'] = 1
    # Negative signal when neg_model predicts 1 (going down) with high confidence
    results_df.loc[(results_df['neg_model_pred'] == 1), 'combined_signal'] = -1
    
    # Calculate the difference between positive and negative probabilities
    results_df['prob_difference'] = results_df['pos_model_prob'] - results_df['neg_model_prob']
    
    # Save the results
    output_file = os.path.join(config['model_output_directory'], 'test_predictions.parquet')
    results_df.to_parquet(output_file, index=False)
    logging.info(f"Test predictions saved to {output_file}")
    
    # Save both models
    dump(pos_clf, pos_model_path)
    dump(neg_clf, neg_model_path)
    logging.info(f"Models saved to {pos_model_path} and {neg_model_path}")
    
    # Save feature importances for both models
    feature_importance_dir = os.path.dirname(config['feature_importance_output'])
    os.makedirs(feature_importance_dir, exist_ok=True)
    
    # Save feature importances for positive model
    pos_importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': pos_clf.feature_importances_
    }).sort_values(by='importance', ascending=False)
    pos_importances['importance'] = pos_importances['importance'].round(5)
    pos_importances.to_parquet(os.path.join(feature_importance_dir, 'feature_importance_positive.parquet'), index=False)
    
    # Save feature importances for negative model
    neg_importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': neg_clf.feature_importances_
    }).sort_values(by='importance', ascending=False)
    neg_importances['importance'] = neg_importances['importance'].round(5)
    neg_importances.to_parquet(os.path.join(feature_importance_dir, 'feature_importance_negative.parquet'), index=False)
    
    logging.info("Feature importances saved for both models")
    
    # Return the model objects and evaluation data
    return {
        'pos_model': pos_clf,
        'neg_model': neg_clf,
        'results_df': results_df,
    }


def predict_dual_models(input_directory, pos_model_path, neg_model_path, output_directory, 
                        target_column, date_column, confidence_threshold_pos=0.7, confidence_threshold_neg=0.7):
    logging.info("Loading trained models for prediction.")
    
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    # Remove all the parquet files in the output directory
    for file in os.listdir(output_directory):
        if file.endswith('.parquet'):
            os.remove(os.path.join(output_directory, file))
    
    # Load the trained models
    pos_clf = load(pos_model_path)
    neg_clf = load(neg_model_path)
    
    # Get the feature names the models were trained on
    model_features = pos_clf.feature_names_in_
    
    # Get the list of files to process
    all_files = [f for f in os.listdir(input_directory) if f.endswith('.parquet')]
    
    # Initialize progress bar
    pbar = tqdm(total=len(all_files), desc="Processing files", ncols=100)
    
    # Set up a custom logger to capture joblib output
    joblib_logger = logging.getLogger('joblib')
    joblib_logger.setLevel(logging.ERROR)  # Only show errors
    
    # Redirect stdout and stderr
    null_io = io.StringIO()
    
    all_predictions = []
    
    for file in all_files:
        df = pd.read_parquet(os.path.join(input_directory, file))
        # Ensure the date column is in datetime format
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Check to see if file has more than 252 rows
        if df.shape[0] < 252:
            pbar.update(1)
            continue

        # Store original data columns we want to keep
        original_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        indicator_columns = ['Distance to Resistance (%)', 'Distance to Support (%)']
        
        # Check if these columns exist and add them only if they do
        available_columns = [col for col in original_columns + indicator_columns if col in df.columns]
        
        # Remove datetime and target columns from X
        datetime_columns = df.select_dtypes(include=['datetime64']).columns
        
        # Copy the DataFrame to prevent modification warnings
        X = df.copy()
        
        # Drop columns we don't need for prediction
        columns_to_drop = list(datetime_columns)
        if target_column in X.columns:
            columns_to_drop.append(target_column)
        X = X.drop(columns=columns_to_drop)
        
        # Align the features with the model's expected features
        X = X.reindex(columns=model_features, fill_value=0)
        
        # Make predictions with probabilities
        with parallel_backend('threading', n_jobs=-1):
            with redirect_stdout(null_io), redirect_stderr(null_io):
                pos_pred_proba = pos_clf.predict_proba(X)
                neg_pred_proba = neg_clf.predict_proba(X)
        
        # Create a results dataframe
        result_df = pd.DataFrame()
        
        # Add the original columns we wanted to keep
        for col in available_columns:
            if col in df.columns:
                result_df[col] = df[col]
        
        # Add positive model predictions
        result_df['UpProbability'] = pos_pred_proba[:, 1]
        result_df['UpPrediction'] = (pos_pred_proba[:, 1] >= confidence_threshold_pos).astype(int)
        
        # Add negative model predictions
        result_df['DownProbability'] = neg_pred_proba[:, 1]
        result_df['DownPrediction'] = (neg_pred_proba[:, 1] >= confidence_threshold_neg).astype(int)
        
        # Calculate combined signal:
        # +1 when model says high confidence of going up
        # -1 when model says high confidence of going down
        # 0 otherwise
        result_df['CombinedSignal'] = 0
        result_df.loc[result_df['UpPrediction'] == 1, 'CombinedSignal'] = 1
        result_df.loc[result_df['DownPrediction'] == 1, 'CombinedSignal'] = -1
        
        # Calculate probability difference (upward - downward)
        result_df['ProbabilityDifference'] = result_df['UpProbability'] - result_df['DownProbability']
        
        # Add the symbol name from the file
        symbol = os.path.splitext(file)[0]
        result_df['Symbol'] = symbol
        
        # Save the prediction file
        output_file_path = os.path.join(output_directory, file)
        result_df.to_parquet(output_file_path, index=False)
        
        # Collect data for overall analysis
        all_predictions.append(result_df)
        
        pbar.update(1)
    
    pbar.close()
    logging.info(f"Predictions saved to {output_directory}")
    
    # Combine all predictions for analysis if there are any
    if all_predictions:
        combined_df = pd.concat(all_predictions)
        combined_output = os.path.join(output_directory, "all_predictions.parquet")
        combined_df.to_parquet(combined_output, index=False)
        logging.info(f"Combined predictions saved to {combined_output}")
        return combined_df
    else:
        logging.warning("No predictions were generated.")
        return None


def explore_prediction_data(prediction_df, output_directory):
    """
    Perform exploratory data analysis on the prediction results.
    """
    if prediction_df is None or prediction_df.empty:
        logging.error("No prediction data available for EDA.")
        return
    
    logging.info("Performing exploratory data analysis on predictions...")
    
    # Create output directory for visualizations
    eda_dir = os.path.join(output_directory, "EDA")
    os.makedirs(eda_dir, exist_ok=True)
    
    # 1. Distribution of probability differences
    plt.figure(figsize=(12, 6))
    sns.histplot(prediction_df['ProbabilityDifference'], bins=50, kde=True)
    plt.title('Distribution of Probability Differences (Up - Down)')
    plt.xlabel('Probability Difference')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(eda_dir, 'probability_difference_dist.png'))
    plt.close()
    
    # 2. Distribution of combined signals
    plt.figure(figsize=(10, 6))
    counts = prediction_df['CombinedSignal'].value_counts().sort_index()
    sns.barplot(x=counts.index, y=counts.values)
    plt.title('Distribution of Combined Signals')
    plt.xlabel('Signal (-1: Down, 0: Neutral, 1: Up)')
    plt.ylabel('Count')
    plt.xticks([-1, 0, 1], ['Down', 'Neutral', 'Up'])
    plt.savefig(os.path.join(eda_dir, 'combined_signal_dist.png'))
    plt.close()
    
    # 3. Relationship between up and down probabilities
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='UpProbability', y='DownProbability', 
                    hue='CombinedSignal', 
                    palette={-1: 'red', 0: 'gray', 1: 'green'},
                    data=prediction_df)
    plt.title('Relationship Between Up and Down Probabilities')
    plt.xlabel('Up Probability')
    plt.ylabel('Down Probability')
    plt.savefig(os.path.join(eda_dir, 'up_down_scatter.png'))
    plt.close()
    
    # 4. Time series analysis if dates are available
    if 'Date' in prediction_df.columns:
        # Filter to most recent 90 days of data for clearer visualization
        recent_df = prediction_df.sort_values('Date').groupby('Symbol').tail(90)
        
        # Sample up to 10 symbols for time series visualization
        symbols = recent_df['Symbol'].unique()
        selected_symbols = symbols[:min(10, len(symbols))]
        
        for symbol in selected_symbols:
            symbol_df = recent_df[recent_df['Symbol'] == symbol].sort_values('Date')
            
            plt.figure(figsize=(14, 8))
            
            # Plot Close price
            ax1 = plt.subplot(211)
            ax1.plot(symbol_df['Date'], symbol_df['Close'], 'b-')
            ax1.set_title(f'{symbol} Close Price and Probability Difference')
            ax1.set_ylabel('Close Price')
            ax1.grid(True)
            
            # Plot probability difference
            ax2 = plt.subplot(212, sharex=ax1)
            ax2.plot(symbol_df['Date'], symbol_df['ProbabilityDifference'], 'g-')
            ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            ax2.set_ylabel('Prob Difference (Up - Down)')
            ax2.set_xlabel('Date')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(eda_dir, f'{symbol}_time_series.png'))
            plt.close()
    
    # 5. Summary statistics
    summary_stats = prediction_df[['UpProbability', 'DownProbability', 
                                  'ProbabilityDifference']].describe()
    summary_stats.to_csv(os.path.join(eda_dir, 'summary_statistics.csv'))
    
    # 6. Correlation matrix
    if 'Close' in prediction_df.columns:
        numeric_cols = ['Close', 'UpProbability', 'DownProbability', 'ProbabilityDifference']
        corr_matrix = prediction_df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(eda_dir, 'correlation_matrix.png'))
        plt.close()
    
    # 7. Generate a summary report
    with open(os.path.join(eda_dir, 'eda_summary.txt'), 'w') as f:
        f.write("DUAL EXTREME RANDOM FOREST MODEL - EDA SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis performed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("DATA OVERVIEW:\n")
        f.write(f"Total records: {len(prediction_df)}\n")
        f.write(f"Unique symbols: {prediction_df['Symbol'].nunique()}\n")
        if 'Date' in prediction_df.columns:
            f.write(f"Date range: {prediction_df['Date'].min()} to {prediction_df['Date'].max()}\n\n")
        
        f.write("SIGNAL DISTRIBUTION:\n")
        signal_counts = prediction_df['CombinedSignal'].value_counts().sort_index()
        for signal, count in signal_counts.items():
            signal_name = "Up" if signal == 1 else ("Down" if signal == -1 else "Neutral")
            percentage = (count / len(prediction_df)) * 100
            f.write(f"{signal_name} signals: {count} ({percentage:.2f}%)\n")
        
        f.write("\nPROBABILITY ANALYSIS:\n")
        f.write(f"Average Up Probability: {prediction_df['UpProbability'].mean():.4f}\n")
        f.write(f"Average Down Probability: {prediction_df['DownProbability'].mean():.4f}\n")
        f.write(f"Average Probability Difference: {prediction_df['ProbabilityDifference'].mean():.4f}\n\n")
        
        f.write("VISUALIZATIONS GENERATED:\n")
        f.write("- Probability difference distribution\n")
        f.write("- Combined signal distribution\n")
        f.write("- Up vs Down probability scatter plot\n")
        if 'Date' in prediction_df.columns:
            f.write("- Time series for selected symbols\n")
        f.write("- Correlation matrix\n\n")
        
        f.write("NOTES:\n")
        f.write("- Positive values in 'ProbabilityDifference' suggest upward movement\n")
        f.write("- Negative values in 'ProbabilityDifference' suggest downward movement\n")
        f.write("- Combined signal of 0 indicates uncertainty or conflicting signals\n")
    
    logging.info(f"EDA completed. Results saved to {eda_dir}")
    print(f"EDA completed. Results saved to {eda_dir}")


def main():
    # Create necessary directories
    for directory in [
        config['model_output_directory'], 
        config['data_output_directory'],
        config['prediction_output_directory']
    ]:
        os.makedirs(directory, exist_ok=True)
    
    setup_logging(config)
    
    if args.clear:
        logging.info("Clearing model and data directories.")
        for directory in [config['model_output_directory'], config['data_output_directory'], config['prediction_output_directory']]:
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    logging.error(f"Error deleting {file_path}: {e}")
    
    if not args.predict:
        # Training mode
        training_data = prepare_training_data(
            input_directory=config['input_directory'],
            output_directory=config['data_output_directory'],
            file_selection_percentage=config['file_selection_percentage'],
            target_column=config['target_column'],
            reuse=args.reuse,
            date_column=config['date_column']
        )
        logging.info("Data preparation complete.")
        
        # Train both models
        results = train_dual_random_forest(
            training_data, 
            config, 
            confidence_threshold_pos=args.threshold_pos, 
            confidence_threshold_neg=args.threshold_neg
        )
        
        # Optionally perform EDA on the training results
        if args.eda and 'results_df' in results:
            explore_prediction_data(results['results_df'], config['model_output_directory'])
    else:
        # Prediction mode
        pos_model_path = os.path.join(config['model_output_directory'], 'rf_model_positive.joblib')
        neg_model_path = os.path.join(config['model_output_directory'], 'rf_model_negative.joblib')
        
        if not os.path.exists(pos_model_path) or not os.path.exists(neg_model_path):
            logging.error("Model files not found. Please train the models first.")
            print("Error: Model files not found. Please train the models first.")
            return
        
        # Predict using both models
        predictions_df = predict_dual_models(
            input_directory=config['input_directory'],
            pos_model_path=pos_model_path,
            neg_model_path=neg_model_path,
            output_directory=config['prediction_output_directory'],
            target_column=config['target_column'],
            date_column=config['date_column'],
            confidence_threshold_pos=args.threshold_pos,
            confidence_threshold_neg=args.threshold_neg
        )
        
        # Optionally perform EDA on the prediction results
        if args.eda and predictions_df is not None:
            explore_prediction_data(predictions_df, config['prediction_output_directory'])


if __name__ == "__main__":
    main()