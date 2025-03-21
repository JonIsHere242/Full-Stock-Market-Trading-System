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
args = argparser.parse_args()




config = {
    "input_directory": "Data/IndicatorData",
    "model_output_directory": "Data/ModelData",
    "data_output_directory": "Data/ModelData/TrainingData",
    "prediction_output_directory": "Data/RFpredictions",
    "feature_importance_output": "Data/ModelData/FeatureImportances/feature_importance.parquet",
    "log_file": "data/logging/4__Predictor.log",
    "file_selection_percentage": args.runpercent,
    "target_column": "percent_change_Close",

    "n_estimators": 64,  # Increased to 256
    "criterion": "entropy",
    "max_depth": 8,  # Increased depth
    "min_samples_split": 10,  # Adjusted for better performance
    "min_samples_leaf": 5,  
    "min_weight_fraction_leaf": 0,
    "max_features": 0.10,  # Adjusted to use more features
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0,
    "bootstrap": True,
    "oob_score": True,  
    "random_state": 3301,
    "verbose": 2,
    "warm_start": False,
    "class_weight": {0: 0.9, 1: 2.0},  # Adjusted class weights
    "ccp_alpha": 0,  
    "max_samples": None
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

# In your prepare_training_data function, replace the old cleaning step with:
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



##====================================[working method]====================================##
##====================================[working method]====================================##
##====================================[working method]====================================##
##====================================[working method]====================================##
##====================================[working method]====================================##
##====================================[working method]====================================##



def train_random_forest(training_data, config, confidence_threshold_pos=0.70, confidence_threshold_neg=0.70):
    logging.info("Training Random Forest model.")
    
    # Remove the old model file
    model_output_path = os.path.join(config['model_output_directory'], 'random_forest_model.joblib')
    if os.path.exists(model_output_path):
        os.remove(model_output_path)

    # Sort data by date to ensure temporal order
    training_data = training_data.sort_values('Date')
    
    # Separate features and target
    X = training_data.drop(columns=[config['target_column']])
    y = training_data[config['target_column']]
    
    # Binarize the target column
    y = y.apply(lambda x: 1 if x > 0 else 0)
    
    # Get datetime columns before removing them from X
    datetime_columns = X.select_dtypes(include=['datetime64']).columns
    
    # Calculate the split point based on dates
    split_date = X['Date'].quantile(0.8)  # Using last 20% of dates for testing
    logging.info(f"Split date: {split_date}")
    
    # Split based on date
    X_train = X[X['Date'] < split_date]
    X_test = X[X['Date'] >= split_date]
    y_train = y[X['Date'] < split_date]
    y_test = y[X['Date'] >= split_date]
    
    # Now remove datetime columns from both train and test
    X_train = X_train.drop(columns=datetime_columns)
    X_test = X_test.drop(columns=datetime_columns)

    # Train the Random Forest classifier
    clf = RandomForestClassifier(
        n_estimators=config['n_estimators'],
        criterion=config['criterion'],
        max_depth=config['max_depth'],
        min_samples_split=config['min_samples_split'],
        min_samples_leaf=config['min_samples_leaf'],
        min_weight_fraction_leaf=config['min_weight_fraction_leaf'],
        max_features=config['max_features'],
        max_leaf_nodes=config['max_leaf_nodes'],
        min_impurity_decrease=config['min_impurity_decrease'],
        bootstrap=config['bootstrap'],
        oob_score=config['oob_score'],
        random_state=config['random_state'],
        verbose=config['verbose'],
        warm_start=config['warm_start'],
        class_weight=config['class_weight'],
        ccp_alpha=config['ccp_alpha'],
        max_samples=config['max_samples'],
        n_jobs=-1  # Use all available processors
    )
    
    clf.fit(X_train, y_train)
    
    # Make predictions with confidence thresholds for both classes
    y_pred_proba = clf.predict_proba(X_test)
    y_pred = np.where(
        y_pred_proba[:, 1] >= confidence_threshold_pos, 1,
        np.where(y_pred_proba[:, 0] >= confidence_threshold_neg, 0, -1)
    )
    
    # Filter out undecided predictions
    mask = y_pred != -1
    y_test_filtered = y_test[mask]
    y_pred_filtered = y_pred[mask]
    
    # Evaluate the model
    accuracy = accuracy_score(y_test_filtered, y_pred_filtered)
    f1 = f1_score(y_test_filtered, y_pred_filtered, average='weighted')
    precision = precision_score(y_test_filtered, y_pred_filtered, average='weighted')
    recall = recall_score(y_test_filtered, y_pred_filtered, average='weighted')
    
    logging.info(f"Accuracy: {accuracy}")
    logging.info(f"F1 Score: {f1}")
    logging.info(f"Precision: {precision}")
    logging.info(f"Recall: {recall}")
    
    if len(y_test_filtered) == 0 or len(y_pred_filtered) == 0:
        print("Warning: y_test_filtered or y_pred_filtered is empty")
    else:
        # Ensure there are unique labels in y_test_filtered
        unique_labels = set(y_test_filtered)
        if len(unique_labels) == 0:
            print("Warning: No unique labels in y_test_filtered")
        else:
            # Ensure y_pred_filtered has corresponding predictions
            unique_pred_labels = set(y_pred_filtered)
            if len(unique_pred_labels) == 0:
                print("Warning: No unique predictions in y_pred_filtered")
            else:
                print(classification_report(y_test_filtered, y_pred_filtered, zero_division=0))    

    # Save the model
    dump(clf, model_output_path)
    logging.info(f"Model saved to {model_output_path}")
    
    # Save feature importances
    feature_importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': clf.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    feature_importances['importance'] = feature_importances['importance'].round(5)
    feature_importances.to_parquet(config['feature_importance_output'], index=False)
    logging.info(f"Feature importances saved to {config['feature_importance_output']}")














def predict_and_save(input_directory, model_path, output_directory, target_column, date_column, confidence_threshold_pos=0.7, confidence_threshold_neg=0.7):
    logging.info("Loading the trained model.")
    
    # Remove all the parquet files in the output directory
    for file in os.listdir(output_directory):
        if file.endswith('.parquet'):
            os.remove(os.path.join(output_directory, file))
    
    # Load the trained model
    clf = load(model_path)
    
    # Get the feature names the model was trained on
    model_features = clf.feature_names_in_
    
    # Get the list of files to process
    all_files = [f for f in os.listdir(input_directory) if f.endswith('.parquet')]
    
    # Initialize progress bar
    pbar = tqdm(total=len(all_files), desc="Processing files", ncols=100)
    
    # Set up a custom logger to capture joblib output
    joblib_logger = logging.getLogger('joblib')
    joblib_logger.setLevel(logging.ERROR)  # Only show errors
    
    # Redirect stdout and stderr
    null_io = io.StringIO()
    
    for file in all_files:
        df = pd.read_parquet(os.path.join(input_directory, file))
        # Ensure the date column is in datetime format
        df[date_column] = pd.to_datetime(df[date_column])
        
        ##check to see if file has more than 252 rows
        if df.shape[0] < 252:
            continue

        # Remove datetime and target columns from X
        datetime_columns = df.select_dtypes(include=['datetime64']).columns
        X = df.drop(columns=[date_column, target_column] + list(datetime_columns))
        
        # Align the features with the model's expected features
        X = X.reindex(columns=model_features, fill_value=0)
        
        # Make predictions with probabilities
        with parallel_backend('threading', n_jobs=-1):
            with redirect_stdout(null_io), redirect_stderr(null_io):
                y_pred_proba = clf.predict_proba(X)
        
        # Append predictions to dataframe
        df['UpProbability'] = y_pred_proba[:, 1]
        df['UpPrediction'] = (df['UpProbability'] >= confidence_threshold_pos).astype(int)
        
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume','Distance to Resistance (%)', 'Distance to Support (%)', 'UpProbability', 'UpPrediction']]

        output_file_path = os.path.join(output_directory, file)
        df.to_parquet(output_file_path, index=False)
        
        pbar.update(1)
    pbar.close()
    logging.info(f"Predictions saved to {output_directory}")


def main():
    setup_logging(config)
    if not args.predict:
        training_data = prepare_training_data(
            input_directory=config['input_directory'],
            output_directory=config['data_output_directory'],
            file_selection_percentage=config['file_selection_percentage'],
            target_column=config['target_column'],
            reuse=args.reuse,
            date_column='Date'  # Assuming 'Date' is the name of your date column
        )
        logging.info("Data preparation complete.")
        
        train_random_forest(training_data, config)
    else:
        predict_and_save(
            input_directory=config['input_directory'],
            model_path=os.path.join(config['model_output_directory'], 'random_forest_model.joblib'),
            output_directory=config['prediction_output_directory'],
            target_column=config['target_column'],
            date_column='Date'
        )

if __name__ == "__main__":
    main()