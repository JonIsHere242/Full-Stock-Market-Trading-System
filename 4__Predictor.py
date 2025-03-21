##predictor script


import os
import random
import pandas as pd
import numpy as np
import logging
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from joblib import dump, load
import argparse
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm
from joblib import parallel_backend
from contextlib import redirect_stdout, redirect_stderr
import io
from Util import get_logger
logger = get_logger(script_name="4__Predictor")


try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not installed. Run 'pip install catboost' to enable CatBoost mode.")




argparser = argparse.ArgumentParser()
argparser.add_argument("--runpercent", type=int, default=50, help="Percentage of files to process.")
argparser.add_argument("--clear", action='store_true', help="Flag to clear the model and data directories.")
argparser.add_argument("--predict", action='store_true', help="Flag to predict new data.")
argparser.add_argument("--reuse", action='store_true', help="Flag to reuse existing training data if available.")
argparser.add_argument("--model", type=str, default="xgb", choices=["xgb", "cat"], help="Model type to use: xgb (XGBoost) or cat (CatBoost).")
args = argparser.parse_args()


config = {
    "input_directory": "Data/IndicatorData",
    "model_output_directory": "Data/ModelData",
    "data_output_directory": "Data/ModelData/TrainingData",
    "prediction_output_directory": "Data/RFpredictions",
    "feature_importance_output": "Data/ModelData/FeatureImportances/feature_importance.parquet",
    "log_file": "data/logging/4__XGBoostPredictor.log",
    "file_selection_percentage": args.runpercent,
    "target_column": "percent_change_Close",
    "model_type": args.model,
    
    # XGBoost parameters
    "xgb_params": {
        "n_estimators": 1024,
        "max_depth": 8,
        "learning_rate": 0.1,
        "gamma": 0.1,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "scale_pos_weight": 2.0,
        "random_state": 3301,
        "verbosity": 2,
        "use_label_encoder": False
    },
    
    # CatBoost parameters
    "cat_params": {
        "iterations": 4096,              # Balanced for learning complex patterns without overfitting
        "depth": 6,                      # Slightly deeper trees to capture more complex patterns
        "learning_rate": 0.02,           # Slower learning rate for better precision
        "l2_leaf_reg": 2.0,              # Moderate regularization
        "model_size_reg": 0.5,           # Balanced tree size penalty
        "grow_policy": "Lossguide",      # Better for precise predictions
        "has_time": True,                # Keep this for time series data
        "eval_metric": "Precision",      # Focus on precision instead of AUC
        "random_seed": 3301,
        "early_stopping_rounds": 10,     # More patience for precision optimization
        "loss_function": "Focal",        # Keep focal loss for class imbalance
        "loss_function_params": {
            "alpha": 0.75,               # Increased focus on rare positive class
            "gamma": 1.0                # Higher gamma puts more focus on hard examples
        },
        "class_weights": [0.5, 3.0],     # Stronger emphasis on class 1 predictions
        "bootstrap_type": "Bernoulli",   # Good for imbalanced data
        "subsample": 0.8,                # Subsample data to reduce overfitting
        "feature_border_count": 128,     # Reasonable number of splits
        "min_data_in_leaf": 10,          # Smaller to allow more specific predictions
        "one_hot_max_size": 10,          # Conservative encoding
        "random_strength": 0.01,          # Less random noise for more stable predictions
        "nan_mode": "Min",               # Keep this for missing values
        "fold_permutation_block_size": 128 # Respect time series structure
    },
    
    # Common parameters
    "early_stopping_rounds": 10,
    "random_state": 3301
}





def drop_string_columns(df, date_column, target_column):
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
        if df.shape[0] > 50 and target_column in df.columns and date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            df = drop_string_columns(df, date_column, target_column)
            df[target_column] = df[target_column].shift(-1)
            df = df.iloc[2:-2]
            df = df.dropna(subset=[target_column])
            df = df[(df[target_column] <= 10000) & (df[target_column] >= -10000)]
            if not df.empty:
                all_data.append(df)
        pbar.update(1)
    pbar.close()
    
    if len(all_data) == 0:
        logging.error("No valid training data found after processing files. Check your data provider and cleaning logic.")
        raise ValueError("No valid training data found after processing files.")
    
    combined_df = pd.concat(all_data)
    grouped = combined_df.groupby(date_column)
    shuffled_groups = [group.sample(frac=1).reset_index(drop=True) for _, group in grouped]
    
    if len(shuffled_groups) == 0:
        logging.error("No groups available after grouping by date. Check your data's date values.")
        raise ValueError("No groups available after grouping by date.")
    
    final_df = pd.concat(shuffled_groups).reset_index(drop=True)
    final_df.to_parquet(output_file, index=False)
    return final_df



def train_model(training_data, config, target_precision=0.75):
    model_type = config['model_type']
    
    if model_type == "cat" and not CATBOOST_AVAILABLE:
        print("CatBoost not installed. Falling back to XGBoost.")
        model_type = "xgb"
        
    logging.info(f"Training {model_type.upper()} model.")
    
    model_filename = f"{model_type}_model.joblib"
    model_output_path = os.path.join(config['model_output_directory'], model_filename)
    if os.path.exists(model_output_path):
        os.remove(model_output_path)

    training_data = training_data.sort_values('Date')
    
    X = training_data.drop(columns=[config['target_column']])
    y = training_data[config['target_column']]
    
    y = y.apply(lambda x: 0 if x < 0 else 1)
    
    datetime_columns = X.select_dtypes(include=['datetime64']).columns
    
    split_date = X['Date'].quantile(0.8)
    logging.info(f"Split date: {split_date}")
    
    X_train = X[X['Date'] < split_date]
    X_test = X[X['Date'] >= split_date]
    y_train = y[X['Date'] < split_date]
    y_test = y[X['Date'] >= split_date]
    
    X_train = X_train.drop(columns=datetime_columns)
    X_test = X_test.drop(columns=datetime_columns)

    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=config['random_state']
    )
    
    if model_type == "xgb":
        # XGBoost model - adjust weights to make upward predictions more selective
        xgb_params = config['xgb_params'].copy()
        xgb_params["scale_pos_weight"] = 1.0  # Reduced from previous value to make upward predictions more selective
        clf = XGBClassifier(**xgb_params)
        
        # Remove early_stopping_rounds from fit() to avoid errors with older XGBoost versions
        try:
            clf.fit(
                X_train_final, y_train_final,
                eval_set=[(X_val, y_val)],
                verbose=True
            )
        except Exception as e:
            logging.error(f"Error during XGBoost fitting: {str(e)}")
            # Try without eval_set
            clf.fit(X_train_final, y_train_final)
    
    else:  # CatBoost model
        # Create a clean set of parameters for CatBoost
        # Adjust weights to make upward predictions more selective
        try:
            cat_params = {
                "iterations": config['cat_params'].get('iterations', 512),
                "depth": config['cat_params'].get('depth', 5),
                "learning_rate": config['cat_params'].get('learning_rate', 0.08),
                "l2_leaf_reg": config['cat_params'].get('l2_leaf_reg', 5),
                "random_seed": config['cat_params'].get('random_seed', 3301),
                "class_weights": [1.5, 1.0]  # Reversed to favor precision for class 1 (increase weight for class 0)
            }
            
            # Log the parameters being used
            logging.info(f"Attempting to initialize CatBoost with parameters: {cat_params}")
            clf = CatBoostClassifier(**cat_params)
            
            clf.fit(
                X_train_final, y_train_final,
                eval_set=(X_val, y_val),
                verbose=True
            )
        except Exception as e:
            logging.error(f"Error with CatBoost: {str(e)}")
            # Fall back to absolute minimal parameters but still with adjusted class weights
            logging.info("Falling back to minimal CatBoost parameters")
            clf = CatBoostClassifier(iterations=512, random_seed=3301, class_weights=[1.5, 1.0])
            clf.fit(X_train_final, y_train_final, verbose=True)
    
    # Get predicted probabilities
    y_pred_proba = clf.predict_proba(X_test)

    # First, check the distribution of probabilities
    logging.info(f"Probability stats: min={y_pred_proba[:, 1].min():.4f}, max={y_pred_proba[:, 1].max():.4f}, mean={y_pred_proba[:, 1].mean():.4f}")

    # We'll use precision-recall curve but with a focus on high precision for upward moves
    from sklearn.metrics import precision_recall_curve

    # Find thresholds that balance precision and coverage
    target_min_precision_pos = 0.75  # High precision target for upward moves (class 1)
    min_predictions_percent_pos = 0.01  # Very selective - only predict 1% of cases
    
    target_min_precision_neg = 0.65  # More lenient for downward moves (class 0)
    min_predictions_percent_neg = 0.05  # Higher coverage for downward moves

    # For positive class (class 1) - upward moves
    precisions_pos, recalls_pos, thresholds_pos = precision_recall_curve(
        y_test, y_pred_proba[:, 1], pos_label=1
    )

    # Ensure we don't have index mismatches (precision/recalls have one more element than thresholds)
    if len(precisions_pos) > len(thresholds_pos):
        precisions_pos = precisions_pos[:-1]
        recalls_pos = recalls_pos[:-1]

    # Calculate the percentage of data that would receive predictions at each threshold
    prediction_coverage = []
    for threshold in thresholds_pos:
        coverage = (y_pred_proba[:, 1] >= threshold).mean()
        prediction_coverage.append(coverage)

    # Find threshold that gives at least target precision and meets minimum coverage
    valid_indices = (precisions_pos >= target_min_precision_pos) & (np.array(prediction_coverage) >= min_predictions_percent_pos)

    if np.any(valid_indices):
        # For upward moves, prioritize precision over recall
        best_idx = np.argmax(precisions_pos[valid_indices])  # Changed to maximize precision
        valid_idx_positions = np.where(valid_indices)[0]
        optimal_threshold_pos = thresholds_pos[valid_idx_positions[best_idx]]
        pos_precision = precisions_pos[valid_idx_positions[best_idx]]
        pos_recall = recalls_pos[valid_idx_positions[best_idx]]
        pos_coverage = prediction_coverage[valid_idx_positions[best_idx]]
    else:
        # If no threshold meets both criteria, prioritize precision and relax coverage
        logging.warning("No threshold meets both high precision and coverage requirements for UP moves.")
        # Find the threshold with the highest precision that gives at least some predictions
        # Sort by precision and find the first threshold that gives at least a few predictions
        valid_thresholds = [i for i, cov in enumerate(prediction_coverage) if cov >= 0.001]
        if valid_thresholds:
            best_precision_idx = np.argmax(precisions_pos[valid_thresholds])
            idx_to_use = valid_thresholds[best_precision_idx]
            optimal_threshold_pos = thresholds_pos[idx_to_use]
            pos_precision = precisions_pos[idx_to_use]
            pos_recall = recalls_pos[idx_to_use]
            pos_coverage = prediction_coverage[idx_to_use]
        else:
            # If all else fails, use a very high threshold
            optimal_threshold_pos = 0.90  # Very high threshold to ensure high precision
            predicted_pos = y_pred_proba[:, 1] >= optimal_threshold_pos
            if predicted_pos.sum() > 0:
                pos_precision = (y_test[predicted_pos] == 1).mean()
            else:
                pos_precision = 0
            pos_recall = (predicted_pos & (y_test == 1)).sum() / (y_test == 1).sum() if (y_test == 1).sum() > 0 else 0
            pos_coverage = predicted_pos.mean()

    # For negative class (class 0) - can be more lenient
    precisions_neg, recalls_neg, thresholds_neg = precision_recall_curve(
        1 - y_test, y_pred_proba[:, 0], pos_label=1
    )

    # Ensure we don't have index mismatches
    if len(precisions_neg) > len(thresholds_neg):
        precisions_neg = precisions_neg[:-1]
        recalls_neg = recalls_neg[:-1]

    # Calculate the coverage at each threshold
    neg_prediction_coverage = []
    for threshold in thresholds_neg:
        coverage = (y_pred_proba[:, 0] >= threshold).mean()
        neg_prediction_coverage.append(coverage)

    # Use the same logic as for positive class but with different targets
    valid_indices = (precisions_neg >= target_min_precision_neg) & (np.array(neg_prediction_coverage) >= min_predictions_percent_neg)

    if np.any(valid_indices):
        # For downward moves, we can balance precision and recall
        best_idx = np.argmax(recalls_neg[valid_indices])  # Maximize recall for downward predictions
        valid_idx_positions = np.where(valid_indices)[0]
        optimal_threshold_neg = thresholds_neg[valid_idx_positions[best_idx]]
        neg_precision = precisions_neg[valid_idx_positions[best_idx]]
        neg_recall = recalls_neg[valid_idx_positions[best_idx]]
        neg_coverage = neg_prediction_coverage[valid_idx_positions[best_idx]]
    else:
        # Relax requirements for negative class as well
        reduced_precision = max(0.60, target_min_precision_neg - 0.05)
        valid_indices = (precisions_neg >= reduced_precision) & (np.array(neg_prediction_coverage) >= min_predictions_percent_neg)

        if np.any(valid_indices):
            best_idx = np.argmax(precisions_neg[valid_indices])
            valid_idx_positions = np.where(valid_indices)[0]
            optimal_threshold_neg = thresholds_neg[valid_idx_positions[best_idx]]
            neg_precision = precisions_neg[valid_idx_positions[best_idx]]
            neg_recall = recalls_neg[valid_idx_positions[best_idx]]
            neg_coverage = neg_prediction_coverage[valid_idx_positions[best_idx]]
        else:
            # Fallback approach for negative class
            mean_prob = y_pred_proba[:, 0].mean()
            std_prob = y_pred_proba[:, 0].std()
            optimal_threshold_neg = min(0.75, mean_prob + std_prob)

            predicted_neg = y_pred_proba[:, 0] >= optimal_threshold_neg
            if predicted_neg.sum() > 0:
                neg_precision = ((1 - y_test)[predicted_neg] == 1).mean()
            else:
                neg_precision = 0
            neg_recall = (predicted_neg & ((1 - y_test) == 1)).sum() / ((1 - y_test) == 1).sum() if ((1 - y_test) == 1).sum() > 0 else 0
            neg_coverage = predicted_neg.mean()

    # Calculate expected profit factor for upward predictions
    if pos_precision > 0:
        expected_profit_factor = (pos_precision / (1 - pos_precision))
        logging.info(f"Expected profit factor for UP predictions: {expected_profit_factor:.4f}")
        logging.info(f"This means for every $1 lost, you can expect to make ${expected_profit_factor:.2f}")

    # Log detailed information about the thresholds
    logging.info(f"Optimal threshold for class 1 (UP): {optimal_threshold_pos:.4f} with precision {pos_precision:.4f}, recall {pos_recall:.4f}, coverage {pos_coverage:.4f}")
    logging.info(f"Optimal threshold for class 0 (DOWN): {optimal_threshold_neg:.4f} with precision {neg_precision:.4f}, recall {neg_recall:.4f}, coverage {neg_coverage:.4f}")

    # Apply the optimal thresholds - but ensure we get a reasonable number of predictions
    y_pred = np.full(len(y_test), -1)  # Default to "no prediction" (-1)

    # Assign class 1 if probability exceeds its threshold
    y_pred[y_pred_proba[:, 1] >= optimal_threshold_pos] = 1

    # For remaining unassigned predictions, assign class 0 if probability exceeds its threshold
    mask_unassigned = (y_pred == -1)
    y_pred[mask_unassigned & (y_pred_proba[:, 0] >= optimal_threshold_neg)] = 0

    # Calculate what percentage of the test set receives predictions
    prediction_coverage = (y_pred != -1).mean() * 100
    logging.info(f"Percentage of data receiving predictions: {prediction_coverage:.2f}%")

    # If we have extremely few predictions, check if this is because of high precision requirements
    if prediction_coverage < 0.5:  # Less than 0.5% of data gets predictions
        logging.warning(f"Very few predictions made ({prediction_coverage:.2f}%). This is likely due to the high precision requirement of {target_min_precision_pos*100:.1f}%.")
        
        # Check if we're making at least some upward predictions
        upward_preds = (y_pred == 1).sum()
        if upward_preds == 0:
            logging.warning("No upward predictions made. Consider slightly reducing the precision requirement or increasing the coverage threshold.")

    # Debugging: Plot histogram of probabilities to understand distribution
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.hist(y_pred_proba[:, 1], bins=50, alpha=0.7)
        plt.axvline(x=optimal_threshold_pos, color='r', linestyle='--', label=f'UP Threshold: {optimal_threshold_pos:.4f}')
        plt.title('Histogram of Predicted Probabilities for Upward Moves')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(config['model_output_directory'], 'probability_histogram.png'))
        logging.info(f"Probability histogram saved to {os.path.join(config['model_output_directory'], 'probability_histogram.png')}")
    except Exception as e:
        logging.error(f"Error creating probability histogram: {str(e)}")

    # Evaluation using only definitive predictions (excluding -1)
    mask_definitive = y_pred != -1
    y_test_filtered = y_test[mask_definitive]
    y_pred_filtered = y_pred[mask_definitive]
    
    if len(y_test_filtered) > 0:
        accuracy = accuracy_score(y_test_filtered, y_pred_filtered)
        f1 = f1_score(y_test_filtered, y_pred_filtered, average='weighted')
        precision = precision_score(y_test_filtered, y_pred_filtered, average='weighted')
        recall = recall_score(y_test_filtered, y_pred_filtered, average='weighted')
        
        logging.info(f"Definitive predictions: {len(y_pred_filtered)} out of {len(y_pred)} ({len(y_pred_filtered)/len(y_pred)*100:.2f}%)")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"F1 Score: {f1:.4f}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        
        # Print class-specific metrics 
        print(classification_report(y_test_filtered, y_pred_filtered, zero_division=0))
        
        # Calculate and report class-specific metrics separately for better insight
        if (y_pred_filtered == 1).sum() > 0:
            up_precision = precision_score(y_test_filtered, y_pred_filtered, pos_label=1, average='binary')
            up_recall = recall_score(y_test_filtered, y_pred_filtered, pos_label=1, average='binary')
            logging.info(f"UP predictions precision: {up_precision:.4f}, recall: {up_recall:.4f}")
            logging.info(f"Total UP predictions: {(y_pred_filtered == 1).sum()} out of {len(y_pred_filtered)} ({(y_pred_filtered == 1).sum()/len(y_pred_filtered)*100:.2f}%)")
    else:
        logging.warning("No definitive predictions after applying thresholds.")
    
    # Save model and thresholds
    model_data = {
        'model': clf,
        'threshold_pos': optimal_threshold_pos,
        'threshold_neg': optimal_threshold_neg,
        'precision_pos': pos_precision,
        'recall_pos': pos_recall,
        'precision_neg': neg_precision,
        'recall_neg': neg_recall
    }
    
    dump(model_data, model_output_path)
    logging.info(f"Model and thresholds saved to {model_output_path}")
    
    # Handle feature importances appropriately for each model type
    try:
        if model_type == "xgb":
            feature_importances = pd.DataFrame({
                'feature': X_train.columns,
                'importance': clf.feature_importances_
            }).sort_values(by='importance', ascending=False)
        else:  # CatBoost
            feature_importances = pd.DataFrame({
                'feature': X_train.columns,
                'importance': clf.get_feature_importance()
            }).sort_values(by='importance', ascending=False)
        
        feature_importances['importance'] = feature_importances['importance'].round(5)
        feature_importances.to_parquet(config['feature_importance_output'], index=False)
        logging.info(f"Feature importances saved to {config['feature_importance_output']}")
    except Exception as e:
        logging.error(f"Error saving feature importances: {str(e)}")
    
    return model_data








def predict_and_save(input_directory, model_path, output_directory, target_column, date_column):
    logging.info("Loading the trained model and optimized thresholds for prediction.")
    
    for file in os.listdir(output_directory):
        if file.endswith('.parquet'):
            os.remove(os.path.join(output_directory, file))
    
    # Load model and thresholds
    model_data = load(model_path)
    
    if isinstance(model_data, dict) and 'model' in model_data:
        clf = model_data['model']
        threshold_pos = model_data['threshold_pos']
        threshold_neg = model_data['threshold_neg']
        logging.info(f"Using optimized thresholds - Positive: {threshold_pos:.4f}, Negative: {threshold_neg:.4f}")
    else:
        # For backward compatibility with old model files
        clf = model_data
        threshold_pos = 0.7  # Default
        threshold_neg = 0.7  # Default
        logging.warning("Using default thresholds as no optimized thresholds found")
    
    # Handle different model types properly
    if hasattr(clf, 'feature_names_in_'):
        # XGBoost or scikit-learn model with feature_names_in_
        model_features = clf.feature_names_in_
    elif hasattr(clf, 'get_booster') and hasattr(clf.get_booster(), 'feature_names'):
        # XGBoost specific
        model_features = clf.get_booster().feature_names
    elif hasattr(clf, 'feature_names_'):
        # For CatBoost
        model_features = clf.feature_names_
    else:
        # Generic fallback - this might not work but we'll try
        logging.warning("Could not determine feature names from model, using all features.")
        sample_file = os.path.join(input_directory, os.listdir(input_directory)[0])
        sample_df = pd.read_parquet(sample_file)
        datetime_columns = sample_df.select_dtypes(include=['datetime64']).columns
        model_features = [col for col in sample_df.columns if col not in [date_column, target_column] + list(datetime_columns)]
    
    all_files = [f for f in os.listdir(input_directory) if f.endswith('.parquet')]
    pbar = tqdm(total=len(all_files), desc="Processing files", ncols=100)
    
    joblib_logger = logging.getLogger('joblib')
    joblib_logger.setLevel(logging.ERROR)
    
    null_io = io.StringIO()
    
    # Track prediction performance statistics
    total_predictions = 0
    definitive_predictions = 0
    
    for file in all_files:
        df = pd.read_parquet(os.path.join(input_directory, file))
        df[date_column] = pd.to_datetime(df[date_column])
        
        if df.shape[0] < 252:
            pbar.update(1)
            continue

        # Calculate volatility for additional information
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std().fillna(0)
        
        # Prepare features for prediction
        datetime_columns = df.select_dtypes(include=['datetime64']).columns
        X = df.drop(columns=[col for col in [date_column, target_column, 'returns', 'volatility'] + list(datetime_columns) if col in df.columns])
        
        # Ensure X contains only the features the model was trained on
        missing_features = set(model_features) - set(X.columns)
        if missing_features:
            for feature in missing_features:
                X[feature] = 0  # Add missing features with default values
                
        X = X.reindex(columns=model_features, fill_value=0)
        
        # Make predictions
        with parallel_backend('threading', n_jobs=-1):
            with redirect_stdout(null_io), redirect_stderr(null_io):
                try:
                    y_pred_proba = clf.predict_proba(X)
                except Exception as e:
                    logging.error(f"Error making predictions: {str(e)}")
                    logging.error(f"Model type: {type(clf).__name__}")
                    pbar.update(1)
                    continue
        
        
        # Correct assignment of probabilities
        df['UpProbability'] = y_pred_proba[:, 1]  # Use class 1 probability as up (positive change)
        df['DownProbability'] = y_pred_proba[:, 0]  # Use class 0 probability as down (negative change)
        
        # Then use your original threshold logic
        df['UpPrediction'] = -1  # Default to no prediction
        df.loc[df['UpProbability'] >= threshold_pos, 'UpPrediction'] = 1
        mask_undecided = df['UpPrediction'] == -1
        df.loc[mask_undecided & (df['DownProbability'] >= threshold_neg), 'UpPrediction'] = 0
                
        # Update prediction stats
        total_predictions += len(df)
        definitive_predictions += (df['UpPrediction'] != -1).sum()
        
        # Keep necessary columns for output
        try:
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 
                             'UpProbability', 'DownProbability', 
                             'PositiveThreshold', 'NegativeThreshold', 'UpPrediction']
            
            # Try to include these columns if they exist
            optional_columns = ['Distance to Resistance (%)', 'Distance to Support (%)', 'volatility']
            for col in optional_columns:
                if col in df.columns:
                    required_columns.append(col)
                    
            # Check which columns actually exist
            available_columns = [col for col in required_columns if col in df.columns]
            
            # Add any missing required columns with NaN values
            for col in set(required_columns) - set(available_columns):
                df[col] = np.nan
                
            output_df = df[required_columns]
            
            output_file_path = os.path.join(output_directory, file)
            output_df.to_parquet(output_file_path, index=False)
        except Exception as e:
            logging.error(f"Error saving prediction for {file}: {str(e)}")
        
        pbar.update(1)
    
    pbar.close()
    
    # Report prediction coverage
    if total_predictions > 0:
        prediction_rate = (definitive_predictions / total_predictions) * 100
        logging.info(f"Prediction coverage: {definitive_predictions} out of {total_predictions} ({prediction_rate:.2f}%)")
    
    logging.info(f"Predictions using optimized thresholds saved to {output_directory}")



def main():
    
    
    model_type = config['model_type']
    model_filename = f"{model_type}_model.joblib"
    
    if not args.predict:
        training_data = prepare_training_data(
            input_directory=config['input_directory'],
            output_directory=config['data_output_directory'],
            file_selection_percentage=config['file_selection_percentage'],
            target_column=config['target_column'],
            reuse=args.reuse,
            date_column='Date'
        )
        logging.info("Data preparation complete.")
        
        train_model(training_data, config)
    else:
        predict_and_save(
            input_directory=config['input_directory'],
            model_path=os.path.join(config['model_output_directory'], model_filename),
            output_directory=config['prediction_output_directory'],
            target_column=config['target_column'],
            date_column='Date'
        )

if __name__ == "__main__":
    main()