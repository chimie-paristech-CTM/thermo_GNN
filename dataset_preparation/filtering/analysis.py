import pandas as pd
import numpy as np
import argparse


def load_data(file_path):
    """Loads the CSV file and returns the dataframe."""
    return pd.read_csv(file_path)


def calculate_rmse_mae(true_labels, predicted_labels):
    """Calculates and returns the RMSE and MAE."""
    rmse = np.sqrt(np.mean((true_labels - predicted_labels) ** 2))
    mae = np.mean(np.abs(true_labels - predicted_labels))
    return rmse, mae


def detect_upper_outliers(data, factor=3.0):
    """Detects and returns a boolean series indicating the upper outliers."""
    Q3 = data.quantile(0.75)
    IQR = Q3 - data.quantile(0.25)
    upper_bound = Q3 + factor * IQR
    return data > upper_bound


def save_to_csv(dataframe, file_path):
    """Saves the dataframe to a CSV file."""
    dataframe.to_csv(file_path, index=False)


def main(input_file, cleaned_output_file, outliers_output_file):
    # Load data
    file = load_data(input_file)
    smiles = file['smiles']
    true_label = file['true_label']
    pre_label = file['pre_label']

    # Calculate initial RMSE and MAE
    rmse, mae = calculate_rmse_mae(true_label, pre_label)
    print(f'Initial - RMSE: {rmse}, MAE: {mae}')

    # Detect outliers
    errors = np.abs(true_label - pre_label)
    upper_outliers_errors = detect_upper_outliers(errors)

    # Retrieve outliers' information
    upper_outliers_indices = np.where(upper_outliers_errors)[0]
    upper_outliers_info = file.iloc[upper_outliers_indices].copy()  # Use copy() to avoid SettingWithCopyWarning
    print(f'Error Upper Outliers: {upper_outliers_errors.sum()}')
    print('Upper Outliers Information:')
    print(upper_outliers_info)

    # Remove outliers and recalculate RMSE and MAE
    cleaned_data = file.drop(upper_outliers_indices).reset_index(drop=True)
    cleaned_true_label = cleaned_data['true_label']
    cleaned_pre_label = cleaned_data['pre_label']
    rmse_cleaned, mae_cleaned = calculate_rmse_mae(cleaned_true_label, cleaned_pre_label)
    print(f'Cleaned - RMSE: {rmse_cleaned}, MAE: {mae_cleaned}')

    # Sort cleaned data by bias and save to CSV
    cleaned_data['bias'] = np.abs(cleaned_true_label - cleaned_pre_label)
    sorted_cleaned_data = cleaned_data.sort_values(by='bias', ascending=False).reset_index(drop=True)
    save_to_csv(sorted_cleaned_data, cleaned_output_file)

    # Sort outliers data by bias and save to CSV
    upper_outliers_info['bias'] = np.abs(upper_outliers_info['true_label'] - upper_outliers_info['pre_label'])
    sorted_upper_outliers_data = upper_outliers_info.sort_values(by='bias', ascending=False).reset_index(drop=True)
    save_to_csv(sorted_upper_outliers_data, outliers_output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and analyze prediction data.")
    parser.add_argument('--input_file', type=str, default='predictions.csv', help='Path to the input CSV file')
    parser.add_argument('--cleaned_output_file', type=str, default='./pc9.csv',
                        help='Path to save the cleaned data CSV file')
    parser.add_argument('--outliers_output_file', type=str,
                        default='./sorted_upper_outliers_output_pc9_val_formation.csv',
                        help='Path to save the outliers data CSV file')

    args = parser.parse_args()
    main(args.input_file, args.cleaned_output_file, args.outliers_output_file)
