import subprocess
import argparse
import logging
import asyncio
import sys
import os
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

class CommandExecutor:
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Asynchronously run a command
    async def run_command(self, command):
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            logging.error(f"Command failed with return code {process.returncode}")
            logging.error(stderr.decode())
        else:
            logging.info(stdout.decode())

    # Asynchronously execute a list of commands
    async def execute_commands(self, commands):
        for command in commands:
            await self.run_command(command)

class DataProcessor:
    def __init__(self):
        pass

    # Static method to list subdirectories within a directory
    @staticmethod
    def list_subdirectories(directory):
        subdirectories = []
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                subdirectories.append(item_path)
        return subdirectories

    # Static method to process data
    @staticmethod
    def process_data(data_path, directory):
        # Processing the train_val.csv file
        train_val_processed_path = os.path.join(directory, data_path, 'train_val_process.csv')
        if not os.path.exists(train_val_processed_path):
            df_train_val = pd.read_csv(os.path.join(directory, data_path, 'train_val.csv'))
            df_train_val = df_train_val.drop(df_train_val.columns[0], axis=1)
            df_train_val.to_csv(train_val_processed_path, index=False)
        # Processing the test.csv file
        test_processed_path = os.path.join(directory, data_path, 'test_process.csv')
        if not os.path.exists(test_processed_path):
            df_test = pd.read_csv(os.path.join(directory, data_path, 'test.csv'))
            df_test = df_test.drop(df_test.columns[0], axis=1)
            df_test.to_csv(test_processed_path, index=False)

class ModelTrainer:
    def __init__(self):
        pass

    # Asynchronously run a training script
    @staticmethod
    async def run_training_script(data_path, dataset_type, fingerprint, save_dir, epochs, head, message, separate_val, separate_test, cpu_cores=None):
        taskset_command = f"taskset -c {cpu_cores}" if cpu_cores else ""  # Add taskset command if CPU cores specified
        # path_trainfile = os.path.join(os.getcwd(), "train.py")
        script_path = os.path.abspath(__file__)
        path_trainfile = os.path.join(os.path.dirname(script_path), "train.py")
        command = f"{taskset_command} python {path_trainfile} --data_path {data_path} --dataset_type {dataset_type} --fingerprint {fingerprint} --save_dir {save_dir} --epochs {epochs} --head {head} --message {message}"
        if separate_val.split('/')[-1] != "nothing" and separate_test.split('/')[-1] == "nothing":
            command += f" --separate_val_path {separate_val}"
        if separate_test.split('/')[-1] != "nothing" and separate_val.split('/')[-1] == "nothing":
            command += f" --separate_test_path {separate_test}"
        if separate_val.split('/')[-1] != "nothing" and separate_test.split('/')[-1] != "nothing":
            command += f" --separate_test_path {separate_test} --separate_val_path {separate_val}"
        logging.info(f"------conmman:   {command}   ++++++++")
        # print(f"command:\n\n {command}\n\n")
        await CommandExecutor().run_command(command)
class ModelPredictor:
    def __init__(self):
        pass

    # Asynchronously run a predicting script
    @staticmethod
    async def run_predicting_script(data_path, dataset_type, fingerprint, preds_path, model, checkpoint_path):
        path_predictfile = os.path.join(os.getcwd(), "chemprop", "predict.py")
        command = f"python {path_predictfile} --test_path {data_path} --fingerprint {fingerprint} --preds_path {preds_path} --head {model} --checkpoint_path {checkpoint_path}"
        logging.info(f"predict------{command}++++++++")
        await CommandExecutor().run_command_with_cpu_affinity(command)

class ModelEvaluator:
    def __init__(self):
        pass

    # Asynchronously calculate evaluation metrics
    @staticmethod
    async def calculate(preds_path, data_path, dataset_type):
        file_path = preds_path
        parts = file_path.split("/")
        parts[-1] = "test_process.csv"
        file_path = "/".join(parts)
        column_index = 1

        # Assuming check_column, calculate_classification_errors, and calculate_regression_errors are defined elsewhere
        # Here we assume these functions return valid results for demonstration purposes
        if run_chemprop.check_column(file_path, column_index):
            print("classification.")
            result = ModelEvaluator.calculate_classification_errors(file_path, preds_path)
        else:
            print("regression.")
            result = ModelEvaluator.calculate_regression_errors(file_path, preds_path)

        # add pre_path column
        result['pre_path'] = file_path.split("/")[2]

        # convert result to DataFrame
        df = pd.DataFrame([result])
        #column
        if not os.path.exists('classification.csv') and not os.path.exists('regression.csv'):
            if dataset_type == "classification":
                result_names = ["pre_path", "Dataset", "Total predictions", "Correct predictions", "Accuracy", "Precision",
                                "Recall", "F1 Score", "AUC", "Balanced AUC"]
                with open('classification.csv', 'w') as f:
                    f.write(','.join(result_names) + '\n')
            elif dataset_type == "regression":
                result_names = ["pre_path", "Dataset", "MAE", "RMSE", "MSE"]
                with open('regression.csv', 'w') as f:
                    f.write(','.join(result_names) + '\n')
            else:
                raise ValueError("Invalid dataset_type. It should be either classification or regression.")

        # Save as CSV file
        if dataset_type == "classification":
            output_file_path_class = os.path.join(os.path.dirname(__file__), 'classification.csv')
            mode_c = 'a' if os.path.exists(output_file_path_class) else 'w'
            header_c = False if os.path.exists(output_file_path_class) else True
            df.to_csv(output_file_path_class, mode=mode_c, index=False, header=header_c)
        elif dataset_type == "regression":
            output_file_path_regree = os.path.join(os.path.dirname(__file__), 'regression.csv')
            mode_r = 'a' if os.path.exists(output_file_path_regree) else 'w'
            header_r = False if os.path.exists(output_file_path_regree) else True
            df.to_csv(output_file_path_regree, mode=mode_r, index=False, header=header_r)
        else:
            raise ValueError("Invalid dataset_type. It should be either classification or regression.")

    # Calculate AUC
    @staticmethod
    def calculate_auc(predictions, labels):
        y_true = np.array(list(labels.values()))
        y_scores = np.array(list(predictions.values()))
        auc = roc_auc_score(y_true, y_scores)
        return auc

    # Calculate balanced AUC
    @staticmethod
    def calculate_balanced_auc(predictions, labels):
        # Determine class distribution
        num_positives = sum(1 for label in labels.values() if label == 1)
        num_negatives = sum(1 for label in labels.values() if label == 0)
        class_ratio = num_positives / num_negatives

        # Calculate AUC with class weights
        y_true = np.array(list(labels.values()))
        y_scores = np.array(list(predictions.values()))
        weights = np.where(y_true == 1, class_ratio, 1)
        balanced_auc = roc_auc_score(y_true, y_scores, sample_weight=weights)
        return balanced_auc

    # Read predictions from file
    @staticmethod
    def read_predictions(file_path):
        predictions = {}
        with open(file_path, 'r') as file:
            next(file)  # Skip the first line
            lines = file.readlines()
            for line in lines:
                if line.strip():  # Skip empty lines
                    drug, prediction = line.strip().split(',')
                    if prediction != 'Y':  # Skip lines with label 'Y'
                        predictions[drug] = float(prediction)
        return predictions

    # Read labels from file
    @staticmethod
    def read_labels(file_path, error):
        labels = {}
        with open(file_path, 'r') as file:
            next(file)
            lines = file.readlines()
            for line in lines:
                if line.strip():  # Skip empty lines
                    drug, label = line.strip().split(',')
                    label = float(label)
                    if error == "regression":
                        labels[drug] = float(label)
                    else:
                        labels[drug] = int(label)
        return labels

    # Analyze predictions and calculate evaluation metrics for classification
    @staticmethod
    def analyze_predictions(predictions, labels):
        correct = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0

        for drug, prediction in predictions.items():
            if labels.get(drug) == round(prediction):
                correct += 1
                if round(prediction) == 1:
                    true_positive += 1
            else:
                if round(prediction) == 1:
                    false_positive += 1
                else:
                    false_negative += 1

        accuracy = correct / len(predictions) * 100
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        auc = ModelEvaluator.calculate_auc(predictions, labels)
        balanced_auc = ModelEvaluator.calculate_balanced_auc(predictions, labels)

        result = {
            "Total predictions": len(predictions),
            "Correct predictions": correct,
            "Accuracy": f"{accuracy:.4f}%",
            "Precision": f"{precision:.4f}",
            "Recall": f"{recall:.4f}",
            "F1 Score": f"{f1_score:.4f}",
            "AUC": f"{auc:.4f}",
            "Balanced AUC": f"{balanced_auc:.4f}"
        }

        return result

    # Calculate evaluation metrics for classification
    @staticmethod
    def calculate_classification_errors(file_path, preds_path):
        predictions_file = preds_path
        labels_file = file_path

        predictions = ModelEvaluator.read_predictions(predictions_file)
        labels = ModelEvaluator.read_labels(labels_file, error=" classification")

        result = ModelEvaluator.analyze_predictions(predictions, labels)

        return result

    # Calculate evaluation metrics for regression
    @staticmethod
    def calculate_regression_errors(file_path, preds_path):
        squared_errors = []
        predictions_file = preds_path
        labels_file = file_path

        predictions = ModelEvaluator.read_predictions(predictions_file)
        labels = ModelEvaluator.read_labels(labels_file, error="regression")

        for drug, prediction in predictions.items():
            # Calculate the error and add it to the list
            squared_error = (labels.get(drug) - prediction) ** 2
            squared_errors.append(squared_error)

        # Calculate RMSE and MSE
        rmse = np.sqrt(np.mean(squared_errors))
        mse = np.mean(squared_errors)
        absolute_errors = []

        for drug, prediction in predictions.items():
            # Calculate the absolute error and add it to the list
            absolute_error = abs(labels.get(drug) - prediction)
            absolute_errors.append(absolute_error)
        # Calculate MAE, RMSE, and MSE
        mae = np.mean(absolute_errors)
        result = {
            "MAE": f"{mae:.4f}",
            "RMSE": f"{rmse:.4f}",
            "MSE": f"{mse:.4f}"
        }
        return result

class run_chemprop:
    def __init__(self):
        pass

    # Asynchronously run the main program
    @staticmethod
    async def run(args, directory, predict, log_file, epochs, head, message, save_name, fingerprint, cpu_cores=None):
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("Program started.")

        subdirectories = DataProcessor.list_subdirectories(directory)
        logging.info(f"Processing directory: {subdirectories}")
        logging.info(f"Number of subdirectories: {len(subdirectories)} and dictry {subdirectories}")
        for subdir in subdirectories:
            data_path = subdir
            logging.info(f"Processing directory: {data_path}")
            if args.predict:
                DataProcessor.process_data(data_path, directory)
            else:
                pass

            train_files, test_files, val_files = run_chemprop.find_train_test_val_files(data_path)
            if os.path.basename(train_files).split("/")[-1] == "nothing":
                print(f"{os.getcwd()} does not contain any train file")
                continue
            else:
                dataset_type = "classification" if run_chemprop.check_column(os.path.join( data_path, train_files), 1) else "regression"

            parameter_name = f"{save_name}_{fingerprint}_{epochs}_{head}_{message}"
            save_dir = os.path.join(data_path, parameter_name)
            log_file = os.path.join(save_dir, 'predict.log')
            data_path_train = os.path.join(data_path, train_files)
            data_path_test = os.path.join(data_path, test_files)
            data_path_val = os.path.join(data_path, val_files)
            if predict:
                await ModelTrainer.run_training_script(data_path_train, dataset_type, fingerprint, save_dir, epochs, head, message, data_path_val, data_path_test, cpu_cores)
                await ModelPredictor.run_predicting_script(data_path_train, dataset_type, fingerprint, f"{save_dir}.csv", head, f"{save_dir}/fold_0/model_0/model.pt")
                await ModelEvaluator.calculate(f"{save_dir}.csv", data_path, dataset_type)
            else:
                logging.info("using traindataset_======")
                await ModelTrainer.run_training_script(data_path_train, dataset_type, fingerprint, save_dir, epochs, head, message, data_path_val, data_path_test, cpu_cores)

        logging.info("Program finished.")

    @staticmethod
    async def run_multidatset_single(args, directory, predict, log_file, epochs, head, message, save_name, fingerprint, cpu_cores=None):
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("Program started.")
        logging.info(f"Processing directory: {directory}")
        # for subdir in subdirectories:
        data_path = directory
        logging.info(f"Processing directory: {data_path}")
        if args.predict:
            DataProcessor.process_data(data_path, directory)
        else:
            pass


        train_files, test_files, val_files = run_chemprop.find_train_test_val_files(data_path)
        # print(train_files,"--train",subdir,"\n")
        if os.path.basename(train_files).split("/")[-1] == "nothing":
            raise ValueError(f"{os.getcwd()} does not contain any train file")

        else:
            # print(f"{train_files}")
            dataset_type = "classification" if run_chemprop.check_column(os.path.join(data_path, train_files),
                                                                         1) else "regression"

        parameter_name = f"{save_name}_{fingerprint}_{epochs}_{head}_{message}"
        save_dir = os.path.join(data_path, parameter_name)

        log_file = os.path.join(save_dir, 'predict.log')
        data_path_train = os.path.join(data_path, train_files)
        data_path_test = os.path.join(data_path, test_files)
        data_path_val = os.path.join(data_path, val_files)
        #     if predict:
        #         await ModelTrainer.run_training_script(data_path_train, dataset_type, fingerprint, save_dir, epochs,
        #                                                head, message, data_path_val, data_path_test, cpu_cores)
        #         await ModelPredictor.run_predicting_script(data_path_train, dataset_type, fingerprint,
        #                                                    f"{save_dir}.csv", head,
        #                                                    f"{save_dir}/fold_0/model_0/model.pt")
        #         await ModelEvaluator.calculate(f"{save_dir}.csv", data_path, dataset_type)
        #     else:
        logging.info("using traindataset_======")
        await ModelTrainer.run_training_script(data_path_train, dataset_type, fingerprint, save_dir, epochs,
                                               head, message, data_path_val, data_path_test, cpu_cores)

        logging.info("Program finished.")

    # Find train, test, and validation files in a directory
    @staticmethod
    async def run_multidatset_multiple(args, directory, predict, log_file, epochs, head, message, save_name, fingerprint, cpu_cores=None):
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("Program started.")
        logging.info(f"Processing directory: {directory}")
        # for subdir in subdirectories:
        data_path = directory
        logging.info(f"Processing directory: {data_path}")
        if args.predict:
            DataProcessor.process_data(data_path, directory)
        else:
            pass


        train_files, test_files, val_files = run_chemprop.find_train_test_val_files(data_path)
        # print(train_files,"--train",subdir,"\n")
        if os.path.basename(train_files).split("/")[-1] == "nothing":
            raise ValueError(f"{os.getcwd()} does not contain any train file")

        else:
            # print(f"{train_files}")
            dataset_type = "classification" if run_chemprop.check_column(os.path.join(data_path, train_files),
                                                                         1) else "regression"

        parameter_name = f"{save_name}_{fingerprint}_{epochs}_{head}_{message}"
        save_dir = os.path.join(data_path, parameter_name)

        log_file = os.path.join(save_dir, 'predict.log')
        data_path_train = os.path.join(data_path, train_files)
        data_path_test = os.path.join(data_path, test_files)
        data_path_val = os.path.join(data_path, val_files)
        #     if predict:
        #         await ModelTrainer.run_training_script(data_path_train, dataset_type, fingerprint, save_dir, epochs,
        #                                                head, message, data_path_val, data_path_test, cpu_cores)
        #         await ModelPredictor.run_predicting_script(data_path_train, dataset_type, fingerprint,
        #                                                    f"{save_dir}.csv", head,
        #                                                    f"{save_dir}/fold_0/model_0/model.pt")
        #         await ModelEvaluator.calculate(f"{save_dir}.csv", data_path, dataset_type)
        #     else:
        logging.info("using traindataset_======")
        await ModelTrainer.run_training_script(data_path_train, dataset_type, fingerprint, save_dir, epochs,
                                               head, message, data_path_val, data_path_test, cpu_cores)

        logging.info("Program finished.")

    @staticmethod
    def find_train_test_val_files(directory):
        logging.info(f"find_tain_test_val:{directory}========")
        train_files = run_chemprop.find_files_with_keyword(directory, 'train')
        test_files = run_chemprop.find_files_with_keyword(directory, 'test')
        val_files = run_chemprop.find_files_with_keyword(directory, 'val')

        return train_files, test_files, val_files

    # Find files with a specific keyword in their name
    @staticmethod
    def find_files_with_keyword(directory, keyword):
        found = False
        matching_file = None
        for filename in os.listdir(directory):
            if filename.lower().find(keyword.lower()) != -1 and filename.lower().endswith('.csv'):
                if found:
                    raise ModelEvaluator.MultipleFilesFoundError(f"Multiple '{keyword}' files found in directory: {directory}")
                matching_file = filename
                found = True

        if not found:
            matching_file = "nothing"

        return matching_file

    # Check if a specific column in a CSV file contains binary data
    @staticmethod
    def check_column(file_path, column_index):
        with open(file_path, 'r') as file:
            next(file)  # Skip the header
            for line in file:
                columns = line.strip().split(',')
                if len(columns) > column_index:
                    value = columns[column_index].strip()
                    if value not in ("0", "1", "0.0", "1.0"):
                        return False
                else:
                    return False
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some data.')
    parser.add_argument('--directory', type=str,
                        default="./testdataset",
                        help='Directory containing subdirectories')
    parser.add_argument('--predict', action='store_true', help='predict datatype')
    parser.add_argument('--log_file', type=str, default='./execution.log', help='Log file path')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--head', type=str, default='FFN', help='head type')
    parser.add_argument('--message', type=str, default='multi_single_input', help='Message type')
    parser.add_argument('--save_name', type=str, default='sinpt', help='saved_model')
    parser.add_argument('--fingerprint', type=str, default='mol', help='fingerprint')
    parser.add_argument('--cpu_cores', type=str, default=None, help='CPU cores to use')

    args = parser.parse_args()

    asyncio.run(run_chemprop().run(args, args.directory, args.predict, args.log_file, args.epochs, args.head, args.message, args.save_name, args.fingerprint, args.cpu_cores))