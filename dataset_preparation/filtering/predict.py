import logging
from rdkit import Chem
from collections import defaultdict
import argparse
import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

hatree2kcal = 627.509


class MoleculeProcessor:
    def __init__(self):
        self.elements = ["C", "H", "O", "Br", "P", "I", "F", "N", "S", "Cl", "B"]

    def count_elements_in_smiles(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            element_counts = defaultdict(int)
            unrecognized_elements = []
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                if symbol in self.elements:
                    element_counts[symbol] += 1
                else:
                    unrecognized_elements.append(symbol)
            if unrecognized_elements:
                logger.warning(f"Unrecognized elements {set(unrecognized_elements)} in SMILES {smiles}")
                raise ValueError(f"Unrecognized element '{set(unrecognized_elements)}' in SMILES {smiles}")
            return {element: element_counts[element] for element in self.elements}

        except Exception as e:
            logger.error(f"Error processing SMILES {smiles}: {e}")
            return None

    def calculate_element(self, smiles, energy):
        element_count = self.count_elements_in_smiles(smiles)
        if not element_count:
            return None
        return {
            'smiles': smiles,
            **element_count,
            'energy': energy,
        }

    def process_csv(self, filename):
        molecules = []
        df = pd.read_csv(filename)
        for index, row in df.iterrows():
            smiles, energy = row[0], float(row[1])
            try:
                molecule_data = self.calculate_element(smiles, energy)
                if molecule_data:
                    molecules.append(molecule_data)
            except ValueError:
                logger.warning(f"Skipping invalid SMILES {smiles}")
        return molecules

    def save_to_csv(self, data, filename):
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logger.info(f"Data saved to {filename}")

    def split_dataset(self, data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        return {'data': data}


def load_data(file_path):
    data = pd.read_csv(file_path)
    features = data.iloc[:, 1:-1].values
    target = data.iloc[:, -1].values
    return features, target, data


def evaluate(features, target, model):
    preds = model.predict(features)
    rmse = np.sqrt(mean_squared_error(target, preds))
    mae = mean_absolute_error(target, preds)
    r2 = r2_score(target, preds)
    return rmse, mae, r2, preds


def main(args):
    processor = MoleculeProcessor()
    molecules = processor.process_csv(args.file_path)

    # Save processed data to a temporary file
    temp_filename = args.file_path.replace('.csv', '_processed.csv')
    processor.save_to_csv(molecules, temp_filename)

    # Load processed data for prediction
    predict_features, predict_target, predict_data = load_data(temp_filename)

    # Load best model
    with open(args.model_path, 'rb') as f:
        best_model = pickle.load(f)

    # Predict on processed data
    rmse, mae, r2, preds = evaluate(predict_features, predict_target, best_model)

    # Save predictions with specified columns
    predictions_df = predict_data.iloc[:, [0, -1]].copy()
    predictions_df['Predictions'] = preds
    predictions_path = os.path.join(args.save_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)

    print(f'Predictions saved to {predictions_path}')
    print(f'Predict RMSE: {rmse:.6f}, MAE: {mae:.6f}, R2: {r2:.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', default="./pc9_val.csv", help="Path to the input CSV file.")
    parser.add_argument('--model_path', default="./saved_model_pc9/best_linear_model.pkl",
                        help="Path to the saved model file.")
    parser.add_argument('--save_dir', default="./saved_model_pc9/", help="Directory to save predictions.")
    args = parser.parse_args()
    main(args)
