import os
import torch
import pandas as pd
import logging
from torch.utils.data import Dataset, DataLoader
from model import NeuralNetwork  # Ensure the import path is correct
from data_loader import MoleculeDataset  # Ensure the import path is correct


def configure_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_model(model_path, input_dim, hidden_dim1, hidden_dim2, output_dim):
    logging.info("Loading model...")
    model = NeuralNetwork(input_dim, hidden_dim1, hidden_dim2, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    logging.info("Model loaded.")
    return model


def predict(model, data_loader, use_gpu):
    predictions = []
    label_list = []
    smiles_list = []

    with torch.no_grad():
        for data in data_loader:
            features, label, smiles = data
            if use_gpu:
                features = features.cuda()
            output = model(features)
            predictions.extend(output.cpu().numpy().flatten())
            label_list.extend(label.numpy().flatten())
            smiles_list.extend(smiles)

    return predictions, label_list, smiles_list


def save_predictions(smiles, labels, predictions, output_path):
    logging.info("Saving predictions...")
    results_df = pd.DataFrame({'smiles': smiles, 'true_label': labels, 'pre_label': predictions})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure the save path exists
    results_df.to_csv(output_path, index=False)
    logging.info(f"Predictions saved to '{output_path}'")


def main(config):
    configure_logging()

    logging.info(f"Using GPU: {config['use_gpu']}")
    logging.info("Loading data...")
    inference_dataset = MoleculeDataset(config['data_path'])
    inference_loader = DataLoader(inference_dataset, batch_size=config['batch_size'], shuffle=False)
    logging.info("Data loaded.")

    input_dim = inference_dataset.features.shape[1]  # Adjust according to your dataset
    hidden_dim1 = config['hidden_dim1']
    hidden_dim2 = config['hidden_dim2']
    output_dim = config['output_dim']

    model = load_model(config['model_path'], input_dim, hidden_dim1, hidden_dim2, output_dim)
    if config['use_gpu']:
        model = model.cuda()

    logging.info("Starting prediction...")
    predictions, label_list, smiles_list = predict(model, inference_loader, config['use_gpu'])
    logging.info("Prediction completed.")

    save_predictions(smiles_list, label_list, predictions, config['output_path'])


if __name__ == "__main__":
    config = {
        'batch_size': 64,
        'use_gpu': torch.cuda.is_available(),
        'data_path': 'dataset_pc9/pc9.csv',
        'model_path': './saved_model_pc9/best_neural_network.pth',
        'output_path': './dataset_pc9/pc9/predicted_pc9.csv',
        'hidden_dim1': 300,  # Adjust according to your model
        'hidden_dim2': 100,  # Adjust according to your model
        'output_dim': 1,  # Adjust according to your model
    }
    main(config)
