import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from data_loader import MoleculeDataset
from model import NeuralNetwork


def evaluate(loader, model, criterion, use_gpu):
    model.eval()
    preds = []
    targets = []
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            features, target, smiles = data
            if use_gpu:
                features, target = features.cuda(), target.cuda()
            output = model(features)
            loss = criterion(output.squeeze(1), target)
            total_loss += loss.item()
            preds.extend(output.cpu().numpy())
            targets.extend(target.cpu().numpy())
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    return total_loss / len(loader), rmse, mae, r2


def main(config):
    use_gpu = config['use_gpu']

    train_dataset = MoleculeDataset(config['train_data_path'])
    test_dataset = MoleculeDataset(config['test_data_path'])
    val_dataset = MoleculeDataset(config['val_data_path'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    print(f"Number of parameters: {train_dataset.features.shape[1]}   train_dataset.features.shape[1]{train_dataset.features.shape[1]}\n")
    model = NeuralNetwork(train_dataset.features.shape[1], config['output_dim'])
    if use_gpu:
        model = model.cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])

    best_val_loss = float('inf')
    best_model_path = config['model_path']
    latest_model_path = os.path.join(config['save_dir'], 'latest_neural_network.pth')

    # check model path
    if not os.path.exists(config['save_dir']):
        os.makedirs(config['save_dir'])

    for epoch in range(config['num_epochs']):
        print('*' * 10)
        print(f'epoch {epoch + 1}')
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 1):
            features, target, smiles = data
            if use_gpu:
                features, target = features.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output.squeeze(1), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            if i % 300 == 0:
                print(f'[{epoch + 1}/{config["num_epochs"]}] Loss: {running_loss / i:.6f}')
        print(f'Finish {epoch + 1} epoch, Loss: {running_loss / i:.6f}')

        train_loss, train_rmse, train_mae, train_r2 = evaluate(train_loader, model, criterion, use_gpu)
        val_loss, val_rmse, val_mae, val_r2 = evaluate(val_loader, model, criterion, use_gpu)
        test_loss, test_rmse, test_mae, test_r2 = evaluate(test_loader, model, criterion, use_gpu)

        print(f'Train Loss: {train_loss:.6f}, RMSE: {train_rmse:.6f}, MAE: {train_mae:.6f}, R2: {train_r2:.6f}')
        print(f'Validation Loss: {val_loss:.6f}, RMSE: {val_rmse:.6f}, MAE: {val_mae:.6f}, R2: {val_r2:.6f}')
        print(f'Test Loss: {test_loss:.6f}, RMSE: {test_rmse:.6f}, MAE: {test_mae:.6f}, R2: {test_r2:.6f}')

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f'Best model saved with validation loss: {best_val_loss:.6f}')

    # save model
    torch.save(model.state_dict(), latest_model_path)


if __name__ == "__main__":
    config = {
        'batch_size': 64,
        'learning_rate': 1e-3,
        'num_epochs': 10,
        'use_gpu': torch.cuda.is_available(),
        'train_data_path': 'dataset_qmugs25/qmugs25_train.csv',
        'test_data_path': 'dataset_qmugs25/qmugs25_test.csv',
        'val_data_path': 'dataset_qmugs25/qmugs25_val.csv',
        'save_dir': './saved_model_qmugs25',
        'model_path': './saved_model_qmugs25/best_neural_network.pth',
        'output_dim': 1,
    }
    main(config)
