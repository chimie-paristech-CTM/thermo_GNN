#!/usr/bin/env python

from pathlib import Path
import pandas as pd
from lightning import pytorch as pl

from chemprop import data, featurizers, models, nn

OPTIMAL_CONFIG = {
    "dropout": 0.0,
    "message_hidden_dim": 400,
    "depth": 3,
    "ffn_hidden_dim": 500,
    "ffn_num_layers": 2
}

chemprop_dir = Path.cwd()
train_path = chemprop_dir / "dataset" / "qm9" / "qm9_train.csv"
val_path = chemprop_dir / "dataset" / "qm9" / "qm9_val.csv"
test_path = chemprop_dir / "dataset" / "qm9" / "qm9_test.csv"
results_dir = chemprop_dir / "results_qm9_baseline_simplified"
results_dir.mkdir(exist_ok=True, parents=True)

EPOCHS = 1
BATCH_SIZE = 256

print("=" * 70)
print("QM9 Training")
print("=" * 70)
print(f"Optimal Hyperparameters:")
print(f"  dropout: {OPTIMAL_CONFIG['dropout']}")
print(f"  message_hidden_dim: {OPTIMAL_CONFIG['message_hidden_dim']}")
print(f"  depth: {OPTIMAL_CONFIG['depth']}")
print(f"  ffn_hidden_dim: {OPTIMAL_CONFIG['ffn_hidden_dim']}")
print(f"  ffn_num_layers: {OPTIMAL_CONFIG['ffn_num_layers']}")
print(f"  epochs: {EPOCHS}")
print(f"  batch_size: {BATCH_SIZE}")
print("=" * 70)

print("\nLoading data...")
train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

smis_train = train_df["smiles"].values
ys_train = train_df[["formation_energy"]].values
smis_val = val_df["smiles"].values
ys_val = val_df[["formation_energy"]].values
smis_test = test_df["smiles"].values
ys_test = test_df[["formation_energy"]].values

print(f"\nDataset sizes:")
print(f"  Train: {len(smis_train)} samples")
print(f"  Val:   {len(smis_val)} samples")
print(f"  Test:  {len(smis_test)} samples")


train_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis_train, ys_train)]
val_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis_val, ys_val)]
test_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis_test, ys_test)]


atom_featurizer = featurizers.ThermoGNNAtomFeaturizer()
bond_featurizer = featurizers.ThermoGNNBondFeaturizer()
featurizer = featurizers.ThermoGNNMolGraphFeaturizer(
    atom_featurizer=atom_featurizer,
    bond_featurizer=bond_featurizer
)



train_dset = data.MoleculeDataset(train_data, featurizer)
val_dset = data.MoleculeDataset(val_data, featurizer)
test_dset = data.MoleculeDataset(test_data, featurizer)


scaler = train_dset.normalize_targets()
val_dset.normalize_targets(scaler)

train_loader = data.build_dataloader(train_dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = data.build_dataloader(val_dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = data.build_dataloader(test_dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


mp = nn.BondMessagePassing(
    d_v=featurizer.atom_fdim,
    d_e=featurizer.bond_fdim,
    d_h=OPTIMAL_CONFIG["message_hidden_dim"],
    depth=OPTIMAL_CONFIG["depth"],
    dropout=OPTIMAL_CONFIG["dropout"]
)
agg = nn.NormAggregation()
output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
ffn = nn.RegressionFFN(
    input_dim=OPTIMAL_CONFIG["message_hidden_dim"],
    n_tasks=1,
    hidden_dim=OPTIMAL_CONFIG["ffn_hidden_dim"],
    n_layers=OPTIMAL_CONFIG["ffn_num_layers"],
    dropout=OPTIMAL_CONFIG["dropout"],
    output_transform=output_transform
)
metric_list = [nn.metrics.RMSE(), nn.metrics.MAE(), nn.metrics.R2Score()]
mpnn = models.MPNN(mp, agg, ffn, batch_norm=False, metrics=metric_list)

print(f"\nStarting training for {EPOCHS} epochs...")
trainer = pl.Trainer(
    logger=False,
    enable_checkpointing=True,
    enable_progress_bar=True,
    accelerator="auto",
    devices=1,
    max_epochs=EPOCHS,
)

trainer.fit(mpnn, train_loader, val_loader)

print("\n" + "=" * 70)
print("Evaluating on test set...")
print("=" * 70)
test_results = trainer.test(mpnn, test_loader)

print("\n" + "=" * 70)
print("Training Complete!")
print("=" * 70)
print("\nTest Results:")
for key, value in test_results[0].items():
    print(f"  {key}: {value:.6f}")

model_path = results_dir / "qm9.pt"
print(f"\nSaving model to {model_path}")
models.save_model(model_path, mpnn)

test_results_df = pd.DataFrame([test_results[0]])
test_results_df.to_csv(results_dir / "test_results.csv", index=False)

print("Done!")
