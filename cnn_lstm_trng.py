# cnn_lstm_training.py

# Imports
import os
import pandas as pd
from glob import glob
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

# Set multiprocessing start method (for macOS)
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

# Device Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset Class
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return sequence, target
    
# CNN-LSTM Model
class CNNLSTM(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, kernel_size=3, learning_rate=1e-3):
        super(CNNLSTM, self).__init__()

        self.save_hyperparameters()

        # CNN layer (1D convolution)
        self.cnn = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)

        # LSTM layers
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Learning rate
        self.learning_rate = learning_rate

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_dim)

        # CNN expects (batch_size, input_dim, sequence_length), so we permute
        x = x.permute(0, 2, 1)
        x = self.cnn(x)

        # LSTM expects (batch_size, sequence_length, hidden_dim), so we permute back
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)

        # Apply fully connected layer to the last timestep
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        sequences, targets = batch
        predictions = self(sequences)
        loss = nn.MSELoss()(predictions, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        sequences, targets = batch
        predictions = self(sequences)
        loss = nn.MSELoss()(predictions, targets)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        sequences, targets = batch
        predictions = self(sequences)
        loss = nn.MSELoss()(predictions, targets)
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]

# Other Functions
# Compute global mean and std across all files
def compute_global_stats(file_paths, channels):
    all_data = []
    for file in file_paths:
        df = pd.read_csv(file)
        print(f"Processing file: {os.path.basename(file)}")
        print(f"Columns in file: {df.columns.tolist()}")
        # Check if the required columns exist
        if not all(col in df.columns for col in channels):
            print(f"Skipping file {os.path.basename(file)}: Missing required columns.")
            continue
        all_data.append(df[channels])
    if not all_data:
        raise ValueError("No valid data found in the files for the specified channels.")
    combined_data = pd.concat(all_data)
    global_mean = combined_data.mean()
    global_std = combined_data.std()
    return global_mean, global_std

# Normalize each file and prepare sequences
# def prepare_sequences(file_paths, global_mean, global_std, T, channels):
#     dataset_sequences = {}
#     for file in file_paths:
#         df = pd.read_csv(file)
#         # Normalize
#         df[channels] = (df[channels] - global_mean) / global_std
#         # Create sequences
#         sequences, targets = [], []
#         for i in range(len(df) - T):
#             seq = df.iloc[i:i + T][channels].values
#             target = df.iloc[i:i + T][channels].values
#             sequences.append(seq)
#             targets.append(target)
#         # Store sequences for this file
#         dataset_sequences[os.path.basename(file)] = {
#             "sequences": sequences,
#             "targets": targets
#         }
#     return dataset_sequences


# Normalize each file and prepare sequences
def process_file(file_path, global_mean, global_std, T, channels):
    df = pd.read_csv(file_path)
    df[channels] = (df[channels] - global_mean) / global_std
    sequences, targets = [], []
    for i in range(len(df) - T):
        seq = df.iloc[i:i + T][channels].values
        target = df.iloc[i:i + T][channels].values
        sequences.append(seq)
        targets.append(target)
    return os.path.basename(file_path), {"sequences": sequences, "targets": targets}

def prepare_sequences_parallel(file_paths, global_mean, global_std, T, channels):
    with Pool() as pool:  # Use all available cores
        results = pool.starmap(
            process_file,
            [(file_path, global_mean, global_std, T, channels) for file_path in file_paths]
        )
    return dict(results)

# Evaluate on test data
def evaluate_model(model, test_loader):
    all_targets = []
    all_predictions = []

    model.eval()
    with torch.no_grad():
        for sequences, targets in test_loader:
            predictions = model(sequences)
            all_targets.append(targets.numpy())
            all_predictions.append(predictions.numpy())

    # Convert to numpy arrays
    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)

    # Compute metrics
    mse = mean_squared_error(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)

    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R^2: {r2:.4f}")

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(all_targets[:100], label="Actual")
    plt.plot(all_predictions[:100], label="Predicted")
    plt.title("Time-Series Forecast (Sample)")
    plt.legend()
    plt.show()

    # Residual plot
    residuals = all_targets - all_predictions
    plt.figure(figsize=(10, 5))
    plt.plot(residuals, label="Residuals")
    plt.axhline(0, color="r", linestyle="--")
    plt.title("Residual Plot")
    plt.legend()
    plt.show()

# Perform Leave-One-Out Validation
def leave_one_out_validation(dataset_sequences):
    results = []

    for test_file in dataset_sequences.keys():
        print(f"\nLeave-One-Out: Testing on {test_file}")

        train_sequences = []
        train_targets = []
        for file, data in dataset_sequences.items():
            if file != test_file:
                train_sequences.extend(data["sequences"])
                train_targets.extend(data["targets"])

        test_data = dataset_sequences[test_file]
        test_sequences = test_data["sequences"]
        test_targets = test_data["targets"]

        train_dataset = TimeSeriesDataset(train_sequences, train_targets)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=5, persistent_workers=True)

        test_dataset = TimeSeriesDataset(test_sequences, test_targets)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=5, persistent_workers=True)

        model = CNNLSTM(
            input_dim=3,
            hidden_dim=64,
            num_layers=2,
            output_dim=3,
            kernel_size=3,
            learning_rate=1e-3
        )

        early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=True)
        checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", filename="best_model", verbose=True)

        trainer = Trainer(
            max_epochs=20,
            callbacks=[early_stopping, checkpoint],
            log_every_n_steps=10,
            enable_progress_bar=True  # Ensure progress bar is shown
        )

        print(f"Starting training for {test_file}...")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
        print(f"Training for {test_file} completed.")

        # Save the trained model checkpoint
        checkpoint_path = f"model_{test_file}.ckpt"
        trainer.save_checkpoint(checkpoint_path)
        print(f"Model checkpoint saved for {test_file} at {checkpoint_path}.")
        
        # **Use evaluate_model() during this iteration**
        print(f"Evaluation for {test_file}:")
        evaluate_model(model, test_loader)  # Visualization for this iteration

        # Collect and print results
        print(f"Evaluating {test_file}...")
        all_targets = []
        all_predictions = []
        model.eval()
        with torch.no_grad():
            for sequences, targets in test_loader:
                predictions = model(sequences)
                all_targets.append(targets.numpy())
                all_predictions.append(predictions.numpy())

        all_targets = np.concatenate(all_targets, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)

        mse = mean_squared_error(all_targets, all_predictions)
        mae = mean_absolute_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)

        print(f"Results for {test_file}: MSE={mse:.4f}, MAE={mae:.4f}, R^2={r2:.4f}")
        results.append({"file": test_file, "mse": mse, "mae": mae, "r2": r2})

    return results


# Main Function (To Ensure Script Is Executable)
if __name__ == "__main__":
    print("Define Data Paths and Parameters")
    data_dir = "/Users/vinny/-hd-net/data-csv"  # Replace with your directory
    file_paths = glob(os.path.join(data_dir, "*.csv"))
    channels = ["ch.1", "ch.2", "ch.3"]
    T = 50

    print("Preprocessing-1-compute_global_stats")
    global_mean, global_std = compute_global_stats(file_paths, channels)
    print("Preprocessing-2-prepare_sequences")
    dataset_sequences = prepare_sequences_parallel(file_paths, global_mean, global_std, T, channels)
    for file_name, data in dataset_sequences.items():
        print(f"File: {file_name}, Number of Sequences: {len(data['sequences'])}")

    print("Run Leave-One-Out Validation")
    try:
        results = leave_one_out_validation(dataset_sequences)
    except Exception as e:
        print(f"Error occurred: {e}")


    print("Calculating average results across all files")
    avg_mse = np.mean([r["mse"] for r in results])
    avg_mae = np.mean([r["mae"] for r in results])
    avg_r2 = np.mean([r["r2"] for r in results])
    print(f"Average Results Across All Files: MSE={avg_mse:.4f}, MAE={avg_mae:.4f}, R^2={avg_r2:.4f}")

    print("Post-Validation: Detailed evaluation for specific test datasets")
    for test_file in dataset_sequences.keys():
        print(f"\nPost-Validation Evaluation for {test_file}:")
        test_data = dataset_sequences[test_file]
        test_sequences = test_data["sequences"]
        test_targets = test_data["targets"]

        # Create DataLoader for the test data
        test_dataset = TimeSeriesDataset(test_sequences, test_targets)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Load the trained model checkpoint
        checkpoint_path = f"model_{test_file}.ckpt"
        trained_model = CNNLSTM.load_from_checkpoint(checkpoint_path)

        # Perform post-validation evaluation
        evaluate_model(trained_model, test_loader)