import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import numpy as np
import os

class WaterPressureDataset(Dataset):
    def __init__(self, file_paths, k_values, dl_values, window_size=5000, stride=1000, scaler=None):
        """
        Args:
            file_paths (list): List of paths to CSV files
            k_values (list): List of K values corresponding to each file
            dl_values (list): List of DL values corresponding to each file
            window_size (int): Size of sliding window
            stride (int): Stride for sliding window
            scaler (StandardScaler): Scaler for normalization
        """
        self.window_size = window_size
        self.stride = stride
        
        self.windows = []
        self.k_values = []
        self.dl_values = []
        
        # Process each file
        for file_path, k, dl in zip(file_paths, k_values, dl_values):
            # Read CSV
            df = pd.read_csv(file_path)
            data = df[['ch.1', 'ch.2', 'ch.3']].values
            
            # Fit or transform with scaler
            if scaler is None:
                scaler = StandardScaler()
                data = scaler.fit_transform(data)
            else:
                data = scaler.transform(data)
            
            # Create windows
            for start in range(0, len(data) - window_size + 1, stride):
                window = data[start:start + window_size]
                self.windows.append(torch.FloatTensor(window))
                self.k_values.append(k)
                self.dl_values.append(dl)
        
        self.scaler = scaler

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return {
            'pressure_sequence': self.windows[idx],
            'k': torch.FloatTensor([self.k_values[idx]]),
            'dl': torch.FloatTensor([self.dl_values[idx]])
        }

class WaterPressureDataModule(pl.LightningDataModule):
    def __init__(
        self,
        file_paths,
        k_values,
        dl_values,
        test_idx,
        window_size=5000,
        stride=1000,
        batch_size=32,
        num_workers=4
    ):
        super().__init__()
        self.file_paths = file_paths
        self.k_values = k_values
        self.dl_values = dl_values
        self.test_idx = test_idx
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.scaler = None

    def setup(self, stage=None):
        # Split data for leave-one-out validation
        train_files = [f for i, f in enumerate(self.file_paths) if i != self.test_idx]
        train_k = [k for i, k in enumerate(self.k_values) if i != self.test_idx]
        train_dl = [dl for i, dl in enumerate(self.dl_values) if i != self.test_idx]
        
        if stage == 'fit' or stage is None:
            # Create training dataset using all 7 files
            self.train_dataset = WaterPressureDataset(
                train_files,
                train_k,
                train_dl,
                self.window_size,
                self.stride
            )
            self.scaler = self.train_dataset.scaler
            
            # Use the left-out file as validation dataset
            self.val_dataset = WaterPressureDataset(
                [self.file_paths[self.test_idx]],
                [self.k_values[self.test_idx]],
                [self.dl_values[self.test_idx]],
                self.window_size,
                self.stride,
                self.scaler
            )

        if stage == 'test' or stage is None:
            # Test dataset is the same as validation dataset in LOOCV
            self.test_dataset = self.val_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


class WaterPressurePredictor(pl.LightningModule):
    def __init__(
        self, 
        hidden_dim=64, 
        window_size=5000,
        spike_threshold=5.0,
        diff_threshold=10.0,
        learning_rate=0.001
    ):
        super().__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.spike_threshold = spike_threshold
        self.diff_threshold = diff_threshold
        self.learning_rate = learning_rate
        
        # Parameter embedding (K and DL)
        self.parameter_embedding = nn.Linear(2, hidden_dim)
        
        # CNN layers for local pattern extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        
        # Self-attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )
        
        # LSTM with attention
        self.lstm = nn.LSTM(
            input_size=64 + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        
        # Output layers
        self.output_layer = nn.Linear(hidden_dim * 2, 3)  # Pressure values for 3 channels
        self.failure_proximity_layer = nn.Linear(hidden_dim * 2, 1)  # Failure proximity
        
        # Loss functions
        self.pressure_criterion = nn.MSELoss()
        self.proximity_criterion = nn.BCEWithLogitsLoss()

    def forward(self, k, dl, sequence):
        """
        Args:
            k (torch.Tensor): Filter coefficient values [batch_size, 1]
            dl (torch.Tensor): Water level values [batch_size, 1]
            sequence (torch.Tensor): Input pressure sequence [batch_size, seq_len, 3]
        """
        batch_size = k.size(0)
        
        # Parameter embedding
        params = torch.cat([k, dl], dim=1)
        param_embedding = self.parameter_embedding(params)  # [batch_size, hidden_dim]
        
        # CNN feature extraction
        cnn_out = self.cnn(sequence.transpose(1, 2))  # [batch_size, 64, seq_len]
        cnn_out = cnn_out.transpose(1, 2)  # [batch_size, seq_len, 64]
        
        # Self-attention
        attention_out, _ = self.attention(cnn_out, cnn_out, cnn_out)
        
        # Combine with parameter embedding
        param_expanded = param_embedding.unsqueeze(1).expand(-1, attention_out.size(1), -1)
        combined = torch.cat([attention_out, param_expanded], dim=2)
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(combined)
        
        # Generate predictions
        pressure_pred = self.output_layer(lstm_out)
        failure_proximity = self.failure_proximity_layer(lstm_out)
        
        return pressure_pred, failure_proximity

    def detect_failure_by_pattern(self, pressure_values):
        """Rule-based failure detection based on pressure patterns"""
        with torch.no_grad():
            # Detect sudden changes in Channel 3
            ch3_changes = torch.diff(pressure_values[:, :, 2], dim=1)
            sudden_spike = torch.any(ch3_changes > self.spike_threshold)
            
            # Check pressure differentials between channels
            ch1_ch3_diff = torch.abs(pressure_values[:, :, 0] - pressure_values[:, :, 2])
            ch2_ch3_diff = torch.abs(pressure_values[:, :, 1] - pressure_values[:, :, 2])
            pressure_divergence = torch.any(torch.max(ch1_ch3_diff, ch2_ch3_diff) > self.diff_threshold)
            
            return sudden_spike or pressure_divergence

    def generate_sequence(self, k, dl, max_length=300000):
        """Generate full sequence until failure is detected"""
        self.eval()
        with torch.no_grad():
            outputs = []
            current_window = torch.zeros(1, self.window_size, 3).to(self.device)
            
            while len(outputs) < max_length:
                # Generate next predictions
                pressure_pred, failure_prox = self(k, dl, current_window)
                
                # Get the last predicted values
                next_values = pressure_pred[:, -1, :]
                
                # Check for failure
                if self.detect_failure_by_pattern(pressure_pred):
                    break
                
                # Store prediction and update window
                outputs.append(next_values)
                current_window = torch.cat([
                    current_window[:, 1:, :],
                    next_values.unsqueeze(1)
                ], dim=1)
            
            return torch.cat(outputs, dim=0)

    def training_step(self, batch, batch_idx):
        """Training step"""
        k, dl, sequence = batch['k'], batch['dl'], batch['pressure_sequence']
        
        # Generate predictions
        pressure_pred, failure_prox = self(k, dl, sequence)
        
        # Calculate pressure prediction loss
        pressure_loss = self.pressure_criterion(pressure_pred, sequence)
        
        # Calculate failure proximity loss (assuming last 10% of sequence is near failure)
        seq_len = sequence.size(1)
        failure_target = torch.zeros_like(failure_prox)
        failure_target[:, int(0.9 * seq_len):] = 1
        proximity_loss = self.proximity_criterion(failure_prox.squeeze(), failure_target.squeeze())
        
        # Combined loss
        total_loss = pressure_loss + 0.1 * proximity_loss
        
        # Log losses
        self.log('train_pressure_loss', pressure_loss)
        self.log('train_proximity_loss', proximity_loss)
        self.log('train_total_loss', total_loss)
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        return total_loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_pressure_loss"
            }
        }
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step - used during training to monitor model's performance
        on the left-out file and prevent overfitting
        """
        k, dl, sequence = batch['k'], batch['dl'], batch['pressure_sequence']
        pressure_pred, failure_prox = self(k, dl, sequence)
        
        # Calculate basic loss for monitoring training
        val_loss = self.pressure_criterion(pressure_pred, sequence)
        
        # Log validation loss
        self.log('val_loss', val_loss, prog_bar=True)
        
        return val_loss

    def test_step(self, batch, batch_idx):
        """
        Test step - used after training to evaluate model's final performance 
        on the left-out file with comprehensive metrics
        """
        k, dl, sequence = batch['k'], batch['dl'], batch['pressure_sequence']
        pressure_pred, failure_prox = self(k, dl, sequence)
        
        # Calculate various metrics
        mse_loss = F.mse_loss(pressure_pred, sequence)
        mae_loss = F.l1_loss(pressure_pred, sequence)
        
        # Calculate R² score for each channel
        r2_scores = []
        for i in range(3):  # 3 channels
            y_true = sequence[..., i]
            y_pred = pressure_pred[..., i]
            var_true = torch.var(y_true)
            mse = torch.mean((y_true - y_pred) ** 2)
            r2 = 1 - (mse / var_true)
            r2_scores.append(r2)
        
        # Log detailed metrics
        self.log('test_mse', mse_loss)
        self.log('test_mae', mae_loss)
        for i, r2 in enumerate(r2_scores):
            self.log(f'test_r2_channel_{i+1}', r2)
        
        # Return predictions and metrics for later analysis
        return {
            'test_mse': mse_loss,
            'test_mae': mae_loss,
            'r2_scores': r2_scores,
            'predictions': pressure_pred.detach(),
            'targets': sequence.detach(),
            'k_value': k.detach(),
            'dl_value': dl.detach(),
            'failure_proximity': failure_prox.detach()
        }

    def test_epoch_end(self, outputs):
        """
        Aggregate test results from the epoch
        """
        # Combine results from all batches
        all_mse = torch.stack([x['test_mse'] for x in outputs]).mean()
        all_mae = torch.stack([x['test_mae'] for x in outputs]).mean()
        all_r2 = torch.stack([torch.tensor(x['r2_scores']) for x in outputs]).mean(0)
        
        # Log aggregated metrics
        self.log('test_avg_mse', all_mse)
        self.log('test_avg_mae', all_mae)
        for i, r2 in enumerate(all_r2):
            self.log(f'test_avg_r2_channel_{i+1}', r2)
        
        return {
            'avg_mse': all_mse,
            'avg_mae': all_mae,
            'avg_r2_scores': all_r2,
            'k_value': outputs[0]['k_value'],  # Assuming same k,dl for all batches
            'dl_value': outputs[0]['dl_value']
        }

    def visualize_predictions(self, pred_sequence, true_sequence, k, dl, save_path=None):
        """
        Visualize predictions against true values for all channels
        
        Args:
            pred_sequence (torch.Tensor): Predicted pressure values [seq_len, 3]
            true_sequence (torch.Tensor): True pressure values [seq_len, 3]
            k (float): Filter coefficient value
            dl (float): Water level value
            save_path (str, optional): Path to save the plot
        """
        # Convert to numpy for plotting
        pred = pred_sequence.cpu().numpy()
        true = true_sequence.cpu().numpy()
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'Pressure Predictions (K={k:.2f}, DL={dl:.3f}m)', fontsize=16)
        
        channels = ['Channel 1', 'Channel 2', 'Channel 3']
        colors = ['blue', 'orange', 'green']
        
        for i, (ax, channel) in enumerate(zip(axes, channels)):
            # Plot true and predicted values
            ax.plot(true[:, i], label='True', color=colors[i], alpha=0.6)
            ax.plot(pred[:, i], label='Predicted', color=colors[i], linestyle='--')
            
            # Calculate R² score
            r2 = r2_score(true[:, i], pred[:, i])
            
            ax.set_title(f'{channel} (R² = {r2:.3f})')
            ax.set_xlabel('Time')
            ax.set_ylabel('Pressure')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_error_distribution(self, pred_sequence, true_sequence, save_path=None):
        """
        Visualize error distribution for each channel
        """
        # Calculate errors
        errors = (pred_sequence - true_sequence).cpu().numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Error Distribution by Channel', fontsize=16)
        
        for i, ax in enumerate(axes):
            sns.histplot(errors[:, i], ax=ax, kde=True)
            ax.set_title(f'Channel {i+1}')
            ax.set_xlabel('Error')
            ax.set_ylabel('Count')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_failure_proximity(self, failure_prox, true_sequence, save_path=None):
        """
        Visualize predicted failure proximity against pressure values
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot pressure values
        for i in range(3):
            ax1.plot(true_sequence[:, i], label=f'Channel {i+1}')
        ax1.set_title('Pressure Values')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Pressure')
        ax1.legend()
        ax1.grid(True)
        
        # Plot failure proximity
        ax2.plot(failure_prox.cpu().numpy(), color='red')
        ax2.set_title('Predicted Failure Proximity')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Proximity Score')
        ax2.set_ylim(0, 1)
        ax2.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_attention_weights(self, attention_weights, save_path=None):
        """
        Visualize attention weights
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_weights.cpu().numpy(), 
                cmap='viridis', 
                center=0)
        plt.title('Attention Weights Heatmap')
        plt.xlabel('Query Position')
        plt.ylabel('Key Position')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def visualize_training_history(self, trainer, save_path=None):
        """
        Plot training history
        """
        metrics = ['train_pressure_loss', 'train_proximity_loss', 'val_pressure_loss']
        
        plt.figure(figsize=(12, 6))
        for metric in metrics:
            values = trainer.callback_metrics[metric].cpu().numpy()
            plt.plot(values, label=metric)
        
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def visualize_all_results(self, test_results, save_dir=None):
        """
        Generate comprehensive visualization of test results
        """
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Plot predictions
        self.visualize_predictions(
            test_results['predictions'],
            test_results['targets'],
            test_results['k'],
            test_results['dl'],
            save_path=f'{save_dir}/predictions.png' if save_dir else None
        )
        
        # Plot error distribution
        self.plot_error_distribution(
            test_results['predictions'],
            test_results['targets'],
            save_path=f'{save_dir}/error_distribution.png' if save_dir else None
        )
        
        # Plot failure proximity
        self.plot_failure_proximity(
            test_results['failure_proximity'],
            test_results['targets'],
            save_path=f'{save_dir}/failure_proximity.png' if save_dir else None
        )
        
        # Plot attention weights if available
        if 'attention_weights' in test_results:
            self.plot_attention_weights(
                test_results['attention_weights'],
                save_path=f'{save_dir}/attention_weights.png' if save_dir else None
            )
