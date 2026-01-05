import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from torch.utils.data import Dataset

class Noise2NoiseDataset(Dataset):
    """Dataset dynamically generating random pairs from noisy realizations."""
    def __init__(self, x_noisy_list, dtype=torch.float32):
        self.x_noisy = torch.tensor(np.asarray(x_noisy_list), dtype=dtype)
        self.n_realizations, self.n_samples = self.x_noisy.shape[:2]

        if self.n_realizations < 2:
            raise ValueError("Need at least two noisy realizations for Noise2Noise.")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        idx1, idx2 = np.random.choice(self.n_realizations, 2, replace=False)
        input_sample = self.x_noisy[idx1, idx]   # (channels, length) ou (1, length)
        target_sample = self.x_noisy[idx2, idx]

        # Garantir que la dimension du canal est présente explicitement
        if input_sample.ndim == 1:
            input_sample = input_sample.unsqueeze(0)  # (1, length)
        if target_sample.ndim == 1:
            target_sample = target_sample.unsqueeze(0)

        return input_sample, target_sample



class DiamondDAE1D(nn.Module):
    def __init__(self, n_conv_blocks, filters, kernel_size, activations="relu", output_activation="linear", latent_dim=None, dtype=torch.float32, n_conv_per_block=1, n_latent_layers=1, use_batchnorm=False, dropout_rate=0.0, strides=2):
        super(DiamondDAE1D, self).__init__()

        # Store dtype for tensor conversions
        self.dtype = dtype
        
        self.n_conv_per_block = n_conv_per_block 
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate
        self.n_latent_layers = n_latent_layers

        if len(filters) != n_conv_blocks // 2:
            raise ValueError(
                f"Expected {n_conv_blocks // 2} elements in `filters`, "
                f"but got {len(filters)}. `filters` must match encoder layers."
            )

        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd and >= 1.")
        
                # Normalize strides: allow int or list[int]
        if isinstance(strides, int):
            strides = [strides] * (n_conv_blocks // 2)

        if len(strides) != n_conv_blocks // 2:
            raise ValueError(
                f"Expected {n_conv_blocks // 2} elements in `strides`, "
                f"but got {len(strides)}. `strides` must match encoder blocks."
            )

        self.n_conv_blocks = n_conv_blocks
        self.filters = filters
        self.kernel_size = kernel_size
        self.output_activation = output_activation
        self.latent_dim = latent_dim or filters[-1]
        self.strides = strides

        # Store the original string for serialization
        self._activation_string = activations
        # Dynamic activation resolution
        self.activation_fn = self._resolve_activation(activations)

        self.encoder_layers = nn.ModuleList()
        self.latent_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        # Build encoder
        in_channels = 1  # assuming single-channel input
        for block_idx, f in enumerate(filters):
            for i in range(self.n_conv_per_block):  # repeat convs per block
                self.encoder_layers.append(nn.Conv1d(in_channels, f, kernel_size, padding=self.kernel_size // 2))
                if self.use_batchnorm:
                    self.encoder_layers.append(nn.BatchNorm1d(f))
                if self.dropout_rate > 0.0:
                    self.encoder_layers.append(nn.Dropout(self.dropout_rate))
                in_channels = f  # update in_channels after each conv
            # pooling after block, with per-block stride
            self.encoder_layers.append(nn.MaxPool1d(kernel_size=self.strides[block_idx]))


        # Optional latent layer if odd number of layers
        self.use_latent = n_conv_blocks % 2 != 0              
        
        # In __init__ inside the latent layer construction
        if self.use_latent:
            for _ in range(self.n_latent_layers):
                self.latent_layers.append(nn.Conv1d(in_channels, self.latent_dim, kernel_size, padding=self.kernel_size // 2))
                if self.use_batchnorm:  # NEW
                    self.latent_layers.append(nn.BatchNorm1d(self.latent_dim))  # NEW
                if self.dropout_rate > 0.0:  # NEW
                    self.latent_layers.append(nn.Dropout(self.dropout_rate))  # NEW
                in_channels = self.latent_dim  # update channels

        decoder_strides = list(reversed(self.strides))

        # Build decoder (reverse filters)
        for block_idx, f in enumerate(reversed(filters)):
            for i in range(self.n_conv_per_block - 1):  # extra convs without stride
                self.decoder_layers.append(nn.ConvTranspose1d(
                    in_channels,
                    f,
                    kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2
                ))
                in_channels = f
            # final upsampling conv with per-block stride (reverse of encoder)
            stride = decoder_strides[block_idx]
            self.decoder_layers.append(nn.ConvTranspose1d(
                in_channels,
                f,
                kernel_size,
                stride=stride,
                padding=self.kernel_size // 2,
                output_padding=stride - 1
            ))
            in_channels = f



        # Output layer
        self.output_layer = nn.ConvTranspose1d(
            in_channels,
            1,
            kernel_size,
            padding=self.kernel_size // 2
        )

    def _resolve_activation(self, name):
        if callable(name):
            return name
        if isinstance(name, str):
            try:
                fn = getattr(F, name)
                if callable(fn):
                    return fn
            except AttributeError:
                pass
            try:
                fn = getattr(torch, name)
                if callable(fn):
                    return fn
            except AttributeError:
                pass
            raise ValueError(f"Unsupported activation name: {name}")
        raise TypeError(f"Activation must be a string or callable, got {type(name)}.")


    def forward(self, x):
        # Check input channel dimension
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(
                f"Expected input of shape (batch_size, 1, length), got {x.shape}"
            )

        # Encoder
        for layer in self.encoder_layers:
            x = layer(x)
            if isinstance(layer, nn.Conv1d):
                x = self.activation_fn(x)

        # Latent layer
        if self.use_latent:
            for layer in self.latent_layers:
                x = layer(x)
                if isinstance(layer, nn.Conv1d):
                    x = self.activation_fn(x)

        # Decoder
        for layer in self.decoder_layers:
            x = layer(x)
            if isinstance(layer, nn.ConvTranspose1d):
                x = self.activation_fn(x)

        # Output layer
        act_map = {
            "linear": lambda x: x,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "relu": F.relu
        }
        x = self.output_layer(x)
        try:
            return act_map[self.output_activation](x)
        except KeyError:
            raise ValueError(f"Unsupported output activation: {self.output_activation}")
            
    def train_classic(
        self,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        batch_size=32,
        epochs=10,
        optimizer=None,
        learning_rate=0.001,
        loss_fn=None,
        metrics=None,  # placeholder for future use
        verbose=1,
        device=None
    ):
        
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # Default optimizer and loss
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        if loss_fn is None:
            loss_fn = torch.nn.MSELoss()

        # Training data loader
        train_dataset = TensorDataset(torch.tensor(x_train, dtype=self.dtype),
                                    torch.tensor(y_train, dtype=self.dtype))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Validation data loader (if any)
        if x_val is not None and y_val is not None:
            val_dataset = TensorDataset(torch.tensor(x_val, dtype=self.dtype),
                                        torch.tensor(y_val, dtype=self.dtype))
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None

        history = {"loss": [], "val_loss": []}

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = self(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            avg_loss = epoch_loss / len(train_loader.dataset)
            history["loss"].append(avg_loss)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4e}", end="")

            # Validation
            if val_loader:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        preds = self(xb)
                        loss = loss_fn(preds, yb)
                        val_loss += loss.item() * xb.size(0)
                avg_val_loss = val_loss / len(val_loader.dataset)
                history["val_loss"].append(avg_val_loss)
                if verbose:
                    print(f" - val_loss: {avg_val_loss:.4e}")
            else:
                if verbose:
                    print()

        return history
    
    def evaluate_model(self, x_test, y_test, batch_size=32, loss_fn=None, verbose=1, device=None):
        
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()

        if loss_fn is None:
            loss_fn = torch.nn.MSELoss()

        test_dataset = TensorDataset(torch.tensor(x_test, dtype=self.dtype),
                                    torch.tensor(y_test, dtype=self.dtype))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        total_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = self(xb)
                loss = loss_fn(preds, yb)
                total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(test_loader.dataset)

        if verbose:
            print(f"Eval Loss: {avg_loss:.4e}")

        return {"loss": avg_loss}

    def predict(self, x_input, batch_size=32, device=None):

        self.eval()
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        if isinstance(x_input, np.ndarray):
            x_input = torch.tensor(x_input, dtype=self.dtype)
        x_input = x_input.to(device)

        dataset = TensorDataset(x_input)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        outputs = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(device)
                preds = self(xb)
                outputs.append(preds.cpu())

        return torch.cat(outputs, dim=0).numpy()
    

    def visualize_training(self, history):
        """
        Visualize training and validation loss from the PyTorch training history.

        Args:
            history (dict): Must contain 'loss' and optionally 'val_loss'.
        """
        if not isinstance(history, dict) or "loss" not in history:
            raise ValueError("Expected history dict with at least a 'loss' key.")

        plt.figure(figsize=(8, 5))
        plt.plot(history["loss"], label="Training Loss")

        if "val_loss" in history and history["val_loss"]:
            plt.plot(history["val_loss"], label="Validation Loss")

        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training History")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def save_model(self, filepath):
        """
        Save model weights and configuration.

        Args:
            filepath (str): Path to save the model (e.g., 'model.pth').
        """
        model_state = {
            "state_dict": self.state_dict(),
            "config": {
                "n_conv_blocks": self.n_conv_blocks,
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "activations": self._activation_string,
                "output_activation": self.output_activation,
                "latent_dim": self.latent_dim
            }
        }
        torch.save(model_state, filepath)

    @staticmethod
    def load_model(filepath):
        """
        Load a DiamondDAE1D model from a file.

        Args:
            filepath (str): Path to the saved model (.pth file)

        Returns:
            DiamondDAE1D: A restored model instance.
        """

        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        config = checkpoint["config"]

        model = DiamondDAE1D(**config)
        model.load_state_dict(checkpoint["state_dict"])
        return model
    
    def train_noise2noise(
        self,
        x_noisy_list,
        batch_size=32,
        epochs=10,
        optimizer=None,
        learning_rate=0.001,
        loss_fn=None,
        verbose=1,
        device=None,
        x_val_noisy_list=None,
        x_clean_list=None,
        x_val_clean_list=None,
    ):
        """
        Improved Noise2Noise training method with optional clean-based metrics.

        Args:
            x_noisy_list: Noisy realizations shaped like train_classic inputs:
                (n_samples, n_realizations, length) or
                (n_samples, n_realizations, channels, length).
            batch_size (int): Batch size.
            epochs (int): Number of epochs.
            optimizer: PyTorch optimizer.
            learning_rate (float): Learning rate.
            loss_fn: Loss function.
            verbose (int): Verbosity.
            device: torch.device.
            x_val_noisy_list: Validation noisy realizations (same shape as x_noisy_list).
            x_clean_list: Optional clean realizations for training (same shape as x_noisy_list).
            x_val_clean_list: Optional clean realizations for validation
                (same shape as x_val_noisy_list).

        Returns:
            dict: Training history with keys:
                - "loss": Noise2Noise training loss
                - "val_loss": Noise2Noise validation loss (or None)
                - "clean_loss": training loss vs clean targets (or None)
                - "clean_val_loss": validation loss vs clean targets (or None)
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        dtype = torch.float32

        loss_fn = loss_fn or torch.nn.MSELoss()
        optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=learning_rate)

        def _reshape_for_noise2noise(x, name):
            if x is None:
                return None
            if torch.is_tensor(x):
                x_arr = x.detach().cpu().numpy()
            else:
                x_arr = np.asarray(x)
            if x_arr.ndim < 3:
                raise ValueError(
                    f"{name} must have at least 3 dims: "
                    "(n_samples, n_realizations, length) or "
                    "(n_samples, n_realizations, channels, length)."
                )
            x_arr = np.swapaxes(x_arr, 0, 1)
            if x_arr.ndim == 3:
                x_arr = x_arr[:, :, np.newaxis, :]
            return x_arr

        x_noisy_list = _reshape_for_noise2noise(x_noisy_list, "x_noisy_list")
        x_val_noisy_list = _reshape_for_noise2noise(x_val_noisy_list, "x_val_noisy_list")
        x_clean_list = _reshape_for_noise2noise(x_clean_list, "x_clean_list")
        x_val_clean_list = _reshape_for_noise2noise(x_val_clean_list, "x_val_clean_list")

        # ---------------------------------------------------------------------
        # Main Noise2Noise training dataset (dynamic pairing, etc.)
        # ---------------------------------------------------------------------
        train_dataset = Noise2NoiseDataset(x_noisy_list, dtype=dtype)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # ---------------------------------------------------------------------
        # Optional Noise2Noise validation dataset
        # ---------------------------------------------------------------------
        val_loader = None
        if x_val_noisy_list is not None:
            val_dataset = Noise2NoiseDataset(x_val_noisy_list, dtype=dtype)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # ---------------------------------------------------------------------
        # Optional clean datasets: noisy -> clean mapping
        # We assume shapes are compatible with x_noisy_list / x_val_noisy_list.
        # ---------------------------------------------------------------------
        from torch.utils.data import Dataset

        class NoisyToCleanDataset(Dataset):
            """Dataset mapping noisy inputs to clean targets."""

            def __init__(self, x_noisy, x_clean, dtype=torch.float32):
                assert x_noisy.shape == x_clean.shape, "Noisy and clean arrays must have the same shape."
                x_noisy_t = torch.as_tensor(x_noisy, dtype=dtype)
                x_clean_t = torch.as_tensor(x_clean, dtype=dtype)
                # Flatten realizations and samples into a single batch dimension
                # Assumes input shape: (n_realizations, n_samples, channels, length)
                if x_noisy_t.ndim >= 3:
                    self.x_noisy = x_noisy_t.reshape(-1, *x_noisy_t.shape[2:])
                    self.x_clean = x_clean_t.reshape(-1, *x_clean_t.shape[2:])
                else:
                    # Fallback: just keep as is
                    self.x_noisy = x_noisy_t
                    self.x_clean = x_clean_t

            def __len__(self):
                return self.x_noisy.shape[0]

            def __getitem__(self, idx):
                return self.x_noisy[idx], self.x_clean[idx]

        train_clean_loader = None
        if x_clean_list is not None:
            train_clean_dataset = NoisyToCleanDataset(x_noisy_list, x_clean_list, dtype=dtype)
            train_clean_loader = DataLoader(train_clean_dataset, batch_size=batch_size, shuffle=False)

        val_clean_loader = None
        if x_val_clean_list is not None and x_val_noisy_list is not None:
            val_clean_dataset = NoisyToCleanDataset(x_val_noisy_list, x_val_clean_list, dtype=dtype)
            val_clean_loader = DataLoader(val_clean_dataset, batch_size=batch_size, shuffle=False)

        # ---------------------------------------------------------------------
        # History
        # ---------------------------------------------------------------------
        history = {
            "loss": [],
            "val_loss": [],
            "clean_loss": [],
            "clean_val_loss": [],
        }

        # ---------------------------------------------------------------------
        # Training loop
        # ---------------------------------------------------------------------
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0

            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = self(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            avg_loss = epoch_loss / len(train_loader.dataset)
            history["loss"].append(avg_loss)

            # We prepare a base message for printing
            if verbose:
                msg = f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4e}"
            else:
                msg = ""

            # ------------------- Noise2Noise validation -----------------------
            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        preds = self(xb)
                        loss = loss_fn(preds, yb)
                        val_loss += loss.item() * xb.size(0)
                avg_val_loss = val_loss / len(val_loader.dataset)
                history["val_loss"].append(avg_val_loss)
                if verbose:
                    msg += f" - val_loss: {avg_val_loss:.4e}"
            else:
                history["val_loss"].append(None)

            # ------------------- Clean training loss --------------------------
            if train_clean_loader is not None:
                self.eval()
                clean_train_loss = 0.0
                with torch.no_grad():
                    for xb_clean, y_clean in train_clean_loader:
                        xb_clean, y_clean = xb_clean.to(device), y_clean.to(device)
                        preds = self(xb_clean)
                        loss = loss_fn(preds, y_clean)
                        clean_train_loss += loss.item() * xb_clean.size(0)
                avg_clean_train_loss = clean_train_loss / len(train_clean_loader.dataset)
                history["clean_loss"].append(avg_clean_train_loss)
                if verbose:
                    msg += f" - clean_loss: {avg_clean_train_loss:.4e}"
            else:
                history["clean_loss"].append(None)

            # ------------------- Clean validation loss ------------------------
            if val_clean_loader is not None:
                self.eval()
                clean_val_loss = 0.0
                with torch.no_grad():
                    for xb_clean, y_clean in val_clean_loader:
                        xb_clean, y_clean = xb_clean.to(device), y_clean.to(device)
                        preds = self(xb_clean)
                        loss = loss_fn(preds, y_clean)
                        clean_val_loss += loss.item() * xb_clean.size(0)
                avg_clean_val_loss = clean_val_loss / len(val_clean_loader.dataset)
                history["clean_val_loss"].append(avg_clean_val_loss)
                if verbose:
                    msg += f" - clean_val_loss: {avg_clean_val_loss:.4e}"
            else:
                history["clean_val_loss"].append(None)

            if verbose:
                print(msg)

        return history
