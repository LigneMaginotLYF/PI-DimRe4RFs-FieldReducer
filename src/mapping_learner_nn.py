import numpy as np
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class ResBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.Tanh()

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(x)))


class PhysicsDrivenMappingNN(nn.Module):
    """
    Neural network surrogate / mapper.
    Used as:
     - Reduced surrogate S: xi'(3D) -> Y(n_x D)   [Phase 2]
     - Dimension reducer M: xi_E(5D) -> xi'(3D)    [Phase 3]
    """

    def __init__(self, input_dim, output_dim, hidden_dim=64, n_blocks=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(n_blocks)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = torch.tanh(self.input_layer(x))
        for block in self.blocks:
            h = block(h)
        return self.output_layer(h)

    def fit(self, X_train, Y_train, X_val=None, Y_val=None,
            epochs=200, lr=1e-3, batch_size=64):
        """
        Fit the NN to training data.
        All arrays are numpy; converts internally to torch.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        Y_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
        if X_val is not None:
            X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
            Y_v = torch.tensor(Y_val, dtype=torch.float32).to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=20, factor=0.5, min_lr=1e-5
        )
        criterion = nn.MSELoss()

        n = len(X_t)
        for epoch in range(epochs):
            self.train()
            perm = torch.randperm(n)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                x_b = X_t[idx]
                y_b = Y_t[idx]
                optimizer.zero_grad()
                pred = self(x_b)
                loss = criterion(pred, y_b)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            epoch_loss /= n_batches

            if X_val is not None:
                self.eval()
                with torch.no_grad():
                    val_pred = self(X_v)
                    val_loss = criterion(val_pred, Y_v).item()
                scheduler.step(val_loss)
                if (epoch + 1) % 20 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}: train={epoch_loss:.4e} val={val_loss:.4e}")
            else:
                scheduler.step(epoch_loss)
                if (epoch + 1) % 20 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}: train={epoch_loss:.4e}")

        self.to('cpu')
        return self

    def fit_with_surrogate(self, X_train, Y_train, X_val, Y_val,
                           surrogate, epochs=200, lr=1e-3, batch_size=64,
                           colloc_idx=None):
        """
        Train dimension reducer M using frozen surrogate S.
        Loss = MSE(S(M(xi_E))[:, colloc_idx], Y_reference[:, colloc_idx])

        Args:
            X_train: xi_E training data, shape (n_train, input_dim)
            Y_train: reference responses, shape (n_train, n_x)
            X_val, Y_val: validation set
            surrogate: frozen PhysicsDrivenMappingNN (S), frozen during training
            epochs, lr, batch_size: training hyperparameters
            colloc_idx: 1-D integer array of output node indices to include in
                the loss.  If None, all output nodes are used (full-profile MSE).
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        surrogate.to(device)
        surrogate.eval()
        for p in surrogate.parameters():
            p.requires_grad_(False)

        X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        Y_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
        X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
        Y_v = torch.tensor(Y_val, dtype=torch.float32).to(device)

        # Pre-build collocation index tensor (if a subset is requested)
        if colloc_idx is not None:
            cidx = torch.tensor(colloc_idx, dtype=torch.long).to(device)
        else:
            cidx = None

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=20, factor=0.5, min_lr=1e-5
        )
        criterion = nn.MSELoss()

        n = len(X_t)
        for epoch in range(epochs):
            self.train()
            perm = torch.randperm(n)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                x_b = X_t[idx]
                y_b = Y_t[idx]
                optimizer.zero_grad()
                xi_prime = self(x_b)
                y_pred = surrogate(xi_prime)
                if cidx is not None:
                    loss = criterion(y_pred[:, cidx], y_b[:, cidx])
                else:
                    loss = criterion(y_pred, y_b)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            epoch_loss /= n_batches

            self.eval()
            with torch.no_grad():
                xi_prime_val = self(X_v)
                y_pred_val = surrogate(xi_prime_val)
                if cidx is not None:
                    val_loss = criterion(y_pred_val[:, cidx], Y_v[:, cidx]).item()
                else:
                    val_loss = criterion(y_pred_val, Y_v).item()
            scheduler.step(val_loss)
            if (epoch + 1) % 20 == 0:
                logger.info(f"Phase3 Epoch {epoch+1}/{epochs}: train={epoch_loss:.4e} val={val_loss:.4e}")

        for p in surrogate.parameters():
            p.requires_grad_(True)
        self.to('cpu')
        surrogate.to('cpu')
        return self

    def predict(self, X):
        """Predict, input numpy array, output numpy array."""
        self.eval()
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32)
            return self(x_t).numpy()
