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
                           colloc_idx=None, output_representation='direct',
                           n_output_modes=None, n_nodes_x=None,
                           bspline_degree=3):
        """
        Train dimension reducer M using frozen surrogate S.

        Loss is always computed in *node space* so that collocation indices
        (which are node indices 0..n_x-1) remain valid regardless of the
        surrogate's internal output representation.

        When ``output_representation`` is not ``'direct'``, a differentiable
        inverse-basis projection matrix is pre-computed once and used inside
        the training loop to convert surrogate outputs back to n_x node values
        before applying ``colloc_idx``:

        - ``'dct'``: inverse-DCT projection from K DCT-II coefficients to N nodes.
        - ``'poly'``: Vandermonde evaluation from K polynomial coefficients to N nodes.
        - ``'bspline'``: B-spline evaluation from K B-spline coefficients to N nodes.

        Args:
            X_train: xi_E training data, shape (n_train, input_dim)
            Y_train: reference responses in node space, shape (n_train, n_x)
            X_val, Y_val: validation set
            surrogate: frozen PhysicsDrivenMappingNN (S), frozen during training
            epochs, lr, batch_size: training hyperparameters
            colloc_idx: 1-D integer array of output node indices to include in
                the loss.  If None, all output nodes are used (full-profile MSE).
            output_representation: 'direct' | 'dct' | 'poly' | 'bspline'.
            n_output_modes: number of output modes K.
            n_nodes_x: number of spatial nodes N.
            bspline_degree: B-spline degree (only used when 'bspline').
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

        # Pre-compute differentiable inverse-basis projection matrix.
        # coeff_proj has shape (K, N); the forward pass computes:
        #   y_nodes = y_coeff @ coeff_proj   (B,K) @ (K,N) -> (B,N)
        # This is fully differentiable through torch.matmul.
        coeff_proj = None
        if n_output_modes is not None and n_nodes_x is not None:
            if output_representation == 'dct':
                from src.dct_utils import build_idct_basis
                D = build_idct_basis(n_modes=n_output_modes, n_nodes=n_nodes_x)
                coeff_proj_np = D.T  # (K, N): y_nodes = coeff @ coeff_proj
                if colloc_idx is not None and len(colloc_idx) > int(n_output_modes):
                    logger.warning(
                        f"Phase-3: collocation covers {len(colloc_idx)} nodes but "
                        f"surrogate has only {n_output_modes} DCT output modes. "
                        "Loss is computed in node space via inverse-DCT projection."
                    )
                logger.info(
                    f"Phase-3 fit_with_surrogate: DCT mode, K={n_output_modes}, "
                    f"N={n_nodes_x}, using differentiable IDCT projection for node-space loss."
                )
            elif output_representation == 'poly':
                # Vandermonde matrix: V[node, k] = x_norm[node]^(K-1-k)
                import numpy as np_  # avoid shadowing outer np
                x_norm = np_.linspace(0, 1, n_nodes_x)
                K = n_output_modes
                powers = np_.arange(K - 1, -1, -1, dtype=np_.float64)
                V = x_norm[:, None] ** powers[None, :]  # (N, K)
                coeff_proj_np = V.T  # (K, N): y_nodes = coeff @ V.T
                logger.info(
                    f"Phase-3 fit_with_surrogate: poly mode, K={n_output_modes}, "
                    f"N={n_nodes_x}, using differentiable Vandermonde projection."
                )
            elif output_representation == 'bspline':
                import numpy as np_
                from scipy.interpolate import BSpline
                k = bspline_degree
                x_norm = np_.linspace(0, 1, n_nodes_x)
                K = n_output_modes
                if K <= k:
                    raise ValueError(
                        f"n_output_modes={K} must be > bspline_degree={k}."
                    )
                n_internal = K - k - 1
                if n_internal > 0:
                    t_internal = np_.linspace(0, 1, n_internal + 2)[1:-1]
                else:
                    t_internal = np_.array([])
                t_full = np_.concatenate([
                    np_.zeros(k + 1), t_internal, np_.ones(k + 1)
                ])
                B = np_.column_stack([
                    BSpline.basis_element(
                        t_full[i:i + k + 2], extrapolate=False
                    )(x_norm)
                    for i in range(K)
                ])
                B = np_.nan_to_num(B, nan=0.0)  # (N, K)
                coeff_proj_np = B.T  # (K, N)
                logger.info(
                    f"Phase-3 fit_with_surrogate: bspline mode, K={n_output_modes}, "
                    f"degree={k}, N={n_nodes_x}, using differentiable B-spline projection."
                )
            else:
                coeff_proj_np = None

            if coeff_proj_np is not None:
                coeff_proj = torch.tensor(coeff_proj_np, dtype=torch.float32).to(device)
        # Legacy path for dct when n_output_modes/n_nodes_x not supplied
        dct_proj = coeff_proj  # keep internal name for backward compat in loop below

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
                y_pred_raw = surrogate(xi_prime)
                # Project to node space when surrogate outputs basis coefficients
                if dct_proj is not None:
                    y_pred = torch.matmul(y_pred_raw, dct_proj)
                else:
                    y_pred = y_pred_raw
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
                y_pred_val_raw = surrogate(xi_prime_val)
                if dct_proj is not None:
                    y_pred_val = torch.matmul(y_pred_val_raw, dct_proj)
                else:
                    y_pred_val = y_pred_val_raw
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
