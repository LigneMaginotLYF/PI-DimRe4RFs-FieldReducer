import numpy as np


def build_idct_basis(n_modes, n_nodes):
    """Build a truncated inverse-DCT-II basis matrix.

    Returns ``D`` with shape ``(n_nodes, n_modes)`` where column ``D[:, k]``
    is the k-th DCT-II orthonormal basis vector of length ``n_nodes``.

    The node-space reconstruction of a coefficient vector ``c`` (length
    ``n_modes``) is::

        y_nodes = D @ c                      # single sample
        Y_nodes = C @ D.T                    # batch: C has shape (batch, n_modes)

    The batch form is differentiable through ``torch.matmul`` when ``D.T`` is
    stored as a ``torch.Tensor``.

    Args:
        n_modes: Number of DCT modes K to retain.
        n_nodes: Number of spatial nodes N in the full profile.

    Returns:
        D: numpy array of shape (n_nodes, n_modes), dtype float32.
    """
    from scipy.fft import idct as _idct
    K = int(n_modes)
    N = int(n_nodes)
    D = np.zeros((N, K), dtype=np.float32)
    for k in range(K):
        e_k = np.zeros(N, dtype=np.float32)
        e_k[k] = 1.0
        D[:, k] = _idct(e_k, type=2, norm='ortho')
    return D
