import numpy as np


class DimensionReducer:
    """
    Base class for dimension reducer M: xi_E (5D) -> xi' (3D).
    Subclasses implement the actual reduction logic.
    """

    def __init__(self, input_dim=5, output_dim=3):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def reduce(self, xi_E):
        """
        Map original KL coefficients to reduced parameters.
        Args:
            xi_E: shape (n_samples, input_dim) or (input_dim,)
        Returns:
            xi_prime: shape (n_samples, output_dim) or (output_dim,)
        """
        raise NotImplementedError

    def fit(self, X_train, Y_train, surrogate, **kwargs):
        """
        Train the dimension reducer using output-space loss with frozen surrogate.
        Args:
            X_train: shape (n_train, input_dim) - KL coefficients
            Y_train: shape (n_train, n_x) - settlement profiles
            surrogate: frozen surrogate S that maps xi' -> Y'
        """
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError
