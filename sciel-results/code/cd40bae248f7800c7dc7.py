"""Symbolic regression: nonlinear oscillator (2) - closed-form equation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with time-dependent driving force.
    
    Uses a Gaussian Radial Basis Function (RBF) network with 10 fitted constants.
    The RBF centers are fixed at distinct points within the input ranges, and widths
    are scaled to the domain size. This provides a smooth, localized nonlinear basis.
    
    Inputs:
        t: time [30, 50]
        x: position [-0.2056, 0.1749]
        v: velocity [-0.3391, 0.3624]
        params: array of 10 fitted constants
    
    Returns:
        Prediction array with one value per row
    """
    # Normalize inputs to [0, 1] for stable RBF computation
    t_norm = (t - 30.0) / (50.0 - 30.0)
    x_norm = (x - (-0.2056)) / (0.1749 - (-0.2056))
    v_norm = (v - (-0.3391)) / (0.3624 - (-0.3391))
    
    # Define 10 RBF centers in the normalized 3D space
    # Spread centers across the domain to cover different regions
    centers = np.array([
        [0.1, 0.2, 0.3],
        [0.3, 0.7, 0.5],
        [0.5, 0.4, 0.8],
        [0.7, 0.1, 0.2],
        [0.9, 0.8, 0.6],
        [0.2, 0.5, 0.7],
        [0.8, 0.3, 0.4],
        [0.4, 0.9, 0.1],
        [0.6, 0.6, 0.9],
        [0.5, 0.5, 0.5]
    ])
    
    # Widths scaled to domain size (in normalized space)
    width = 0.4
    
    # Compute RBF activations
    # t_norm, x_norm, v_norm have shape (N,)
    # centers has shape (10, 3)
    
    # Reshape for broadcasting: (N, 1, 3) - (1, 10, 3) -> (N, 10, 3)
    diff = np.stack([t_norm, x_norm, v_norm], axis=1)[:, np.newaxis, :] - centers[np.newaxis, :, :]
    
    # Squared distances: (N, 10)
    sq_dist = np.sum(diff ** 2, axis=2)
    
    # RBF activations: exp(-sq_dist / (2 * width^2))
    activations = np.exp(-sq_dist / (2.0 * width ** 2))
    
    # Weighted sum of RBF activations
    # params has shape (10,)
    prediction = np.sum(activations * params[np.newaxis, :], axis=1)
    
    return prediction
# EVOLVE-BLOCK-END
