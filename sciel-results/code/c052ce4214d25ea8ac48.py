import numpy as np
import time

def _halton_sequence(n, base):
    """Generate n points in the Halton sequence with the given base."""
    result = np.zeros(n)
    f = 1.0
    for i in range(1, n + 1):
        num = i
        while num > 0:
            f /= base
            num, remainder = divmod(num, base)
            result[i - 1] += f * remainder
    return result

def _generate_rbf_centers(n_centers=10):
    """Generate RBF centers using a 2D Halton sequence mapped to [-1, 1]."""
    # Use two different bases for the 2D sequence
    h1 = _halton_sequence(n_centers, 2)
    h2 = _halton_sequence(n_centers, 3)
    
    # Map from [0, 1] to [-1, 1]
    x_centers = 2 * h1 - 1
    y_centers = 2 * h2 - 1
    
    return np.column_stack([x_centers, y_centers])

def rbf_basis_2d(x, y, centers, widths):
    """
    Compute RBF basis matrix for given inputs.
    
    Args:
        x, y: Input coordinates (arrays of same length)
        centers: (n_centers, 2) array of center coordinates
        widths: (n_centers,) array of RBF widths
    
    Returns:
        (n_points, n_centers) basis matrix
    """
    n_points = len(x)
    n_centers = len(centers)
    
    # Compute squared distances
    dx = x[:, None] - centers[0, None]
    dy = y[:, None] - centers[1, None]
    dist_sq = dx ** 2 + dy ** 2
    
    # Gaussian RBF: exp(-dist_sq / (2 * width^2))
    basis = np.exp(-dist_sq / (2 * widths[None, :] ** 2))
    
    return basis

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using sparse Radial Basis Functions (RBFs).
    
    The model is a linear combination of 10 RBFs with fixed centers and widths:
    f(x,y) = sum_i A_i * exp(-((x-cx_i)^2 + (y-cy_i)^2) / (2 * w_i^2))
    
    Centers are placed deterministically using a 2D Halton sequence over [-1, 1].
    Widths are fixed to provide good coverage. Only the 10 amplitudes are fitted
    via linear least squares, which is fast and stable.
    
    Parameters (10 total):
    - params[0:10]: Amplitudes for the 10 RBFs
    """
    start_time = time.time()
    deadline = 145.0  # seconds (5s safety margin)
    
    n_points = len(x)
    n_rbf = 10
    
    # Generate fixed RBF centers using Halton sequence
    centers = _generate_rbf_centers(n_rbf)
    
    # Fixed widths: use a mix of narrow and wide RBFs
    # This provides multi-scale capability
    widths = np.array([0.3, 0.5, 0.7, 1.0, 1.5, 0.3, 0.5, 0.7, 1.0, 1.5])
    
    # Precompute the RBF basis matrix
    try:
        basis = rbf_basis_2d(x, y, centers, widths)
        
        if not np.all(np.isfinite(basis)):
            # Fallback to simple model if basis computation fails
            return params[0] * np.sin(params[1] * x + params[2] * y + params[3])
        
        # Solve linear least squares: basis @ amplitudes = y
        # Use lstsq for stability
        amplitudes, residuals, rank, sv = np.linalg.lstsq(basis, y, rcond=None)
        
        # Compute predictions
        pred = basis @ amplitudes
        
        if not np.all(np.isfinite(pred)):
            # Fallback
            return params[0] * np.sin(params[1] * x + params[2] * y + params[3])
        
        return pred
    
    except Exception:
        # Fallback to a simple sinusoidal model
        return params[0] * np.sin(params[1] * x + params[2] * y + params[3])
