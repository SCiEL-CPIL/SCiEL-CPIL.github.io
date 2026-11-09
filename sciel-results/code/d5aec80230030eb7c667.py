# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def _precompute_bins(n=10):
    """
    Pre-compute the binning indices for the slopes -1, 0, 1, 2.
    Returns a dict mapping slope to (unique_vals, inverse_indices).
    """
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    slopes = [-1.0, 0.0, 1.0, 2.0]
    bins = {}
    for r in slopes:
        values = X + r * Y
        # We only need the inverse indices to sum probabilities
        # unique_vals are not strictly needed for entropy if we just sum into a vector of size len(unique_vals)
        # But we need to know the size.
        unique_vals, inverse = np.unique(values, return_inverse=True)
        bins[r] = (len(unique_vals), inverse)
        
    return bins

def _fast_objective(z, bins):
    """
    Objective function for optimization.
    z is a flat array of length 100.
    Returns - (D / M) to be minimized.
    """
    n = 10
    P = z.reshape(10, 10)
    # Convert to probabilities using softmax
    # Subtract max for numerical stability
    P_exp = np.exp(P - np.max(P))
    P_norm = P_exp / np.sum(P_exp)
    
    # Compute entropies using pre-computed bins
    # We need to sum P_norm into the bins
    # For each slope, we create a vector of probabilities and compute entropy
    
    Hs = {}
    for r, (num_bins, inverse) in bins.items():
        probs = np.zeros(num_bins)
        np.add.at(probs, inverse, P_norm)
        
        # Filter out zero probabilities
        probs = probs[probs > 1e-12]
        if len(probs) == 0:
            Hs[r] = 0.0
        else:
            Hs[r] = -np.sum(probs * np.log2(probs))
            
    D = Hs[-1.0]
    M = max(Hs[0.0], Hs[1.0], Hs[2.0])
    
    if M < 1e-9:
        return 1e9 # Large penalty
        
    return -D / M

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio.

    This implementation uses Differential Evolution (a global, gradient-free, population-based
    optimizer) to maximize the ratio D/M. It pre-computes the binning structure for the
    projections to speed up the objective function evaluations.
    """
    n = 10
    # Pre-compute bins
    bins = _precompute_bins(n)
    
    # Define bounds for z (log-probabilities)
    # z in [-10, 10] covers a wide range of probabilities
    bounds = [(-10.0, 10.0)] * (n * n)
    
    # Objective function wrapper
    def obj(z):
        return _fast_objective(z, bins)
        
    # Run Differential Evolution
    # seed for reproducibility
    res = differential_evolution(obj, bounds, seed=42, maxiter=500, popsize=15, tol=1e-6, polish=True)
    
    # Extract best P
    P_opt = res.x.reshape(n, n)
    P_exp = np.exp(P_opt - np.max(P_opt))
    P_norm = P_exp / np.sum(P_exp)
    
    # Verify validity
    if np.any(np.isnan(P_norm)) or np.any(np.isinf(P_norm)):
        # Fallback to uniform
        P_norm = np.ones((n, n)) / (n * n)
    else:
        # Normalize again just in case
        P_norm = P_norm / np.sum(P_norm)
        
    return P_norm

# EVOLVE-BLOCK-END
