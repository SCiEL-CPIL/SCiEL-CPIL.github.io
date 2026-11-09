# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt
import itertools

# Global variables for caching intermediate computations
N = 10
X_GRID, Y_GRID = np.meshgrid(np.arange(N), np.arange(N), indexing='xy')
# Flatten for vectorized operations
X_FLAT = X_GRID.flatten()
Y_FLAT = Y_GRID.flatten()

# Pre-compute projection values
U_VALS = X_FLAT - Y_FLAT       # x - y
V_VALS = X_FLAT                # x
W_VALS = X_FLAT + Y_FLAT       # x + y
Z_VALS = X_FLAT + 2 * Y_FLAT   # x + 2y

# Unique values and inverse indices for bincount
U_UNIQ, U_INV = np.unique(U_VALS, return_inverse=True)
V_UNIQ, V_INV = np.unique(V_VALS, return_inverse=True)
W_UNIQ, W_INV = np.unique(W_VALS, return_inverse=True)
Z_UNIQ, Z_INV = np.unique(Z_VALS, return_inverse=True)

def _entropy_from_probs(probs):
    """Calculate Shannon entropy in bits, ignoring zeros."""
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))

def _compute_ratio(P_flat):
    """
    Compute the ratio D / M for a given flat probability vector.
    P_flat: 1D array of length 100.
    """
    P_flat = np.clip(P_flat, 1e-15, None)
    total = np.sum(P_flat)
    if total <= 1e-9:
        return 0.0
    P_flat = P_flat / total
    
    # Project probabilities
    p_u = np.bincount(U_INV, weights=P_flat, minlength=len(U_UNIQ))
    p_v = np.bincount(V_INV, weights=P_flat, minlength=len(V_UNIQ))
    p_w = np.bincount(W_INV, weights=P_flat, minlength=len(W_UNIQ))
    p_z = np.bincount(Z_INV, weights=P_flat, minlength=len(Z_UNIQ))
    
    h_u = _entropy_from_probs(p_u)
    h_v = _entropy_from_probs(p_v)
    h_w = _entropy_from_probs(p_w)
    h_z = _entropy_from_probs(p_z)
    
    D = h_u
    M = max(h_v, h_w, h_z)
    
    if M <= 1e-9:
        return 0.0
        
    return D / M

def _generate_gaussian(params):
    """
    Generate a 10x10 probability distribution from Gaussian parameters.
    params: [cx, cy, sigma_x, sigma_y, rho]
    """
    cx, cy, sx, sy, rho = params
    # Ensure positive sigmas
    sx = abs(sx) + 1e-6
    sy = abs(sy) + 1e-6
    # Clamp rho to (-1, 1)
    rho = np.clip(rho, -0.999, 0.999)
    
    X = X_GRID
    Y = Y_GRID
    
    dx = X - cx
    dy = Y - cy
    
    # Exponent of the bivariate Gaussian
    # Using simplified form for stability
    z = (dx**2 / (2*sx**2) + dy**2 / (2*sy**2) - rho * dx * dy / (sx * sy))
    
    P = np.exp(-z)
    P = np.clip(P, 1e-15, None)
    total = np.sum(P)
    if total <= 1e-9:
        # Fallback to uniform if degenerate
        return np.ones(100) / 100.0
    return P.flatten() / total

def _objective(params):
    """
    Objective function for minimization (negative ratio).
    """
    P_flat = _generate_gaussian(params)
    ratio = _compute_ratio(P_flat)
    return -ratio

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimize a parametric Gaussian distribution to maximize the arithmetic Kakeya entropy ratio.
    """
    start_time = time.time()
    deadline = start_time + 500  # 500s safety margin for 600s limit
    
    best_ratio = -np.inf
    best_P = np.ones(100) / 100.0
    
    # Initial uniform
    uniform_P = np.ones(100) / 100.0
    best_ratio = _compute_ratio(uniform_P)
    best_P = uniform_P
    
    # Define bounds for parameters:
    # cx, cy in [0, 9]
    # sx, sy in [0.1, 10]
    # rho in [-1, 1]
    bounds = [(0, 9), (0, 9), (0.1, 10), (0.1, 10), (-1, 1)]
    
    # Initial guesses
    # 1. Center
    # 2. Off-center
    # 3. Corner
    initial_guesses = [
        [4.5, 4.5, 1.0, 1.0, 0.0],
        [4.5, 4.5, 2.0, 0.5, 0.0],
        [4.5, 4.5, 0.5, 2.0, 0.0],
        [2.0, 2.0, 1.0, 1.0, 0.5],
        [7.0, 7.0, 1.0, 1.0, -0.5],
        [4.5, 4.5, 3.0, 3.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [9.0, 9.0, 1.0, 1.0, 0.0],
    ]
    
    # Add random restarts
    rng = np.random.default_rng(42)
    for _ in range(20):
        cx = rng.uniform(0, 9)
        cy = rng.uniform(0, 9)
        sx = rng.uniform(0.5, 5.0)
        sy = rng.uniform(0.5, 5.0)
        rho = rng.uniform(-0.9, 0.9)
        initial_guesses.append([cx, cy, sx, sy, rho])
        
    for guess in initial_guesses:
        if time.time() > deadline:
            break
            
        try:
            res = opt.minimize(
                _objective, 
                guess, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-6}
            )
            
            if res.success:
                P_opt = _generate_gaussian(res.x)
                ratio_opt = _compute_ratio(P_opt)
                
                if ratio_opt > best_ratio:
                    best_ratio = ratio_opt
                    best_P = P_opt
                    
        except Exception:
            continue
            
    # If time remains, try a second pass with Nelder-Mead on the best found
    if time.time() < deadline - 50 and best_ratio > 0:
        # Reconstruct params from best_P is hard, so we just stick with the best found 
        # or do a local search around the best parameters if we saved them.
        # We didn't save params, so we skip this to save code complexity, 
        # but the L-BFGS-B from multiple starts is usually sufficient.
        pass
    P_final = best_P.reshape((10, 10))
    
    # Final validation and normalization
    P_final = np.clip(P_final, 0, None)
    total = np.sum(P_final)
    if total <= 1e-9:
        P_final = np.ones((10, 10)) / 100.0
    else:
        P_final = P_final / total
        
    return P_final

# EVOLVE-BLOCK-END
