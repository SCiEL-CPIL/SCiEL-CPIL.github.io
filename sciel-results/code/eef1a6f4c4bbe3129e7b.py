"""
Strategy: Hybrid Basis Portfolio with Feature Whitening
1. On first call, generate multiple candidate bases (Poly, Trig, Exp, RBF).
2. For each base, perform a quick internal fit (using a few BFGS steps or least squares if linear) to estimate quality.
3. Select the best base.
4. Whitening: Normalize features to zero mean, unit variance to help the evaluator's BFGS.
5. equation(x, y, params): Reconstruct the whitened feature matrix for current x, 
   and return X_whitened @ params.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Global state to cache the decision
_state = {
    "initialized": False,
    "best_basis_type": None,
    "means": None,
    "stds": None,
    "x_train": None,
    "y_train": None
}

def _build_features(x, y, basis_type):
    """
    Builds the feature matrix for a given basis type.
    Returns a matrix of shape (n, 10).
    """
    n = len(x)
    ones = np.ones(n)
    
    if basis_type == 'poly':
        # Multivariate polynomial up to degree 3 or 4
        # 1, x, y, x2, y2, xy, x3, y3, x2y, xy2
        return np.column_stack([
            ones, x, y, x**2, y**2, x*y, x**3, y**3, x**2*y, x*y**2
        ])
    elif basis_type == 'trig':
        # Trigonometric functions
        # 1, sin(x), cos(x), sin(y), cos(y), sin(x+y), cos(x+y), sin(x-y), cos(x-y), sin(2x)
        return np.column_stack([
            ones, np.sin(x), np.cos(x), np.sin(y), np.cos(y),
            np.sin(x+y), np.cos(x+y), np.sin(x-y), np.cos(x-y), np.sin(2*x)
        ])
    elif basis_type == 'exp':
        # Exponential decay/growth
        # 1, exp(-x), exp(-y), exp(-x^2), exp(-y^2), exp(-|x|), exp(-|y|), exp(-x*y), exp(-(x^2+y^2)), x*exp(-x^2)
        return np.column_stack([
            ones, np.exp(-x), np.exp(-y), np.exp(-x**2), np.exp(-y**2),
            np.exp(-np.abs(x)), np.exp(-np.abs(y)), np.exp(-x*y), 
            np.exp(-(x**2+y**2)), x*np.exp(-x**2)
        ])
    elif basis_type == 'rbf':
        # Radial Basis Functions (Gaussian kernels)
        # Centers chosen to cover [-1, 1]x[-1, 1]
        # 1, K((0,0)), K((0.5,0.5)), K((-0.5,-0.5)), K((0.5,-0.5)), K((-0.5,0.5)), 
        # K((0, 0.5)), K((0, -0.5)), K((0.5, 0)), K((-0.5, 0))
        # Gamma = 1.0 (width ~ 1)
        gamma = 1.0
        centers = [(0, 0), (0.5, 0.5), (-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5),
                   (0, 0.5), (0, -0.5), (0.5, 0), (-0.5, 0)]
        
        feats = [ones]
        for cx, cy in centers:
            dist_sq = (x - cx)**2 + (y - cy)**2
            feats.append(np.exp(-gamma * dist_sq))
        
        return np.column_stack(feats)
    elif basis_type == 'mixed':
        # Mix of polynomial and trigonometric
        return np.column_stack([
            ones, x, y, x**2, y**2, x*y, np.sin(x), np.cos(y), x*np.sin(y), y*np.cos(x)
        ])
    else:
        # Fallback to poly
        return np.column_stack([
            ones, x, y, x**2, y**2, x*y, x**3, y**3, x**2*y, x*y**2
        ])

def _calculate_mse(pred, y):
    return np.mean((pred - y)**2)

def _fit_basis(x, y, basis_type):
    """
    Fits the basis and returns the best MSE found.
    Uses a quick BFGS run.
    """
    X = _build_features(x, y, basis_type)
    
    # Check for NaN/Inf
    if not np.all(np.isfinite(X)):
        return float('inf')
        
    # Whitening for better conditioning in the quick fit
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    stds[stds < 1e-8] = 1.0 # Avoid division by zero
    
    X_white = (X - means) / stds
    
    def obj(p):
        pred = X_white @ p
        if not np.all(np.isfinite(pred)):
            return 1e30
        return _calculate_mse(pred, y)
    
    try:
        # Start from ones
        res = minimize(obj, np.ones(10), method='BFGS', options={'maxiter': 50, 'gtol': 1e-3})
        if res.success or res.fun < 1e30:
            return res.fun
        else:
            return 1e30
    except Exception:
        return float('inf')

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    global _state
    
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    # 1. Initialization Phase
    if not _state["initialized"]:
        _state["x_train"] = x
        _state["y_train"] = y
        
        # Candidate bases
        basis_types = ['poly', 'trig', 'exp', 'rbf', 'mixed']
        
        best_mse = float('inf')
        best_type = 'poly'
        
        for btype in basis_types:
            if time.time() > deadline:
                break
                
            mse = _fit_basis(x, y, btype)
            if mse < best_mse:
                best_mse = mse
                best_type = btype
                
        _state["best_basis_type"] = best_type
        _state["initialized"] = True
        
        # Calculate whitening stats based on the best basis and training data
        X_train = _build_features(x, y, best_type)
        if np.all(np.isfinite(X_train)):
            _state["means"] = np.mean(X_train, axis=0)
            _state["stds"] = np.std(X_train, axis=0)
            _state["stds"][_state["stds"] < 1e-8] = 1.0
        else:
            # Fallback stats if something went wrong
            _state["means"] = np.zeros(10)
            _state["stds"] = np.ones(10)
            
        # If we used up time, return a valid prediction immediately
        if time.time() > deadline:
             # Use the current params with whitened features
             X_white = (X_train - _state["means"]) / _state["stds"]
             pred = X_white @ params
             if not np.all(np.isfinite(pred)):
                 return np.zeros_like(y)
             return pred

    # 2. Prediction Phase
    # Build features for current x
    X_raw = _build_features(x, y, _state["best_basis_type"])
    
    if not np.all(np.isfinite(X_raw)):
        # Fallback to simple linear if features explode
        X_raw = np.column_stack([np.ones(len(x)), x, y, x**2, y**2, x*y, x**3, y**3, x**2*y, x*y**2])
        # Re-compute stats? No, just use identity whitening to be safe?
        # Better: If features are bad, return zeros or mean
        # Let's try to salvage with simple poly and identity whitening
        _state["means"] = np.zeros(10)
        _state["stds"] = np.ones(10)

    # Apply whitening using stats from training
    X_white = (X_raw - _state["means"]) / _state["stds"]
    
    # Ensure finite
    if not np.all(np.isfinite(X_white)):
        # Replace non-finite with 0
        X_white = np.nan_to_num(X_white, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute prediction
    pred = X_white @ params
    
    if not np.all(np.isfinite(pred)):
        return np.zeros_like(y)
        
    return pred

# EVOLVE-BLOCK-END
