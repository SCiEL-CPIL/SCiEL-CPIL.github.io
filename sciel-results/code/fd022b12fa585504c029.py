"""Bermudan put exercise policy via backward induction with Chebyshev regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
from scipy.stats import norm
from scipy.special import roots_chebyshev

_DEADLINE = 145.0  # seconds; safety margin under 150 s wall-clock limit


def _bs_put_price(S, K, r, sigma, T):
    """European put price using Black-Scholes."""
    if T <= 0:
        return np.maximum(K - S, 0.0)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(np.maximum(S, 1e-12) / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    N = norm.cdf
    price = K * np.exp(-r * T) * N(-d2) - S * N(-d1)
    return np.maximum(price, 0.0)


def _chebyshev_basis(x, n):
    """Evaluate Chebyshev polynomials of first kind at x (normalized to [-1,1]).
    Returns array of shape (len(x), n+1)."""
    x = np.asarray(x, dtype=np.float64)
    # Map to [-1, 1]
    xn = 2.0 * x - 1.0  # assuming x in [0, 1]
    # Actually we need to map the actual S range to [-1,1]
    # For generality, let's just use x as-is if it's already in [-1,1]
    # We'll handle mapping in the caller
    
    # Clenshaw's method for stability
    n = int(n)
    if n == 0:
        return np.ones((len(x), 1))
    
    # T_0 = 1, T_1 = x
    T = np.ones((len(x), n + 1), dtype=np.float64)
    T[:, 0] = 1.0
    if n >= 1:
        T[:, 1] = xn
    for k in range(2, n + 1):
        T[:, k] = 2.0 * xn * T[:, k - 1] - T[:, k - 2]
    return T


def _chebyshev_fit_predict(X_train, y_train, degree, X_predict):
    """Fit Chebyshev polynomial regression and predict.
    X_train, X_predict: 1D arrays in [0,1] (will be mapped to [-1,1])
    y_train: 1D array of targets
    Returns predictions for X_predict."""
    # Map to [-1, 1]
    Xn_train = 2.0 * X_train - 1.0
    Xn_pred = 2.0 * X_predict - 1.0
    
    # Build design matrix
    n = int(degree)
    T_train = np.ones((len(X_train), n + 1), dtype=np.float64)
    T_train[:, 0] = 1.0
    if n >= 1:
        T_train[:, 1] = Xn_train
    for k in range(2, n + 1):
        T_train[:, k] = 2.0 * Xn_train * T_train[:, k - 1] - T_train[:, k - 2]
    
    # Solve least squares
    coeffs, _, _, _ = np.linalg.lstsq(T_train, y_train, rcond=None)
    
    # Build prediction design matrix
    T_pred = np.ones((len(X_predict), n + 1), dtype=np.float64)
    T_pred[:, 0] = 1.0
    if n >= 1:
        T_pred[:, 1] = Xn_pred
    for k in range(2, n + 1):
        T_pred[:, k] = 2.0 * Xn_pred * T_pred[:, k - 1] - T_pred[:, k - 2]
    
    return T_pred @ coeffs


class ExercisePolicy:
    def __init__(self, thresholds):
        """thresholds: array of 11 thresholds (for t=1..11), index i corresponds to step i+1."""
        self.thresholds = thresholds
    
    def exercise(self, current_state, time):
        """
        current_state: array of spot prices
        time: array of time values (years)
        Returns boolean array: True if exercise.
        """
        current_state = np.asarray(current_state, dtype=np.float64)
        time = np.asarray(time, dtype=np.float64)
        
        # Determine which exercise step each path is at
        # Exercise dates are at t = 1/12, 2/12, ..., 12/12
        # Step index = round(time * 12) - 1, clamped to [0, 10]
        step_idx = np.round(time * 12.0).astype(int) - 1
        step_idx = np.clip(step_idx, 0, 10)
        
        # For each path, compare spot to threshold at that step
        thresh = self.thresholds[step_idx]
        # Exercise put if S < threshold
        return current_state < thresh


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction with Chebyshev regression.
    training_paths: array of shape (n_paths, n_time_steps) of GBM paths
    contract: dict with contract parameters
    Returns an ExercisePolicy object.
    """
    t_start = _time.time()
    
    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_steps = contract.get('n_steps', 12)  # 12 monthly exercise dates
    
    # Parse training paths
    paths = np.asarray(training_paths, dtype=np.float64)
    if paths.ndim == 1:
        paths = paths.reshape(-1, 1)
    
    n_paths = paths.shape[0]
    n_time = paths.shape[1]
    
    # We expect paths to have n_steps+1 time points (t=0, 1/12, ..., 1)
    # Exercise dates are at indices 1..n_steps (not at t=0)
    
    # Check time budget
    if _time.time() - t_start > _DEADLINE - 10:
        # Fallback: simple threshold based on BS
        thresholds = np.array([K * 0.95] * n_steps)
        return ExercisePolicy(thresholds)
    
    # Backward induction
    # Value at maturity (last exercise date)
    # At the last step (index n_steps), value = max(K - S, 0)
    S_last = paths[:, n_steps]
    V_next = np.maximum(K - S_last, 0.0)
    
    # Store thresholds for each step
    thresholds = np.zeros(n_steps, dtype=np.float64)
    
    # For each step from n_steps-1 down to 1 (0-indexed step in exercise dates: 10 down to 0)
    for step in range(n_steps - 1, -1, -1):
        t = _time.time()
        if t - t_start > _DEADLINE - 5:
            break
        
        S_curr = paths[:, step]
        
        # Discounted continuation value
        dt = T / n_steps
        disc = np.exp(-r * dt)
        cont = disc * V_next
        
        # Intrinsic value
        intrinsic = np.maximum(K - S_curr, 0.0)
        
        # Exercise if intrinsic >= continuation
        # We want to find the threshold where intrinsic = continuation
        # For a put, this is monotonic in S (lower S -> higher intrinsic, lower continuation)
        
        # Fit continuation value as function of S using Chebyshev
        # Use only paths where continuation is meaningful
        degree = 4  # Moderate degree for stability
        
        # Normalize S to [0, 1] for Chebyshev
        S_min = 0.0
        S_max = max(2.0 * K, np.max(S_curr) * 1.1) if len(S_curr) > 0 else 2.0 * K
        S_norm = (S_curr - S_min) / (S_max - S_min)
        S_norm = np.clip(S_norm, 0.0, 1.0)
        
        # Fit continuation value
        cont_fit = _chebyshev_fit_predict(S_norm, cont, degree, S_norm)
        
        # Find threshold: where intrinsic = continuation
        # intrinsic = max(K - S, 0), continuation = f(S)
        # For S < K: K - S = f(S)
        # We solve for S in [0, K]
        
        # Sample points to find the crossing
        S_grid = np.linspace(0.0, K, 200)
        S_grid_norm = (S_grid - S_min) / (S_max - S_min)
        S_grid_norm = np.clip(S_grid_norm, 0.0, 1.0)
        
        intrinsic_grid = np.maximum(K - S_grid, 0.0)
        cont_grid = _chebyshev_fit_predict(S_norm, cont, degree, S_grid_norm)
        
        diff = intrinsic_grid - cont_grid
        # Find where diff changes sign from positive to negative
        # We want the largest S where intrinsic >= continuation
        
        # Find sign changes
        signs = np.sign(diff)
        # Find indices where sign changes from + to -
        sign_changes = np.where((signs[:-1] >= 0) & (signs[1:] < 0))[0]
        
        if len(sign_changes) > 0:
            # Take the last sign change (largest S)
            idx = sign_changes[-1]
            # Linear interpolation
            x0, x1 = S_grid[idx], S_grid[idx + 1]
            y0, y1 = diff[idx], diff[idx + 1]
            if abs(y1 - y0) > 1e-12:
                thresh_val = x0 - y0 * (x1 - x0) / (y1 - y0)
            else:
                thresh_val = x0
        else:
            # No sign change: check if always exercise or never exercise
            if np.all(diff >= 0):
                thresh_val = K  # Always exercise
            else:
                thresh_val = 0.0  # Never exercise
        
        thresholds[step] = thresh_val
        
        # Update V_next for next iteration
        V_next = np.maximum(intrinsic, cont)
    
    # Handle any steps that weren't processed due to time limit
    # Use the last computed threshold or a default
    for i in range(len(thresholds)):
        if thresholds[i] == 0.0 and i < n_steps - 1:
            # Estimate from neighboring threshold
            if i + 1 < n_steps and thresholds[i + 1] > 0:
                thresholds[i] = thresholds[i + 1] * 0.99
            else:
                thresholds[i] = K * 0.95
    
    return ExercisePolicy(thresholds)
# EVOLVE-BLOCK-END
