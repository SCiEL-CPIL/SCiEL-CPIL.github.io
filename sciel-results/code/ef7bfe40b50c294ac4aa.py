# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_N = 10
_VALS = np.arange(_N)
_YS = np.repeat(_VALS, _N)
_XS = np.tile(_VALS, _N)
_XY = _XS + _YS
_X2Y = _XS + 2 * _YS
_XM1Y = _XS - _YS  # range -9..9

# Pre-compute bin indices for efficiency
_IDX_X = _XS
_IDX_XY = _XY
_IDX_X2Y = _X2Y
_IDX_XM1Y = _XM1Y + 9  # Shift to 0..18

def _entropy(p):
    """Compute Shannon entropy in bits, ignoring zero probabilities."""
    s = p.sum()
    if s <= 1e-30:
        return 0.0
    q = p / s
    q = q[q > 0.0]
    if len(q) == 0:
        return 0.0
    return float(-(q * np.log2(q)).sum())

def _compute_value(P):
    """Compute the ratio D/M for a given distribution P using hard max."""
    p = P.ravel()
    total = p.sum()
    if total <= 1e-30:
        return 0.0

    # D: distribution of x - y (slope -1), 19 bins
    d_bins = np.zeros(19)
    np.add.at(d_bins, _IDX_XM1Y, p)
    hd = _entropy(d_bins)

    # H_0: distribution of x, 10 bins
    h0_bins = np.zeros(10)
    np.add.at(h0_bins, _IDX_X, p)
    h0 = _entropy(h0_bins)

    # H_1: distribution of x + y, 19 bins
    h1_bins = np.zeros(19)
    np.add.at(h1_bins, _IDX_XY, p)
    h1 = _entropy(h1_bins)

    # H_2: distribution of x + 2y, 28 bins
    h2_bins = np.zeros(28)
    np.add.at(h2_bins, _IDX_X2Y, p)
    h2 = _entropy(h2_bins)

    m = max(h0, h1, h2)
    if m <= 1e-9:
        return 0.0
    return hd / m

def _build_from_theta(theta):
    """Convert log-probability parameters to a probability distribution."""
    theta_shifted = theta - np.max(theta)
    p = np.exp(theta_shifted)
    s = p.sum()
    if s <= 1e-30:
        return np.ones(_N * _N) / (_N * _N)
    p = p / s
    return p

def _soft_max(vals, t=10.0):
    """
    Approximate max(vals) using log-sum-exp.
    Larger t makes the approximation sharper (closer to max).
    """
    # Shift for numerical stability
    v = np.array(vals)
    v_max = np.max(v)
    # If all are very small/negative, handle carefully
    # log-sum-exp: v_max + log(sum(exp(v - v_max)))
    # We want max, so we use a high temperature parameter in the exponent?
    # Actually, max(x) = lim_{t->inf} (1/t) log(sum(exp(t*x)))
    # So we compute (1/t) * log(sum(exp(t*x)))
    # To avoid overflow, we use: (1/t) * (t*v_max + log(sum(exp(t*(v - v_max)))))
    # = v_max + (1/t) * log(sum(exp(t*(v - v_max))))
    
    # If t is large, exp(t*(v-v_max)) will be 1 for max and 0 for others.
    # log(sum) will be log(1) = 0. So result is v_max.
    # For t=10, it's a smooth approximation.
    
    # Let's use a simpler robust soft-max:
    # soft_max = v_max + (1/t) * log(sum(exp(t * (v - v_max))))
    
    # Handle case where v_max is -inf or similar
    if not np.isfinite(v_max):
        return 0.0
        
    # Scale
    scaled = t * (v - v_max)
    # Clip to avoid underflow/overflow issues in exp, though v-v_max <= 0 so exp <= 1
    # exp(scaled) is safe as long as scaled is not too negative (underflow to 0 is fine)
    exp_sum = np.sum(np.exp(scaled))
    if exp_sum <= 0:
        return v_max
    return v_max + (1.0/t) * np.log(exp_sum)

def _objective_soft(theta, t=10.0):
    """
    Negative of the soft-max value we want to maximize.
    Uses soft-max for M to ensure differentiability.
    """
    p = _build_from_theta(theta)
    
    # Compute entropies
    # D
    d_bins = np.zeros(19)
    np.add.at(d_bins, _IDX_XM1Y, p)
    hd = _entropy(d_bins)
    
    # H0
    h0_bins = np.zeros(10)
    np.add.at(h0_bins, _IDX_X, p)
    h0 = _entropy(h0_bins)
    
    # H1
    h1_bins = np.zeros(19)
    np.add.at(h1_bins, _IDX_XY, p)
    h1 = _entropy(h1_bins)
    
    # H2
    h2_bins = np.zeros(28)
    np.add.at(h2_bins, _IDX_X2Y, p)
    h2 = _entropy(h2_bins)
    
    m_soft = _soft_max([h0, h1, h2], t)
    if m_soft <= 1e-9:
        return 0.0
    return -hd / m_soft

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    
    Uses multiple random restarts with L-BFGS-B in log-probability space
    to maximize D/M where D = H_{x-y} and M = max(H_x, H_{x+y}, H_{x+2y}).
    Uses a soft-max surrogate for M during optimization to handle non-smoothness.
    
    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1.
    """
    deadline = time.time() + 550  # Safety margin before 600s limit
    n_cells = _N * _N
    
    best_P = np.ones((_N, _N), dtype=np.float64) / n_cells
    best_val = _compute_value(best_P)
    
    rng = np.random.RandomState(42)
    
    # Strategy 1: Start from uniform and optimize with soft-max objective
    theta0 = np.zeros(n_cells)
    result = minimize(
        _objective_soft, theta0, method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    if time.time() > deadline:
        return best_P
    
    p_opt = _build_from_theta(result.x)
    P_opt = p_opt.reshape(_N, _N)
    val_opt = _compute_value(P_opt)
    if val_opt > best_val:
        best_val = val_opt
        best_P = P_opt.copy()
    
    # Strategy 2: Multiple random restarts
    n_restarts = 50 # Increased restarts as soft-max may converge faster/more reliably
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Random initialization in log-space
        if restart % 3 == 0:
            # Random sparse initialization
            theta = rng.randn(n_cells) * 2.0
        elif restart % 3 == 1:
            # Concentrate on anti-diagonal
            theta = np.full(n_cells, -10.0)
            for y in range(_N):
                for x in range(_N):
                    if x + y == 9:
                        theta[y * _N + x] = 0.0
                    elif abs(x - y) >= 5:
                        theta[y * _N + x] = -2.0
        else:
            # Concentrate on corners
            theta = np.full(n_cells, -10.0)
            corners = [(0, 0), (0, 9), (9, 0), (9, 9)]
            for (cy, cx) in corners:
                theta[cy * _N + cx] = 0.0
            for y in range(_N):
                for x in range(_N):
                    if abs(x - y) >= 4:
                        theta[y * _N + x] = -3.0
        
        result = minimize(
            _objective_soft, theta, method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        p_opt = _build_from_theta(result.x)
        P_opt = p_opt.reshape(_N, _N)
        val_opt = _compute_value(P_opt)
        if val_opt > best_val:
            best_val = val_opt
            best_P = P_opt.copy()
    
    # Strategy 3: Refine the best solution with higher precision (hard max objective for final polish?)
    # Actually, sticking to soft-max for smoothness is safer, but we can increase t
    # Or switch to hard max for a final polish if we are confident in the local region
    if best_val > 0 and time.time() < deadline - 60:
        p_flat = best_P.ravel()
        theta_best = np.log(p_flat + 1e-15)
        
        # Use a higher t for a sharper approximation of max, effectively switching to hard max behavior
        result = minimize(
            lambda th: _objective_soft(th, t=50.0), theta_best, method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        p_opt = _build_from_theta(result.x)
        P_opt = p_opt.reshape(_N, _N)
        val_opt = _compute_value(P_opt)
        if val_opt > best_val:
            best_val = val_opt
            best_P = P_opt.copy()
            
    # Strategy 4: Structured grid search (kept from original as it's fast)
    if time.time() < deadline - 30:
        for beta in [0.0, 0.5, 1.0, 2.0, 5.0]:
            for gamma in [0.0, 0.5, 1.0, 2.0]:
                if time.time() > deadline - 10:
                    break
                P_test = np.zeros((_N, _N))
                for y in range(_N):
                    for x in range(_N):
                        P_test[y, x] = np.exp(-beta * (x - y) ** 2 - gamma * x ** 2)
                s = P_test.sum()
                if s > 1e-30:
                    P_test /= s
                    val_test = _compute_value(P_test)
                    if val_test > best_val:
                        best_val = val_test
                        best_P = P_test.copy()
    
    # Final cleanup
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s <= 1e-9:
        best_P = np.ones((_N, _N), dtype=np.float64) / n_cells
    else:
        best_P /= s
    
    return best_P


# EVOLVE-BLOCK-END
