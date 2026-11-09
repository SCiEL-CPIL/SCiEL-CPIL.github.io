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
    """Compute the ratio D/M for a given distribution P."""
    p = P.ravel()
    total = p.sum()
    if total <= 1e-30:
        return 0.0

    # D: distribution of x - y (slope -1), 19 bins
    d_bins = np.zeros(19)
    np.add.at(d_bins, _XM1Y + 9, p)
    hd = _entropy(d_bins)

    # H_0: distribution of x, 10 bins
    h0_bins = np.zeros(10)
    np.add.at(h0_bins, _XS, p)
    h0 = _entropy(h0_bins)

    # H_1: distribution of x + y, 19 bins
    h1_bins = np.zeros(19)
    np.add.at(h1_bins, _XY, p)
    h1 = _entropy(h1_bins)

    # H_2: distribution of x + 2y, 28 bins
    h2_bins = np.zeros(28)
    np.add.at(h2_bins, _X2Y, p)
    h2 = _entropy(h2_bins)

    m = max(h0, h1, h2)
    if m <= 1e-9:
        return 0.0
    return hd / m


def _build_from_theta(theta):
    """Convert log-probability parameters to a probability distribution."""
    # theta has shape (100,), representing log(p_i + eps) for each cell
    # Use softmax-like transformation for stability
    # Subtract max for numerical stability
    theta_shifted = theta - np.max(theta)
    p = np.exp(theta_shifted)
    s = p.sum()
    if s <= 1e-30:
        return np.ones(_N * _N) / (_N * _N)
    p = p / s
    return p


def _objective(theta):
    """Negative of the value we want to maximize."""
    p = _build_from_theta(theta)
    P = p.reshape(_N, _N)
    val = _compute_value(P)
    return -val


def _objective_and_grad(theta):
    """Objective and numerical gradient (fallback)."""
    p = _build_from_theta(theta)
    P = p.reshape(_N, _N)
    val = _compute_value(P)
    # Numerical gradient
    eps = 1e-7
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_plus[i] += eps
        p_plus = _build_from_theta(theta_plus)
        P_plus = p_plus.reshape(_N, _N)
        val_plus = _compute_value(P_plus)
        grad[i] = (val_plus - val) / eps
    return -val, -grad


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    
    Uses multiple random restarts with L-BFGS-B in log-probability space
    to maximize D/M where D = H_{x-y} and M = max(H_x, H_{x+y}, H_{x+2y}).
    
    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1.
    """
    deadline = time.time() + 550  # Safety margin before 600s limit
    n_cells = _N * _N
    
    best_P = np.ones((_N, _N), dtype=np.float64) / n_cells
    best_val = _compute_value(best_P)
    
    rng = np.random.RandomState(42)
    
    # Strategy 1: Start from uniform and optimize
    theta0 = np.zeros(n_cells)  # log-uniform
    result = minimize(
        _objective, theta0, method='L-BFGS-B',
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
    
    # Strategy 2: Multiple random restarts with different initializations
    n_restarts = 30
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Random initialization in log-space
        # Use various strategies for initialization
        
        if restart % 3 == 0:
            # Random sparse initialization
            theta = rng.randn(n_cells) * 2.0
        elif restart % 3 == 1:
            # Concentrate on diagonal (x == y) for low x-y entropy? No, we want HIGH x-y entropy
            # Actually we want x-y spread out, so concentrate where x-y varies
            # Try concentrating on anti-diagonal
            theta = np.full(n_cells, -10.0)
            for y in range(_N):
                for x in range(_N):
                    if x + y == 9:  # anti-diagonal
                        theta[y * _N + x] = 0.0
                    elif abs(x - y) >= 5:
                        theta[y * _N + x] = -2.0
        else:
            # Concentrate on corners
            theta = np.full(n_cells, -10.0)
            corners = [(0, 0), (0, 9), (9, 0), (9, 9)]
            for (cy, cx) in corners:
                theta[cy * _N + cx] = 0.0
            # Add some spread
            for y in range(_N):
                for x in range(_N):
                    if abs(x - y) >= 4:
                        theta[y * _N + x] = -3.0
        
        result = minimize(
            _objective, theta, method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        p_opt = _build_from_theta(result.x)
        P_opt = p_opt.reshape(_N, _N)
        val_opt = _compute_value(P_opt)
        if val_opt > best_val:
            best_val = val_opt
            best_P = P_opt.copy()
    
    # Strategy 3: Refine the best solution with more iterations
    if best_val > 0 and time.time() < deadline - 60:
        # Extract theta from best_P
        p_flat = best_P.ravel()
        # Convert to log-space (add small epsilon to avoid log(0))
        theta_best = np.log(p_flat + 1e-15)
        
        result = minimize(
            _objective, theta_best, method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        p_opt = _build_from_theta(result.x)
        P_opt = p_opt.reshape(_N, _N)
        val_opt = _compute_value(P_opt)
        if val_opt > best_val:
            best_val = val_opt
            best_P = P_opt.copy()
    
    # Strategy 4: Try structured distributions
    if time.time() < deadline - 30:
        # Try: put mass on cells where x-y is maximally spread
        # and x, x+y, x+2y are concentrated
        
        # Idea: concentrate on a few cells that have diverse x-y values
        # but similar x, x+y, x+2y values
        
        # Try: fix y, vary x -> x-y varies with x, but x also varies
        # Try: fix x, vary y -> x-y varies with y, x+y varies, x+2y varies
        
        # Better: find cells where x-y takes many different values
        # but x takes few different values (impossible since x-y and x are related)
        
        # Let's try: put mass on cells (x, y) where x is in a small set
        # and y is in a large set, but x-y is spread
        # If x is fixed, x-y = const - y, so x-y has same spread as y
        # And x+y = const + y, same spread
        # x+2y = const + 2y, same spread
        # So H_0 = 0 (if x fixed), H_1 = H_y, H_2 = H_y
        # D = H_y, M = H_y, ratio = 1. Not great.
        
        # Try: x takes 2 values, y takes 5 values
        # x-y: for each x, y gives 5 values, total up to 10 distinct values
        # x: 2 values, H_0 = 1 bit
        # x+y: up to 10 values
        # x+2y: up to 10 values
        
        # Let's just do a grid search over simple structured distributions
        # Try: P[y, x] = a * exp(-beta * (x-y)^2) * exp(-gamma * x^2)
        
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
    
    # Final cleanup: clip and normalize
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s <= 1e-9:
        best_P = np.ones((_N, _N), dtype=np.float64) / n_cells
    else:
        best_P /= s
    
    return best_P


# EVOLVE-BLOCK-END
