# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio optimization on 10x10 grid.
    Maximizes H(x-y) / max(H(x), H(x+y), x+2y).
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second deadline
    
    # Precompute projection indices and masks for fast entropy computation
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='xy')  # X[y,x] = x, Y[y,x] = y
    
    # x - y ranges from -9 to 9 (19 values)
    proj_minus1 = X - Y
    # x ranges from 0 to 9 (10 values)
    proj_0 = X
    # x + y ranges from 0 to 18 (19 values)
    proj_1 = X + Y
    # x + 2y ranges from 0 to 27 (28 values)
    proj_2 = X + 2 * Y
    
    # Precompute index arrays for each projection value
    # For proj_minus1: value v corresponds to index v+9 (0..18)
    idx_minus1 = proj_minus1 + 9  # 0..18
    # For proj_0: value is 0..9
    idx_0 = proj_0  # 0..9
    # For proj_1: value is 0..18
    idx_1 = proj_1  # 0..18
    # For proj_2: value is 0..27
    idx_2 = proj_2  # 0..27
    
    def compute_entropy_fast(idx_array, P, num_vals):
        """Compute Shannon entropy in bits using precomputed indices."""
        prob = np.zeros(num_vals)
        np.add.at(prob, idx_array.ravel(), P.ravel())
        prob = prob[prob > 1e-15]
        if len(prob) == 0:
            return 0.0
        return -np.sum(prob * np.log2(prob))
    
    def compute_ratio_from_logw(log_w):
        """Compute D/M ratio from log-weights."""
        log_w = log_w.reshape((n, n))
        log_w -= np.max(log_w)
        w = np.exp(log_w)
        P = w / np.sum(w)
        
        D = compute_entropy_fast(idx_minus1, P, 19)
        H0 = compute_entropy_fast(idx_0, P, 10)
        H1 = compute_entropy_fast(idx_1, P, 19)
        H2 = compute_entropy_fast(idx_2, P, 28)
        
        M = max(H0, H1, H2)
        if M < 1e-9:
            return 0.0, P
        return D / M, P
    
    def objective(params, use_smooth_max=False, beta=10.0):
        """Negative ratio to minimize (we want to maximize ratio)."""
        ratio, P = compute_ratio_from_logw(params)
        if ratio == 0.0:
            return 1e9
        return -ratio
    
    def objective_smooth(params, beta=10.0):
        """
        Smooth surrogate using log-sum-exp for the max function.
        This makes the objective smoother for gradient-based methods.
        """
        log_w = params.reshape((n, n))
        log_w -= np.max(log_w)
        w = np.exp(log_w)
        P = w / np.sum(w)
        
        D = compute_entropy_fast(idx_minus1, P, 19)
        H0 = compute_entropy_fast(idx_0, P, 10)
        H1 = compute_entropy_fast(idx_1, P, 19)
        H2 = compute_entropy_fast(idx_2, P, 28)
        
        # Smooth max using log-sum-exp: max(a,b,c) ≈ (1/beta)*log(exp(beta*a)+exp(beta*b)+exp(beta*c))
        h_vals = np.array([H0, H1, H2])
        if np.max(h_vals) < 1e-9:
            return 1e9
        log_sum_exp = np.log(np.sum(np.exp(beta * h_vals))) / beta
        M_smooth = log_sum_exp
        
        if M_smooth < 1e-9:
            return 1e9
        
        ratio = D / M_smooth
        return -ratio
    
    # Initialize with uniform distribution
    best_params = np.zeros(n * n)
    best_score = 0.0
    best_P = np.ones((n, n)) / (n * n)
    
    def update_best(params):
        nonlocal best_params, best_score, best_P
        ratio, P = compute_ratio_from_logw(params)
        if ratio > best_score:
            best_score = ratio
            best_params = params.copy()
            best_P = P.copy()
    
    # Strategy 1: Try multiple structured initializations with Nelder-Mead
    initializations = []
    
    # Uniform
    initializations.append(np.zeros(n * n))
    
    # Diagonal weighting to flatten x-y
    init_diag = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = i - j
            init_diag[i, j] = 1.0 / (10 - abs(d))
    init_diag = init_diag - np.mean(init_diag)
    initializations.append(init_diag.ravel())
    
    # Random initializations with different scales
    np.random.seed(42)
    for scale in [0.5, 1.0, 2.0, 3.0]:
        initializations.append(np.random.randn(n * n) * scale)
    
    # Concentrate on specific patterns
    # Diagonal x=y
    init_diag2 = np.zeros((n, n))
    for i in range(n):
        init_diag2[i, i] = 1.0
    initializations.append(init_diag2.ravel())
    
    # Anti-diagonal
    init_anti = np.zeros((n, n))
    for i in range(n):
        init_anti[i, n-1-i] = 1.0
    initializations.append(init_anti.ravel())
    
    # Corners
    init_corner = np.zeros((n, n))
    init_corner[0, 0] = 1.0
    init_corner[0, -1] = 1.0
    init_corner[-1, 0] = 1.0
    init_corner[-1, -1] = 1.0
    initializations.append(init_corner.ravel())
    
    # Weight toward center
    init_center = np.zeros((n, n))
    cx, cy = 4.5, 4.5
    for i in range(n):
        for j in range(n):
            init_center[i, j] = np.exp(-((i-cy)**2 + (j-cx)**2) / 10.0)
    initializations.append(init_center.ravel())
    
    # Run Nelder-Mead from each initialization
    for idx, init in enumerate(initializations):
        if time.time() > deadline - 60:
            break
        
        try:
            result = minimize(objective, init, method='Nelder-Mead',
                             options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10,
                                     'adaptive': True})
            update_best(result.x)
        except Exception:
            pass
    
    # Strategy 2: Use smooth surrogate with L-BFGS-B for gradient-based refinement
    if time.time() < deadline - 90:
        # Start from current best
        try:
            result = minimize(objective_smooth, best_params, method='L-BFGS-B',
                             options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10})
            update_best(result.x)
        except Exception:
            pass
        
        # Also try from a few random starts with smooth objective
        for _ in range(5):
            if time.time() > deadline - 60:
                break
            init = np.random.randn(n * n) * 1.0
            try:
                result = minimize(objective_smooth, init, method='L-BFGS-B',
                                 options={'maxiter': 3000, 'ftol': 1e-12})
                update_best(result.x)
            except Exception:
                pass
    
    # Strategy 3: Final high-precision polish
    if time.time() < deadline - 30 and best_score > 0:
        try:
            result = minimize(objective, best_params, method='Nelder-Mead',
                             options={'maxiter': 50000, 'xatol': 1e-12, 'fatol': 1e-12,
                                     'adaptive': True})
            update_best(result.x)
        except Exception:
            pass
        
        # One more L-BFGS-B pass with very tight tolerance
        if time.time() < deadline - 15:
            try:
                result = minimize(objective, best_params, method='L-BFGS-B',
                                 options={'maxiter': 10000, 'ftol': 1e-14, 'gtol': 1e-12})
                update_best(result.x)
            except Exception:
                pass
    
    # Ensure valid output
    best_P = np.clip(best_P, 0, None)
    total = np.sum(best_P)
    if total <= 1e-9:
        best_P = np.ones((n, n)) / (n * n)
    else:
        best_P = best_P / total
    
    return best_P


# EVOLVE-BLOCK-END
