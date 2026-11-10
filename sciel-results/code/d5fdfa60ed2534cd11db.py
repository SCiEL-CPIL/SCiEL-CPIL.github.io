# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp


def _shannon_entropy_bits(probs):
    """Shannon entropy in bits, ignoring zero probabilities."""
    probs = np.asarray(probs, dtype=np.float64)
    mask = probs > 1e-15
    if not np.any(mask):
        return 0.0
    p = probs[mask]
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p)))


def _compute_ratio(P):
    """Compute the ratio D/M for a given 10x10 probability matrix P."""
    n = P.shape[0]
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    # H_{-1}: distribution of x - y
    vals_minus1 = X - Y
    unique_vals = np.unique(vals_minus1)
    probs_minus1 = np.array([P[vals_minus1 == v].sum() for v in unique_vals])
    H_minus1 = _shannon_entropy_bits(probs_minus1)
    
    # H_0: distribution of x
    probs_0 = P.sum(axis=0)
    H_0 = _shannon_entropy_bits(probs_0)
    
    # H_1: distribution of x + y
    vals_1 = X + Y
    unique_vals_1 = np.unique(vals_1)
    probs_1 = np.array([P[vals_1 == v].sum() for v in unique_vals_1])
    H_1 = _shannon_entropy_bits(probs_1)
    
    # H_2: distribution of x + 2y
    vals_2 = X + 2 * Y
    unique_vals_2 = np.unique(vals_2)
    probs_2 = np.array([P[vals_2 == v].sum() for v in unique_vals_2])
    H_2 = _shannon_entropy_bits(probs_2)
    
    M = max(H_0, H_1, H_2)
    if M < 1e-9:
        return 0.0
    return H_minus1 / M


def _objective_product(params, n=10):
    """
    Objective for product-form P[y,x] = R[y] * C[x].
    params: [log_R_0, ..., log_R_{n-1}, log_C_0, ..., log_C_{n-1}]
    We normalize so that sum(R)*sum(C) = 1.
    """
    log_R = params[:n]
    log_C = params[n:2*n]
    
    # Use softmax-like normalization to ensure positivity and sum to 1
    # R = exp(log_R) / sum(exp(log_R)), C = exp(log_C) / sum(exp(log_C))
    # Then P = outer(R, C) has sum = 1 automatically
    
    # Subtract max for numerical stability
    log_R_shifted = log_R - np.max(log_R)
    log_C_shifted = log_C - np.max(log_C)
    
    R = np.exp(log_R_shifted)
    C = np.exp(log_C_shifted)
    
    R = R / R.sum()
    C = C / C.sum()
    
    P = np.outer(R, C)  # P[y, x] = R[y] * C[x]
    
    ratio = _compute_ratio(P)
    return -ratio


def _objective_general(params, n=10):
    """
    Objective for general P, parameterized by log-weights.
    params: log(P_{y,x} + epsilon) for all y, x
    """
    log_P = params.reshape(n, n)
    log_P_shifted = log_P - np.max(log_P)
    P = np.exp(log_P_shifted)
    s = P.sum()
    if s < 1e-15:
        return 1e6
    P = P / s
    
    ratio = _compute_ratio(P)
    return -ratio


def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    deadline = time.time() + 600 - 30  # safety margin
    
    # Start with uniform distribution
    P0 = np.ones((n, n), dtype=np.float64) / (n * n)
    best_P = P0.copy()
    best_ratio = _compute_ratio(P0)
    
    np.random.seed(42)
    
    # Strategy 1: Product-form optimization P[y,x] = R[y] * C[x]
    # This reduces the search space from 100 to 20 dimensions
    for trial in range(50):
        if time.time() > deadline - 60:
            break
        
        # Random initialization in log-space
        log_R = np.random.randn(n) * 0.5
        log_C = np.random.randn(n) * 0.5
        params = np.concatenate([log_R, log_C])
        
        try:
            result = minimize(
                _objective_product,
                params,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100},
                args=(n,)
            )
            
            if result.fun < -best_ratio:
                log_R_opt = result.x[:n]
                log_C_opt = result.x[n:2*n]
                
                log_R_shifted = log_R_opt - np.max(log_R_opt)
                log_C_shifted = log_C_opt - np.max(log_C_opt)
                
                R = np.exp(log_R_shifted)
                C = np.exp(log_C_shifted)
                R = R / R.sum()
                C = C / C.sum()
                
                P_opt = np.outer(R, C)
                ratio_opt = _compute_ratio(P_opt)
                if ratio_opt > best_ratio:
                    best_ratio = ratio_opt
                    best_P = P_opt.copy()
        except Exception:
            pass
    
    # Strategy 2: General optimization with log-parameterization
    # Start from the best product-form solution
    log_P_init = np.log(best_P + 1e-15)
    
    for trial in range(30):
        if time.time() > deadline - 30:
            break
        
        if trial == 0:
            params = log_P_init.flatten().copy()
        else:
            # Perturb the best solution
            params = log_P_init.flatten() + np.random.randn(n*n) * 0.1
        
        try:
            result = minimize(
                _objective_general,
                params,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100},
                args=(n,)
            )
            
            if result.fun < -best_ratio:
                log_P_opt = result.x.reshape(n, n)
                log_P_shifted = log_P_opt - np.max(log_P_opt)
                P_opt = np.exp(log_P_shifted)
                s_opt = P_opt.sum()
                if s_opt > 1e-15:
                    P_opt = P_opt / s_opt
                    ratio_opt = _compute_ratio(P_opt)
                    if ratio_opt > best_ratio:
                        best_ratio = ratio_opt
                        best_P = P_opt.copy()
                        log_P_init = np.log(best_P + 1e-15)
        except Exception:
            pass
    
    # Strategy 3: Local perturbations on the best solution
    if time.time() < deadline - 10:
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            
            # Different perturbation scales
            scale = np.random.choice([0.01, 0.05, 0.1, 0.2])
            P_pert = best_P + np.random.randn(n, n) * scale
            P_pert = np.clip(P_pert, 1e-15, None)
            s_pert = P_pert.sum()
            if s_pert > 1e-12:
                P_pert = P_pert / s_pert
                ratio_pert = _compute_ratio(P_pert)
                if ratio_pert > best_ratio:
                    best_ratio = ratio_pert
                    best_P = P_pert.copy()
    
    # Ensure validity
    best_P = np.clip(best_P, 0, None)
    s_final = best_P.sum()
    if s_final < 1e-9:
        best_P = P0
    else:
        best_P = best_P / s_final
    
    return best_P


# EVOLVE-BLOCK-END
