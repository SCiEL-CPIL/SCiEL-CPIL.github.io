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


def _objective(params, n=10):
    """
    Objective to maximize: H_{-1} / max(H_0, H_1, H_2)
    We minimize the negative of this ratio.
    
    params: flattened U (n x k) and V (n x k) such that P = U @ V.T
    """
    k = 5  # rank
    U = params[:n * k].reshape(n, k)
    V = params[n * k:].reshape(n, k)
    
    # Compute P = U @ V.T, clip to non-negative, normalize
    P = U @ V.T
    P = np.clip(P, 0, None)
    s = P.sum()
    if s < 1e-12:
        return 1e6  # large penalty
    P = P / s
    
    # Precompute indices
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='xy')  # X[i,j] = i, Y[i,j] = j
    # P[y, x] so P[Y, X]
    
    # H_{-1}: distribution of x - y
    vals_minus1 = X - Y  # shape (n, n)
    unique_vals = np.unique(vals_minus1)
    probs_minus1 = np.array([P[vals_minus1 == v].sum() for v in unique_vals])
    H_minus1 = _shannon_entropy_bits(probs_minus1)
    
    # H_0: distribution of x
    probs_0 = P.sum(axis=0)  # sum over y
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
        return 1e6
    
    ratio = H_minus1 / M
    return -ratio


def _gradient(params, n=10, k=5, eps=1e-7):
    """Numerical gradient."""
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        params_minus = params.copy()
        params_minus[i] -= eps
        grad[i] = (_objective(params_plus, n, k) - _objective(params_minus, n, k)) / (2 * eps)
    return grad


def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    k = 5  # rank
    deadline = time.time() + 600 - 30  # safety margin
    
    # Start with uniform distribution
    P0 = np.ones((n, n), dtype=np.float64) / (n * n)
    
    # Initialize low-rank factorization
    # Use SVD of P0 as starting point
    U0, S0, Vt0 = np.linalg.svd(P0, full_matrices=False)
    # Truncate to rank k
    U0 = U0[:, :k] * np.sqrt(S0[:k])
    V0 = Vt0[:k, :].T * np.sqrt(S0[:k])
    
    params0 = np.concatenate([U0.flatten(), V0.flatten()])
    
    # Compute baseline score
    P_init = np.clip(U0 @ V0.T, 0, None)
    s = P_init.sum()
    if s > 0:
        P_init = P_init / s
    else:
        P_init = P0
    
    # Helper to compute the ratio for a given P
    def compute_ratio(P):
        x = np.arange(n)
        y = np.arange(n)
        X, Y = np.meshgrid(x, y, indexing='xy')
        
        vals_minus1 = X - Y
        unique_vals = np.unique(vals_minus1)
        probs_minus1 = np.array([P[vals_minus1 == v].sum() for v in unique_vals])
        H_minus1 = _shannon_entropy_bits(probs_minus1)
        
        probs_0 = P.sum(axis=0)
        H_0 = _shannon_entropy_bits(probs_0)
        
        vals_1 = X + Y
        unique_vals_1 = np.unique(vals_1)
        probs_1 = np.array([P[vals_1 == v].sum() for v in unique_vals_1])
        H_1 = _shannon_entropy_bits(probs_1)
        
        vals_2 = X + 2 * Y
        unique_vals_2 = np.unique(vals_2)
        probs_2 = np.array([P[vals_2 == v].sum() for v in unique_vals_2])
        H_2 = _shannon_entropy_bits(probs_2)
        
        M = max(H_0, H_1, H_2)
        if M < 1e-9:
            return 0.0
        return H_minus1 / M
    
    best_P = P_init.copy()
    best_ratio = compute_ratio(P_init)
    
    # Try multiple random initializations and optimize
    np.random.seed(42)
    
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Random initialization
        U_rand = np.random.randn(n, k)
        V_rand = np.random.randn(n, k)
        params = np.concatenate([U_rand.flatten(), V_rand.flatten()])
        
        # Also try perturbation of initial
        if trial == 0:
            params = params0.copy()
        
        # Use Nelder-Mead or L-BFGS-B
        try:
            result = minimize(
                _objective,
                params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50},
                args=(n, k)
            )
            
            if result.success or result.fun < -best_ratio:
                U_opt = result.x[:n * k].reshape(n, k)
                V_opt = result.x[n * k:].reshape(n, k)
                P_opt = np.clip(U_opt @ V_opt.T, 0, None)
                s_opt = P_opt.sum()
                if s_opt > 1e-12:
                    P_opt = P_opt / s_opt
                    ratio_opt = compute_ratio(P_opt)
                    if ratio_opt > best_ratio:
                        best_ratio = ratio_opt
                        best_P = P_opt.copy()
        except Exception:
            pass
    
    # Final polish: local search on the best P found
    if time.time() < deadline - 10:
        # Try small perturbations
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            P_pert = best_P + np.random.randn(n, n) * 0.01
            P_pert = np.clip(P_pert, 0, None)
            s_pert = P_pert.sum()
            if s_pert > 1e-12:
                P_pert = P_pert / s_pert
                ratio_pert = compute_ratio(P_pert)
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
