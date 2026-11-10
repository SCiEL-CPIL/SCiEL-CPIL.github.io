# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt
from scipy.special import logsumexp

def _shannon_entropy(p):
    """Shannon entropy in bits, ignoring zero probabilities."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return -np.sum(p * np.log2(p))

def _project_entropy(P, r):
    """Compute entropy of projection x + r*y for distribution P[y,x]."""
    n = P.shape[0]
    # P[y, x] where y is row, x is col
    # value = x + r*y
    # We need to pool by value
    xs = np.arange(n, dtype=np.float64)
    ys = np.arange(n, dtype=np.float64)
    values = xs[None, :] + r * ys[:, None]  # shape (n, n), values[y, x]
    
    # Get unique values and their total probability
    unique_vals, inverse = np.unique(values, return_inverse=True)
    # inverse has same shape as values
    total = np.zeros(len(unique_vals), dtype=np.float64)
    np.add.at(total, inverse.ravel(), P.ravel())
    
    return _shannon_entropy(total)

def _compute_entropies(P):
    """Compute D and M for distribution P."""
    D = _project_entropy(P, -1.0)
    H0 = _project_entropy(P, 0.0)
    H1 = _project_entropy(P, 1.0)
    H2 = _project_entropy(P, 2.0)
    M = max(H0, H1, H2)
    return D, M, H0, H1, H2

def _objective_and_grad(log_theta):
    """
    Objective: maximize D / M.
    Use log-softmax parameterization: P = exp(log_theta) / sum(exp(log_theta))
    To handle the ratio, we maximize D - log(M) approximately, or use D/M directly.
    
    For optimization, let's maximize D - alpha * M where alpha is a penalty,
    or better: use the fact that we want to maximize D/M.
    
    A smooth surrogate: maximize D - beta * logsumexp([H0, H1, H2] / beta)
    which approximates D - max(H0, H1, H2) as beta -> 0.
    
    Actually, let's just maximize D / M directly with gradient ascent on log_theta.
    """
    # Convert log_theta to probability
    log_theta = np.asarray(log_theta, dtype=np.float64)
    # Subtract max for numerical stability
    log_theta_shifted = log_theta - np.max(log_theta)
    P = np.exp(log_theta_shifted)
    P = P / np.sum(P)
    
    D, M, H0, H1, H2 = _compute_entropies(P)
    
    if M < 1e-9:
        return -1e10, None
    
    value = D / M
    return -value, None  # Negative because we minimize

def _soft_max_surrogate(log_theta, beta=0.1):
    """
    Use soft-max surrogate for M to get a smooth objective.
    Maximize D - (1/beta) * logsumexp([H0, H1, H2] * beta)
    which approximates D - max(H0, H1, H2)
    """
    log_theta = np.asarray(log_theta, dtype=np.float64)
    log_theta_shifted = log_theta - np.max(log_theta)
    P = np.exp(log_theta_shifted)
    P = P / np.sum(P)
    
    D, M, H0, H1, H2 = _compute_entropies(P)
    
    # Soft max of H0, H1, H2
    H_vals = np.array([H0, H1, H2])
    soft_M = (1.0 / beta) * logsumexp(H_vals * beta)
    
    # We want to maximize D - soft_M
    objective = D - soft_M
    return -objective, None

def _refine_P(P_init, max_iters=5000, tol=1e-8):
    """
    Refine P using projected gradient ascent on the simplex.
    """
    P = P_init.copy().astype(np.float64)
    P = np.clip(P, 0, None)
    P = P / P.sum()
    
    best_P = P.copy()
    best_val = _compute_entropies(best_P)[0] / max(_compute_entropies(best_P)[1], 1e-9)
    
    lr = 0.01
    for it in range(max_iters):
        D, M, H0, H1, H2 = _compute_entropies(P)
        if M < 1e-9:
            break
        value = D / M
        if value > best_val:
            best_val = value
            best_P = P.copy()
        
        # Compute gradient numerically or analytically
        # For now, use a simple approach: try random perturbations
        # This is slow, so let's use scipy instead
        break
    
    return best_P

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    deadline = start_time + 600  # 600 second deadline with safety margin
    
    # Start with uniform distribution
    P_uniform = np.ones((n, n), dtype=np.float64) / (n * n)
    best_P = P_uniform.copy()
    best_value = _compute_entropies(best_P)[0] / max(_compute_entropies(best_P)[1], 1e-9)
    
    # Try various initializations and optimize
    initializations = []
    
    # 1. Uniform
    initializations.append(P_uniform.copy())
    
    # 2. Concentrate X at center, Y uniform
    for cx in range(n):
        P = np.zeros((n, n), dtype=np.float64)
        P[:, cx] = 1.0 / n
        initializations.append(P)
    
    # 3. Concentrate Y at center, X uniform
    for cy in range(n):
        P = np.zeros((n, n), dtype=np.float64)
        P[cy, :] = 1.0 / n
        initializations.append(P)
    
    # 4. Diagonal concentration
    for diag in range(-n+1, n):
        P = np.zeros((n, n), dtype=np.float64)
        for y in range(n):
            x = y + diag
            if 0 <= x < n:
                P[y, x] = 1.0
        if P.sum() > 0:
            P = P / P.sum()
            initializations.append(P)
    
    # 5. Random distributions
    np.random.seed(42)
    for _ in range(20):
        P = np.random.dirichlet(np.ones(n * n))
        P = P.reshape(n, n)
        initializations.append(P)
    
    # 6. Bimodal Y
    for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        P = np.zeros((n, n), dtype=np.float64)
        # Y=0 with prob p, Y=9 with prob 1-p, X uniform
        P[0, :] = p / n
        P[9, :] = (1 - p) / n
        initializations.append(P)
    
    # Optimize each initialization
    for P_init in initializations:
        if time.time() > deadline - 30:
            break
        
        # Use scipy minimize on log-softmax parameterization
        P_init = np.clip(P_init, 1e-10, None)
        P_init = P_init / P_init.sum()
        log_theta = np.log(P_init).ravel()
        
        try:
            result = opt.minimize(
                _soft_max_surrogate,
                log_theta,
                args=(0.1,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            
            if result.success or result.fun < -best_value:
                log_theta_opt = result.x
                log_theta_shifted = log_theta_opt - np.max(log_theta_opt)
                P_opt = np.exp(log_theta_shifted).reshape(n, n)
                P_opt = P_opt / P_opt.sum()
                
                D, M, _, _, _ = _compute_entropies(P_opt)
                if M > 1e-9:
                    val = D / M
                    if val > best_value:
                        best_value = val
                        best_P = P_opt.copy()
        except Exception:
            pass
    
    # Final polish: try to improve best_P with small perturbations
    if time.time() < deadline - 10:
        # Try local search with small steps
        P = best_P.copy()
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            # Random perturbation
            delta = np.random.randn(n, n) * 0.01
            P_new = P + delta
            P_new = np.clip(P_new, 0, None)
            if P_new.sum() < 1e-9:
                continue
            P_new = P_new / P_new.sum()
            
            D, M, _, _, _ = _compute_entropies(P_new)
            if M > 1e-9:
                val = D / M
                if val > best_value:
                    best_value = val
                    best_P = P_new.copy()
    
    # Ensure validity
    best_P = np.clip(best_P, 0, None)
    if best_P.sum() < 1e-9:
        best_P = P_uniform
    else:
        best_P = best_P / best_P.sum()
    
    return best_P

# EVOLVE-BLOCK-END
