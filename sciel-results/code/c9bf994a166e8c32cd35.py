# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio optimization using full 10x10 parameterization
    with softmax and efficient projection entropy computation.
    """
    n = 10
    deadline = time.time() + 570  # Safety margin before 630s limit
    
    # Precompute projection value indices for each slope
    # For each slope r, map (y,x) -> index in the pooled distribution
    x_idx = np.arange(n)
    y_idx = np.arange(n)
    
    # proj_minus1: x - y, range [-9, 9], offset by 9 -> [0, 18]
    proj_m1_vals = (x_idx[None, :] - y_idx[:, None] + 9).astype(int)  # shape (10,10), values 0..18
    num_m1 = 19
    
    # proj_0: x, range [0, 9]
    proj_0_vals = x_idx[None, :].astype(int)  # shape (10,10), values 0..9
    num_0 = 10
    
    # proj_1: x + y, range [0, 18]
    proj_1_vals = (x_idx[None, :] + y_idx[:, None]).astype(int)  # shape (10,10), values 0..18
    num_1 = 19
    
    # proj_2: x + 2y, range [0, 27]
    proj_2_vals = (x_idx[None, :] + 2 * y_idx[:, None]).astype(int)  # shape (10,10), values 0..27
    num_2 = 28
    
    def compute_entropies(P):
        """Compute D and M from a 10x10 probability matrix P."""
        # Pool by projection values using bincount
        p_flat = P.ravel()
        
        # For each projection, sum P over cells with that projection value
        pool_m1 = np.bincount(proj_m1_vals.ravel(), weights=p_flat, minlength=num_m1)
        pool_0 = np.bincount(proj_0_vals.ravel(), weights=p_flat, minlength=num_0)
        pool_1 = np.bincount(proj_1_vals.ravel(), weights=p_flat, minlength=num_1)
        pool_2 = np.bincount(proj_2_vals.ravel(), weights=p_flat, minlength=num_2)
        
        def entropy(pool):
            p = pool[pool > 1e-15]
            if len(p) == 0:
                return 0.0
            return -np.sum(p * np.log2(p))
        
        D = entropy(pool_m1)
        H0 = entropy(pool_0)
        H1 = entropy(pool_1)
        H2 = entropy(pool_2)
        M = max(H0, H1, H2)
        
        if M <= 1e-9:
            return 0.0, 0.0  # Return 0 ratio
        return D, M
    
    def objective(params):
        """Objective function to minimize (negative of D/M)."""
        # Softmax parameterization for full 10x10
        P = np.exp(params - np.max(params))
        P = P / np.sum(P)
        
        D, M = compute_entropies(P)
        
        if M <= 1e-9:
            return 1e10
        return -(D / M)
    
    # Initial guess: uniform
    params0 = np.zeros(n * n)
    
    best_score = -np.inf
    best_params = params0.copy()
    
    # First try from uniform with L-BFGS-B
    result = minimize(objective, params0, method='L-BFGS-B',
                      options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
    if result.fun < best_score:
        best_score = result.fun
        best_params = result.x.copy()
    
    # Random restarts with different initializations
    rng = np.random.default_rng(42)
    n_restarts = 50
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Various random initializations
        if i % 5 == 0:
            # Small perturbation from uniform
            params_rand = rng.normal(0, 0.1, n * n)
        elif i % 5 == 1:
            # Moderate perturbation
            params_rand = rng.normal(0, 0.5, n * n)
        elif i % 5 == 2:
            # Large perturbation
            params_rand = rng.normal(0, 1.0, n * n)
        elif i % 5 == 3:
            # Sparse initialization
            params_rand = rng.normal(0, 0.3, n * n)
            mask = rng.random(n * n) < 0.5
            params_rand[mask] = rng.normal(0, 1.0, mask.sum())
        else:
            # Diagonal emphasis
            params_rand = np.zeros(n * n)
            for y in range(n):
                for x in range(n):
                    params_rand[y * n + x] = rng.normal(0, 0.5)
                    if x == y:
                        params_rand[y * n + x] += rng.normal(0, 1.0)
        
        result = minimize(objective, params_rand, method='L-BFGS-B',
                         options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 40})
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x.copy()
    
    # Extract best solution
    P = np.exp(best_params - np.max(best_params))
    P = P / np.sum(P)
    
    # Ensure validity
    P = np.clip(P, 0, None)
    total = np.sum(P)
    if total <= 1e-9:
        P = np.ones((n, n)) / (n * n)
    else:
        P = P / total
    
    return P


# EVOLVE-BLOCK-END
