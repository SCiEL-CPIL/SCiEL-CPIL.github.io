# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Uses product-of-marginals model: P[y,x] = R[y] * C[x] / (sum(R)*sum(C))
    This reduces the 100-dim search to 20 dimensions.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    # Pre-compute grid coordinates
    y_grid, x_grid = np.mgrid[0:n, 0:n]
    x_grid = x_grid.astype(np.float64)
    y_grid = y_grid.astype(np.float64)
    
    def _shannon_entropy(probs):
        """Compute Shannon entropy in bits, ignoring zero probabilities."""
        probs = probs[probs > 0]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))
    
    def compute_entropies_from_marginals(R, C):
        """
        Compute H_{-1}, H_0, H_1, H_2 from row marginal R (length n) and column marginal C (length n).
        P[y,x] = R[y] * C[x] (assumed normalized: sum(R)=1, sum(C)=1)
        """
        # H_0: distribution of X = C (since P[y,x] = R[y]*C[x], marginal of X is C)
        H_0 = _shannon_entropy(C)
        
        # H_{-1}: distribution of X - Y
        # For each value d = x - y, probability = sum_{x-y=d} R[y]*C[x]
        # d ranges from -(n-1) to n-1
        d_values = np.arange(-(n-1), n)  # -9 to 9
        prob_m1 = np.zeros(len(d_values))
        for i, d in enumerate(d_values):
            # x - y = d => x = y + d
            # y ranges from max(0, -d) to min(n-1, n-1-d)
            y_min = max(0, -d)
            y_max = min(n-1, n-1-d)
            if y_min <= y_max:
                ys = np.arange(y_min, y_max+1)
                xs = ys + d
                prob_m1[i] = np.sum(R[ys] * C[xs])
        H_m1 = _shannon_entropy(prob_m1)
        
        # H_1: distribution of X + Y
        # s = x + y, ranges from 0 to 2*(n-1)
        s_values = np.arange(0, 2*n-1)
        prob_1 = np.zeros(len(s_values))
        for i, s in enumerate(s_values):
            # x + y = s => x = s - y
            # y ranges from max(0, s-(n-1)) to min(n-1, s)
            y_min = max(0, s-(n-1))
            y_max = min(n-1, s)
            if y_min <= y_max:
                ys = np.arange(y_min, y_max+1)
                xs = s - ys
                prob_1[i] = np.sum(R[ys] * C[xs])
        H_1 = _shannon_entropy(prob_1)
        
        # H_2: distribution of X + 2Y
        # v = x + 2y, ranges from 0 to (n-1) + 2*(n-1) = 3*(n-1)
        v_max = 3*(n-1)
        prob_2 = np.zeros(v_max+1)
        for y in range(n):
            for x in range(n):
                v = x + 2*y
                prob_2[v] += R[y] * C[x]
        H_2 = _shannon_entropy(prob_2)
        
        return H_m1, H_0, H_1, H_2
    
    def compute_full_entropies(P):
        """Compute entropies from full 10x10 matrix P."""
        P_flat = P.ravel()
        
        # H_{-1}: x - y
        offset = n - 1
        idx_m1 = (x_grid - y_grid + offset).astype(np.int32).ravel()
        counts_m1 = np.bincount(idx_m1, weights=P_flat, minlength=2*n-1)
        H_m1 = _shannon_entropy(counts_m1)
        
        # H_0: x
        idx_0 = x_grid.astype(np.int32).ravel()
        counts_0 = np.bincount(idx_0, weights=P_flat, minlength=n)
        H_0 = _shannon_entropy(counts_0)
        
        # H_1: x + y
        idx_1 = (x_grid + y_grid).astype(np.int32).ravel()
        counts_1 = np.bincount(idx_1, weights=P_flat, minlength=2*n-1)
        H_1 = _shannon_entropy(counts_1)
        
        # H_2: x + 2y
        idx_2 = (x_grid + 2*y_grid).astype(np.int32).ravel()
        counts_2 = np.bincount(idx_2, weights=P_flat, minlength=3*n-1)
        H_2 = _shannon_entropy(counts_2)
        
        return H_m1, H_0, H_1, H_2
    
    def get_ratio_from_marginals(R, C):
        """Compute D/M ratio from marginals."""
        H_m1, H_0, H_1, H_2 = compute_entropies_from_marginals(R, C)
        M = max(H_0, H_1, H_2)
        if M <= 1e-9:
            return 0.0
        return H_m1 / M
    
    # Strategy: Coordinate descent on log-R and log-C parameters
    # Represent R and C via softmax of parameters to ensure positivity and sum=1
    
    def softmax(params):
        """Convert params to probability vector via softmax."""
        exp_params = np.exp(params - np.max(params))
        return exp_params / np.sum(exp_params)
    
    def objective_marginals(params):
        """
        Objective: minimize -ratio
        params: [log_R_0, ..., log_R_9, log_C_0, ..., log_C_9]
        """
        if time.time() > deadline - 10:
            return 1000.0
        
        log_R = params[:n]
        log_C = params[n:2*n]
        
        R = softmax(log_R)
        C = softmax(log_C)
        
        ratio = get_ratio_from_marginals(R, C)
        return -ratio
    
    # Initial guess: uniform marginals
    n_params = 2 * n
    uniform_params = np.zeros(n_params)
    
    # Try several initializations
    best_ratio = 0.0
    best_R = None
    best_C = None
    
    # Baseline: uniform
    R_uniform = np.ones(n) / n
    C_uniform = np.ones(n) / n
    ratio_uniform = get_ratio_from_marginals(R_uniform, C_uniform)
    best_ratio = ratio_uniform
    best_R = R_uniform.copy()
    best_C = C_uniform.copy()
    
    # Also try some structured initializations
    initializations = []
    
    # 1. Uniform
    initializations.append(np.zeros(n_params))
    
    # 2. Concentrated R on low y, C on high x (to make x-y large spread)
    R_conc = np.zeros(n)
    R_conc[:3] = 1.0/3
    C_conc = np.zeros(n)
    C_conc[7:] = 1.0/3
    initializations.append(np.concatenate([
        np.log(R_conc + 1e-10),
        np.log(C_conc + 1e-10)
    ]))
    
    # 3. Anti-diagonal concentration
    R_anti = np.zeros(n)
    R_anti[8:] = 1.0/2
    C_anti = np.zeros(n)
    C_anti[:2] = 1.0/2
    initializations.append(np.concatenate([
        np.log(R_anti + 1e-10),
        np.log(C_anti + 1e-10)
    ]))
    
    # 4. Bimodal
    R_bi = np.zeros(n)
    R_bi[0] = 0.5
    R_bi[9] = 0.5
    C_bi = np.zeros(n)
    C_bi[0] = 0.5
    C_bi[9] = 0.5
    initializations.append(np.concatenate([
        np.log(R_bi + 1e-10),
        np.log(C_bi + 1e-10)
    ]))
    
    # 5-15: Random initializations
    np.random.seed(42)
    for _ in range(10):
        R_rand = np.random.dirichlet(np.ones(n))
        C_rand = np.random.dirichlet(np.ones(n))
        initializations.append(np.concatenate([
            np.log(R_rand + 1e-10),
            np.log(C_rand + 1e-10)
        ]))
    
    # Coordinate descent with multiple restarts
    max_iters = 200
    tol = 1e-8
    
    for init_idx, p0 in enumerate(initializations):
        if time.time() > deadline - 30:
            break
        
        # Check initial point
        ratio_init = get_ratio_from_marginals(softmax(p0[:n]), softmax(p0[n:]))
        if ratio_init > best_ratio:
            best_ratio = ratio_init
            best_R = softmax(p0[:n]).copy()
            best_C = softmax(p0[n:]).copy()
        
        # Coordinate descent
        params = p0.copy()
        prev_ratio = -objective_marginals(params)
        
        for it in range(max_iters):
            if time.time() > deadline - 20:
                break
            
            improved = False
            
            # Update R parameters (first n)
            for i in range(n):
                if time.time() > deadline - 20:
                    break
                for step in [1e-4, 1e-3, 1e-2, 1e-1, 0.1, 0.5, 1.0, 2.0, 5.0]:
                    for direction in [1, -1]:
                        params_test = params.copy()
                        params_test[i] += direction * step
                        ratio_test = get_ratio_from_marginals(softmax(params_test[:n]), softmax(params_test[n:]))
                        if ratio_test > prev_ratio + 1e-10:
                            params = params_test
                            prev_ratio = ratio_test
                            improved = True
                            break
                    if improved:
                        break
            
            # Update C parameters (last n)
            for i in range(n, 2*n):
                if time.time() > deadline - 20:
                    break
                for step in [1e-4, 1e-3, 1e-2, 1e-1, 0.1, 0.5, 1.0, 2.0, 5.0]:
                    for direction in [1, -1]:
                        params_test = params.copy()
                        params_test[i] += direction * step
                        ratio_test = get_ratio_from_marginals(softmax(params_test[:n]), softmax(params_test[n:]))
                        if ratio_test > prev_ratio + 1e-10:
                            params = params_test
                            prev_ratio = ratio_test
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
            
            # Check convergence
            if abs(prev_ratio - (-objective_marginals(params))) < tol:
                break
        
        # Final check
        R_final = softmax(params[:n])
        C_final = softmax(params[n:])
        ratio_final = get_ratio_from_marginals(R_final, C_final)
        if ratio_final > best_ratio:
            best_ratio = ratio_final
            best_R = R_final.copy()
            best_C = C_final.copy()
    
    # Now try to refine using scipy minimize on the 20-dim parameter space
    if time.time() < deadline - 60:
        # Use the best found as starting point
        best_params = np.concatenate([
            np.log(best_R + 1e-10),
            np.log(best_C + 1e-10)
        ])
        
        # Run Nelder-Mead on the 20-dim space
        try:
            res = minimize(objective_marginals, best_params, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-7})
            if res.fun < -best_ratio:
                R_opt = softmax(res.x[:n])
                C_opt = softmax(res.x[n:])
                ratio_opt = get_ratio_from_marginals(R_opt, C_opt)
                if ratio_opt > best_ratio:
                    best_ratio = ratio_opt
                    best_R = R_opt.copy()
                    best_C = C_opt.copy()
        except Exception:
            pass
    
    # Construct final P from best marginals
    P = np.outer(best_C, best_R)  # P[y,x] = C[x] * R[y]
    
    # Validate and normalize
    P = np.clip(P, 0, None)
    s = np.sum(P)
    if s > 1e-9:
        P = P / s
    else:
        P = np.ones((n, n)) / (n*n)
    
    return P

# EVOLVE-BLOCK-END
