# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Pre-compute grid coordinates
    y_grid, x_grid = np.mgrid[0:n, 0:n]
    x_grid = x_grid.astype(np.float64)
    y_grid = y_grid.astype(np.float64)
    
    # Indices for bincount
    idx_m1 = (x_grid - y_grid + n - 1).astype(np.int32).ravel()
    idx_0 = x_grid.astype(np.int32).ravel()
    idx_1 = (x_grid + y_grid).astype(np.int32).ravel()
    idx_2 = (x_grid + 2 * y_grid).astype(np.int32).ravel()
    
    min_len_m1 = 2 * n - 1
    min_len_0 = n
    min_len_1 = 2 * n - 1
    min_len_2 = 3 * n - 1

    def _shannon_entropy(probs):
        probs = probs[probs > 0]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))

    def compute_entropies(P_flat):
        """
        Compute H_m1, H_0, H_1, H_2 from flattened P.
        Also returns the bin probabilities for gradient calculation if needed, 
        but here we just return entropies for the objective.
        """
        counts_m1 = np.bincount(idx_m1, weights=P_flat, minlength=min_len_m1)
        H_m1 = _shannon_entropy(counts_m1)
        
        counts_0 = np.bincount(idx_0, weights=P_flat, minlength=min_len_0)
        H_0 = _shannon_entropy(counts_0)
        
        counts_1 = np.bincount(idx_1, weights=P_flat, minlength=min_len_1)
        H_1 = _shannon_entropy(counts_1)
        
        counts_2 = np.bincount(idx_2, weights=P_flat, minlength=min_len_2)
        H_2 = _shannon_entropy(counts_2)
        
        return H_m1, H_0, H_1, H_2

    def get_ratio(P_flat):
        H_m1, H_0, H_1, H_2 = compute_entropies(P_flat)
        M = max(H_0, H_1, H_2)
        if M <= 1e-9:
            return 0.0
        return H_m1 / M

    # Phase 1: Fast search on product of marginals (20 dims)
    def compute_entropies_from_marginals(R, C):
        P_flat = np.outer(C, R).ravel()
        return compute_entropies(P_flat)

    def get_ratio_from_marginals(R, C):
        H_m1, H_0, H_1, H_2 = compute_entropies_from_marginals(R, C)
        M = max(H_0, H_1, H_2)
        if M <= 1e-9:
            return 0.0
        return H_m1 / M

    def softmax(params):
        exp_params = np.exp(params - np.max(params))
        return exp_params / np.sum(exp_params)

    # Initial uniform
    R_init = np.ones(n) / n
    C_init = np.ones(n) / n
    best_ratio = get_ratio_from_marginals(R_init, C_init)
    best_R = R_init.copy()
    best_C = C_init.copy()
    
    # Simple coordinate descent on marginals
    # Parameters: log_R, log_C
    log_R = np.log(R_init + 1e-10)
    log_C = np.log(C_init + 1e-10)
    params = np.concatenate([log_R, log_C])
    
    max_iters = 50
    for it in range(max_iters):
        if time.time() > deadline - 300:
            break
        improved = False
        # Update R
        for i in range(n):
            for step in [0.5, 0.1, 0.01]:
                for dir in [1, -1]:
                    p_test = params.copy()
                    p_test[i] += dir * step
                    R_t = softmax(p_test[:n])
                    C_t = softmax(p_test[n:])
                    r_t = get_ratio_from_marginals(R_t, C_t)
                    if r_t > best_ratio + 1e-8:
                        params = p_test
                        best_ratio = r_t
                        best_R = R_t.copy()
                        best_C = C_t.copy()
                        improved = True
                        break
                if improved:
                    break
        # Update C
        for i in range(n, 2*n):
            for step in [0.5, 0.1, 0.01]:
                for dir in [1, -1]:
                    p_test = params.copy()
                    p_test[i] += dir * step
                    R_t = softmax(p_test[:n])
                    C_t = softmax(p_test[n:])
                    r_t = get_ratio_from_marginals(R_t, C_t)
                    if r_t > best_ratio + 1e-8:
                        params = p_test
                        best_ratio = r_t
                        best_R = R_t.copy()
                        best_C = C_t.copy()
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            break

    # Construct initial P_full from best marginals
    P_init = np.outer(best_C, best_R)
    P_init = P_init / np.sum(P_init)
    P_init_flat = P_init.ravel()
    
    # Phase 2: Refine on full 100-dim space using Gradient Descent (BFGS)
    # Parameterize P via logits Z (100 dims)
    # P = softmax(Z)
    
    def objective_and_grad(Z):
        if time.time() > deadline - 10:
            return 1e10, np.zeros_like(Z)
        
        Z_max = np.max(Z)
        exp_Z = np.exp(Z - Z_max)
        P = exp_Z / np.sum(exp_Z)
        P_flat = P.ravel()
        
        H_m1, H_0, H_1, H_2 = compute_entropies(P_flat)
        
        # Soft max for M to ensure differentiability
        # M_soft = tau * log(sum(exp(H_i / tau)))
        # We want to approximate max(H_0, H_1, H_2)
        # As tau -> 0, M_soft -> max(H_i)
        # Let's use a fixed small tau, e.g., 0.1
        tau = 0.1
        H_vec = np.array([H_0, H_1, H_2])
        H_max = np.max(H_vec)
        exp_H = np.exp((H_vec - H_max) / tau)
        M_soft = H_max + tau * np.log(np.sum(exp_H))
        
        if M_soft <= 1e-9:
            return 1e10, np.zeros_like(Z)
            
        ratio = H_m1 / M_soft
        obj = -ratio
        
        # Gradient calculation
        # grad_Z obj = - grad_Z (H_m1 / M_soft)
        # = - (M_soft * grad_Z H_m1 - H_m1 * grad_Z M_soft) / M_soft^2
        
        # Helper: grad_Z H_r
        # H_r = -1/ln2 sum_v p_v ln p_v
        # d H_r / d P_{y,x} = -1/ln2 (ln p_{v(y,x)} + 1)
        # Let G_r(y,x) be this gradient w.r.t P_{y,x}
        
        # Compute p_v for each projection
        counts_m1 = np.bincount(idx_m1, weights=P_flat, minlength=min_len_m1)
        counts_0 = np.bincount(idx_0, weights=P_flat, minlength=min_len_0)
        counts_1 = np.bincount(idx_1, weights=P_flat, minlength=min_len_1)
        counts_2 = np.bincount(idx_2, weights=P_flat, minlength=min_len_2)
        
        # Avoid log(0)
        eps = 1e-15
        p_m1_safe = np.maximum(counts_m1, eps)
        p_0_safe = np.maximum(counts_0, eps)
        p_1_safe = np.maximum(counts_1, eps)
        p_2_safe = np.maximum(counts_2, eps)
        
        # Gradient w.r.t P for H_m1
        # For each cell (y,x), it contributes to bin idx_m1[y,x]
        # grad_P_m1[y,x] = -1/ln2 (ln(counts_m1[idx] + 1) ... wait, p_v is the sum)
        # d p_v / d P_{y,x} = 1 if idx==v
        # d H / d P_{y,x} = -1/ln2 (ln p_{v(y,x)} + 1)
        
        G_m1_flat = -1.0 / np.log(2.0) * (np.log(p_m1_safe[idx_m1]) + 1.0)
        G_0_flat = -1.0 / np.log(2.0) * (np.log(p_0_safe[idx_0]) + 1.0)
        G_1_flat = -1.0 / np.log(2.0) * (np.log(p_1_safe[idx_1]) + 1.0)
        G_2_flat = -1.0 / np.log(2.0) * (np.log(p_2_safe[idx_2]) + 1.0)
        
        # Gradient w.r.t Z for H_r
        # grad_Z H_r = P * (G_r - sum(P * G_r))
        # This formula holds for softmax parameterization
        
        grad_Z_H_m1 = P_flat * (G_m1_flat - np.sum(P_flat * G_m1_flat))
        
        # Gradient for M_soft
        # M_soft = H_max + tau * log(sum(exp((H_i - H_max)/tau)))
        # d M_soft / d H_i = exp((H_i - H_max)/tau) / sum(exp((H_j - H_max)/tau))
        # Let w_i be these weights
        w = exp_H / np.sum(exp_H)
        
        # grad_Z M_soft = w[0] * grad_Z H_0 + w[1] * grad_Z H_1 + w[2] * grad_Z H_2
        grad_Z_H_0 = P_flat * (G_0_flat - np.sum(P_flat * G_0_flat))
        grad_Z_H_1 = P_flat * (G_1_flat - np.sum(P_flat * G_1_flat))
        grad_Z_H_2 = P_flat * (G_2_flat - np.sum(P_flat * G_2_flat))
        
        grad_Z_M_soft = w[0] * grad_Z_H_0 + w[1] * grad_Z_H_1 + w[2] * grad_Z_H_2
        
        # Combine
        grad_Z_obj = - (M_soft * grad_Z_H_m1 - H_m1 * grad_Z_M_soft) / (M_soft * M_soft)
        
        return obj, grad_Z_obj.reshape(n, n)

    # Initial Z from P_init
    Z_init = np.log(P_init + 1e-10)
    
    # Optimize
    best_P = P_init.copy()
    best_ratio_full = get_ratio(P_init_flat)
    
    try:
        res = minimize(objective_and_grad, Z_init, method='L-BFGS-B',
                       jac=True,
                       options={'maxiter': 200, 'maxls': 50, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 500})
        
        if res.fun < -best_ratio_full:
            Z_opt = res.x
            Z_max = np.max(Z_opt)
            P_opt = np.exp(Z_opt - Z_max)
            P_opt = P_opt / np.sum(P_opt)
            ratio_opt = get_ratio(P_opt.ravel())
            if ratio_opt > best_ratio_full:
                best_P = P_opt
                best_ratio_full = ratio_opt
    except Exception as e:
        pass

    # If we have time, try a few random restarts with Nelder-Mead on full space
    if time.time() < deadline - 60:
        np.random.seed(42)
        for _ in range(5):
            if time.time() > deadline - 30:
                break
            # Random perturbation of best_P
            noise = np.random.normal(0, 0.1, (n, n))
            P_pert = best_P + noise
            P_pert = np.clip(P_pert, 1e-6, None)
            P_pert = P_pert / np.sum(P_pert)
            Z_pert = np.log(P_pert)
            
            try:
                res2 = minimize(lambda Z: -get_ratio(np.exp(Z - np.max(Z)).ravel()), Z_pert,
                                method='Nelder-Mead',
                                options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-6})
                if res2.fun < -best_ratio_full:
                    Z_opt2 = res2.x
                    P_opt2 = np.exp(Z_opt2 - np.max(Z_opt2))
                    P_opt2 = P_opt2 / np.sum(P_opt2)
                    ratio_opt2 = get_ratio(P_opt2.ravel())
                    if ratio_opt2 > best_ratio_full:
                        best_P = P_opt2
                        best_ratio_full = ratio_opt2
            except Exception:
                pass

    # Final validation and normalization
    P = best_P
    P = np.clip(P, 0, None)
    s = np.sum(P)
    if s > 1e-9:
        P = P / s
    else:
        P = np.ones((n, n)) / (n*n)
        
    return P

# EVOLVE-BLOCK-END
