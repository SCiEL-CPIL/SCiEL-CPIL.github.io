import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import roots_genlaguerre, eval_genlaguerre
import time

def construct():
    start_time = time.time()
    deadline = start_time + 590

    m = 12
    alpha = 1.0
    
    # Gauss-Laguerre quadrature for weight x^1 e^-x
    # We need to integrate polynomials up to degree 12 (from P) + 12 (from L_k) = 24.
    # N nodes are exact for degree 2N-1. 2N-1 >= 24 => N >= 12.5 => N=13.
    # Use N=15 for safety.
    n_quad = 15
    x_nodes, w_weights = roots_genlaguerre(n_quad, alpha)
    
    # Precompute L_k^(1)(x_i) for k=0..12
    # eval_genlaguerre(k, alpha, x)
    L_vals = np.zeros((13, n_quad))
    for k in range(13):
        L_vals[k, :] = eval_genlaguerre(k, alpha, x_nodes)
        
    # Norms of L_k: ||L_k^(1)||^2 = k+1
    norms_sq_L = np.arange(13) + 1.0

    def compute_norm_sq(roots):
        """
        Compute squared norm of P(x) = prod(x-r_i) in L^2(x e^-x).
        """
        # Evaluate P(x) at quadrature nodes
        # P(x) = prod (x - r_i)
        # Vectorized: for each node x_i, compute prod_j (x_i - r_j)
        # x_nodes is shape (15,), roots is shape (12,)
        # diff is shape (15, 12)
        diff = x_nodes[:, np.newaxis] - roots[np.newaxis, :]
        P_vals = np.prod(diff, axis=1) # shape (15,)
        
        # Compute inner products <P, L_k>
        # I_k = sum_i w_i P(x_i) L_k(x_i)
        # I = w * P_vals (shape 15) dot L_vals (shape 13, 15) -> (13,)
        # w_weights is shape (15,)
        weighted_P = w_weights * P_vals
        I = L_vals @ weighted_P # shape (13,)
        
        # c_k = I_k / (k+1)
        # norm_sq = sum c_k^2 (k+1) = sum (I_k^2 / (k+1)^2) * (k+1) = sum I_k^2 / (k+1)
        norm_sq = np.sum(I**2 / norms_sq_L)
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        """
        Minimize 1/norm_sq.
        """
        roots = np.exp(log_roots)
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
        
        norm_sq = compute_norm_sq(roots)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    # Optimization Strategy:
    # 1. Differential Evolution for global search
    # 2. Local polish with L-BFGS-B or Powell
    
    # Bounds for log-roots
    # Seed: 4 * 2^i. i=0..11.
    # min log: log(4) ~ 1.38
    # max log: log(8192) ~ 9.0
    # Let's allow a wider range: [0, 15]
    bounds = [(0.0, 15.0)] * 12
    
    best_val = 1e10
    best_log_roots = None
    best_roots = None
    
    # Run Differential Evolution
    # popsize=25, maxiter=200
    try:
        result_de = differential_evolution(
            cost_func,
            bounds,
            seed=42,
            maxiter=200,
            popsize=25,
            tol=1e-8,
            polish=True, # Uses L-BFGS-B for final polish
            init='sobol'
        )
        
        if result_de.fun < best_val:
            best_val = result_de.fun
            best_log_roots = result_de.x
            best_roots = np.exp(result_de.x)
    except Exception as e:
        pass

    # If DE failed or we have time, do a local polish
    if best_log_roots is not None:
        if time.time() - start_time < deadline - 10:
            try:
                result_powell = minimize(
                    cost_func,
                    best_log_roots,
                    method='Powell',
                    options={'maxiter': 1000, 'xtol': 1e-12, 'ftol': 1e-15}
                )
                if result_powell.fun < best_val:
                    best_val = result_powell.fun
                    best_log_roots = result_powell.x
                    best_roots = np.exp(result_powell.x)
            except Exception:
                pass

    # Fallback to seed if optimization failed
    if best_roots is None:
        seed_roots = np.array([4.0 * (2.0 ** i) for i in range(12)])
        best_roots = seed_roots
        best_val = cost_func(np.log(seed_roots))

    # Ensure roots are positive and sorted
    best_roots = np.maximum(best_roots, 1e-6)
    best_roots = np.sort(best_roots)
    
    return list(best_roots)
