import numpy as np
from scipy.optimize import minimize
import time
import random

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def score_raw(h):
        """Compute R = ||h||_2^2 / (||h||_1 * ||h*h||_inf) for non-negative h."""
        h = np.asarray(h, dtype=float)
        if len(h) == 0:
            return 0.0
        h1 = np.sum(h)
        if h1 <= 0:
            return 0.0
        h2 = np.sum(h * h)
        c = np.convolve(h, h)
        cmax = np.max(c)
        if cmax <= 0:
            return 0.0
        return h2 / (h1 * cmax)

    def optimize_with_peak_constraint(h_init, max_iter=300, ftol=1e-10):
        """
        Maximize ||h||_2^2 subject to:
        - sum(h) = 1
        - h >= 0
        - c_k <= M for all k (where M is the peak)
        
        We use an epigraph formulation: minimize -||h||_2^2 + penalty*M
        Actually, better: minimize M subject to c_k <= M, sum(h)=1, h>=0,
        then maximize ||h||_2^2 given that M.
        
        Alternative: directly optimize using log-sum-exp smoothing of the max.
        """
        h_init = np.asarray(h_init, dtype=float).copy()
        if np.sum(h_init) <= 0:
            return None, 0.0
        h_init = h_init / np.sum(h_init)
        
        # Use a smooth surrogate: replace max(c) with soft-max via log-sum-exp
        # R = ||h||_2^2 / (||h||_1 * soft_max(c))
        # Minimize -log(R) = -log(||h||_2^2) + log(||h||_1) + log(soft_max(c))
        # For simplicity, let's use the epigraph with SLSQP but with better init
        
        # First, identify the dominant lags
        c = np.convolve(h_init, h_init)
        cmax = np.max(c)
        
        # Use only the top few constraints to reduce problem size
        top_k = min(20, len(c))
        idx = np.argsort(c)[::-1][:top_k]
        
        def obj(x):
            h = x[:N]
            h2 = np.sum(h * h)
            h1 = np.sum(h)
            if h1 <= 0 or h2 <= 0:
                return 1e10
            c = np.convolve(h, h)
            cmax = np.max(c)
            if cmax <= 0:
                return 1e10
            return -h2 / (h1 * cmax)
        
        def obj_grad(x):
            h = x[:N]
            h1 = np.sum(h)
            h2 = np.sum(h * h)
            if h1 <= 0 or h2 <= 0:
                return np.zeros(N)
            c = np.convolve(h, h)
            cmax = np.max(c)
            if cmax <= 0:
                return np.zeros(N)
            
            # d/dh_i [h2/(h1*cmax)] = (2*h_i*h1*cmax - h2*(cmax/h1 + d(cmax)/dh_i)) / (h1*cmax)^2
            # This is complex; use numerical gradient or simplify
            # For now, return zeros and let SLSQP use numerical gradients
            return np.zeros(N)
        
        def constraint_sum(x):
            return np.sum(x) - 1.0
        
        # Bounds
        bounds = [(0.0, None)] * N
        
        try:
            res = minimize(
                obj,
                h_init,
                method='SLSQP',
                bounds=bounds,
                constraints=[{'type': 'eq', 'fun': constraint_sum}],
                options={'maxiter': max_iter, 'ftol': ftol}
            )
            
            if res.success:
                h_res = res.x
                s = np.sum(h_res)
                if s > 0:
                    h_res = h_res / s
                else:
                    return None, 0.0
                h_res = np.clip(h_res, 0, None)
                s = np.sum(h_res)
                if s > 0:
                    h_res = h_res / s
                else:
                    return None, 0.0
                ratio_res = score_raw(h_res)
                return h_res, ratio_res
        except Exception:
            pass
        
        # Fallback to epigraph formulation
        return optimize_epigraph(h_init, max_iter, ftol)

    def optimize_epigraph(h_init, max_iter=300, ftol=1e-10):
        """Epigraph formulation: minimize M s.t. c_k <= M, sum(h)=1, h>=0"""
        h_init = np.asarray(h_init, dtype=float).copy()
        if np.sum(h_init) <= 0:
            return None, 0.0
        h_init = h_init / np.sum(h_init)
        
        c_init = np.convolve(h_init, h_init)
        M_init = np.max(c_init)
        x0 = np.concatenate([h_init, [M_init]])
        
        def obj(x):
            return x[N]
        
        def obj_grad(x):
            grad = np.zeros(N + 1)
            grad[N] = 1.0
            return grad
        
        def constraint_sum(x):
            return np.sum(x[:N]) - 1.0
        
        def constraint_sum_grad(x):
            grad = np.zeros(N + 1)
            grad[:N] = 1.0
            return grad
        
        # Only use top 30 convolution constraints to keep problem tractable
        c = np.convolve(h_init, h_init)
        top_k = min(30, 2 * N - 1)
        idx = np.argsort(c)[::-1][:top_k]
        
        def constraints_autoconv(x):
            h = x[:N]
            M = x[N]
            c = np.convolve(h, h)
            return M - c[idx]
        
        def constraints_autoconv_grad(x):
            h = x[:N]
            grad = np.zeros((top_k, N + 1))
            for ii, k in enumerate(idx):
                derivs = np.zeros(N)
                for i in range(N):
                    j = k - i
                    if 0 <= j < N:
                        derivs[i] = 2.0 * h[j]
                grad[ii, :N] = -derivs
                grad[ii, N] = 1.0
            return grad
        
        bounds = [(0.0, None)] * N + [(0.0, None)]
        
        cons = [
            {'type': 'eq', 'fun': constraint_sum, 'jac': constraint_sum_grad},
            {'type': 'ineq', 'fun': constraints_autoconv, 'jac': constraints_autoconv_grad}
        ]
        
        try:
            res = minimize(
                obj,
                x0,
                method='SLSQP',
                jac=obj_grad,
                bounds=bounds,
                constraints=cons,
                options={'maxiter': max_iter, 'ftol': ftol}
            )
            
            if res.success:
                h_res = res.x[:N]
                s = np.sum(h_res)
                if s > 0:
                    h_res = h_res / s
                else:
                    return None, 0.0
                h_res = np.clip(h_res, 0, None)
                s = np.sum(h_res)
                if s > 0:
                    h_res = h_res / s
                else:
                    return None, 0.0
                ratio_res = score_raw(h_res)
                return h_res, ratio_res
        except Exception:
            pass
        
        return None, 0.0

    def polish(h, n_iter=50):
        """Final high-precision polish using coordinate descent on the peak constraint."""
        h = np.asarray(h, dtype=float).copy()
        s = np.sum(h)
        if s <= 0:
            return h
        h = h / s
        
        for _ in range(n_iter):
            if time.time() > deadline - 3:
                break
            c = np.convolve(h, h)
            cmax = np.max(c)
            k_max = np.argmax(c)
            
            # Find pairs (i, j) with i+j == k_max that contribute to the peak
            # Try to reduce the peak by adjusting h at these positions
            # while maintaining sum(h) = 1
            
            # Simple coordinate descent: for each i, try small perturbations
            improved = False
            for i in range(N):
                if h[i] <= 0:
                    continue
                # Try moving mass from i to another position j
                for j in range(N):
                    if j == i or h[j] >= h[i]:
                        continue
                    # Try small transfer
                    for eps in [0.001, 0.0001]:
                        h_new = h.copy()
                        h_new[i] -= eps
                        h_new[j] += eps
                        if h_new[i] < 0:
                            continue
                        s_new = np.sum(h_new)
                        if s_new <= 0:
                            continue
                        h_new = h_new / s_new
                        score_new = score_raw(h_new)
                        score_old = score_raw(h)
                        if score_new > score_old * (1 + 1e-10):
                            h = h_new
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
        
        return h

    best_h = None
    best_score = 0.0

    # 1. Start with uniform distribution
    h_uniform = np.ones(N) / N
    h_opt, s_opt = optimize_with_peak_constraint(h_uniform)
    if h_opt is not None and s_opt > best_score:
        best_score = s_opt
        best_h = h_opt.copy()

    # 2. Try sparse random starts with Sidon-like structures
    num_restarts = 25
    for i in range(num_restarts):
        if time.time() > deadline - 15:
            break
            
        K = random.randint(4, 18)
        support = np.random.choice(N, K, replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / K
        
        h_opt, s_opt = optimize_with_peak_constraint(h_rand)
        if h_opt is not None and s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()

    # 3. Structured starts: geometric progressions
    for base in [2, 3, 4]:
        if time.time() > deadline - 15:
            break
        support = []
        val = 1
        while val < N:
            support.append(val)
            val *= base
        if len(support) > 0:
            h_geom = np.zeros(N)
            h_geom[support] = 1.0 / len(support)
            h_opt, s_opt = optimize_with_peak_constraint(h_geom)
            if h_opt is not None and s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()

    # 4. More random restarts if time permits
    while time.time() < deadline - 10:
        K = random.randint(3, 22)
        support = np.random.choice(N, K, replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / K
        
        h_opt, s_opt = optimize_with_peak_constraint(h_rand, max_iter=150)
        if h_opt is not None and s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()

    # 5. Final polish pass
    if best_h is not None and time.time() < deadline - 5:
        best_h = polish(best_h, n_iter=100)
        best_score = score_raw(best_h)

    # Fallback
    if best_h is None:
        best_h = np.ones(N) / N
        best_score = score_raw(best_h)

    # Ensure non-negative and normalized
    best_h = np.clip(best_h, 0, None)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return list(best_h)
