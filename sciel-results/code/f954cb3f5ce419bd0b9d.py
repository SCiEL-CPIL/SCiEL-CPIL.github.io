import numpy as np
from scipy.optimize import minimize
import time
import random

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    def score_raw(h):
        """Compute R = ||h||_2^2 / (||h||_1 * ||h*h||_inf) for non-negative h."""
        h = np.asarray(h, dtype=float)
        if len(h) == 0:
            return 0.0
        h1 = np.sum(h)
        if h1 <= 0:
            return 0.0
        h2 = np.sum(h * h)
        # Linear convolution
        c = np.convolve(h, h)
        cmax = np.max(c)
        if cmax <= 0:
            return 0.0
        return h2 / (h1 * cmax)

    def logsumexp(c, beta=100.0):
        """Smooth approximation of max using log-sum-exp."""
        c_max = np.max(c)
        if c_max <= 0:
            return 0.0
        return c_max + np.log(np.sum(np.exp(beta * (c - c_max)))) / beta

    def optimize_smooth(h_init, max_iter=300, ftol=1e-10):
        """
        Optimize using smooth surrogate. 
        We maximize log(||h||_2^2) - log(||h||_1) - log(||h*h||_inf_smooth)
        subject to sum(h)=1, h>=0.
        
        Using change of variables: h_i = exp(y_i) / sum(exp(y_j))
        This automatically satisfies sum(h)=1 and h>0.
        """
        h_init = np.asarray(h_init, dtype=float)
        # Ensure h_init is positive for log transform
        h_init = np.clip(h_init, 1e-10, None)
        h_init = h_init / np.sum(h_init)
        
        # Initial y values
        y0 = np.log(h_init)
        
        def objective(y):
            h = np.exp(y - np.max(y))  # For numerical stability
            h = h / np.sum(h)
            
            h1 = np.sum(h)
            h2 = np.sum(h * h)
            c = np.convolve(h, h)
            cmax_smooth = logsumexp(c, beta=100.0)
            
            if cmax_smooth <= 0 or h2 <= 0:
                return 0.0
            
            # We want to maximize h2 / (h1 * cmax)
            # So minimize -log(h2) + log(h1) + log(cmax)
            return -np.log(h2) + np.log(h1) + np.log(cmax_smooth)
        
        def gradient(y):
            h = np.exp(y - np.max(y))
            h = h / np.sum(h)
            
            h1 = np.sum(h)
            h2 = np.sum(h * h)
            c = np.convolve(h, h)
            
            # Gradient of logsumexp
            c_max = np.max(c)
            exp_terms = np.exp(100.0 * (c - c_max))
            weights = exp_terms / np.sum(exp_terms)
            
            # d/dh_i log(cmax_smooth) = sum_k weights[k] * d c_k / d h_i
            # d c_k / d h_i = 2 * h_{k-i} (with bounds check)
            grad_cmax = np.zeros(N)
            for k in range(len(c)):
                wk = weights[k]
                for i in range(N):
                    j = k - i
                    if 0 <= j < N:
                        grad_cmax[i] += 2.0 * wk * h[j]
            
            # d/dh_i (-log(h2) + log(h1) + log(cmax_smooth))
            # d/dh_i log(h2) = 2*h_i / h2
            # d/dh_i log(h1) = 1 / h1
            # d/dh_i log(cmax_smooth) = grad_cmax[i] / cmax_smooth
            
            grad_h = -2.0 * h / h2 + 1.0 / h1 + grad_cmax / cmax_smooth
            
            # Chain rule: d/dy_i = h_i * (1 - h_i) * sum_j grad_h[j] * delta_{ij} ... 
            # Actually, h = softmax(y), so d h_i / d y_j = h_i (delta_{ij} - h_j)
            # So d obj / d y_j = sum_i grad_h[i] * d h_i / d y_j = sum_i grad_h[i] * h_i * (delta_{ij} - h_j)
            # = h_j * (grad_h[j] - sum_i grad_h[i] * h_i)
            
            sum_grad_h_h = np.sum(grad_h * h)
            grad_y = h * (grad_h - sum_grad_h_h)
            
            return grad_y
        
        try:
            res = minimize(
                objective,
                y0,
                method='L-BFGS-B',
                jac=gradient,
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < objective(y0):
                y_res = res.x
                h_res = np.exp(y_res - np.max(y_res))
                h_res = h_res / np.sum(h_res)
                h_res = np.clip(h_res, 0, None)
                s = np.sum(h_res)
                if s > 0:
                    h_res = h_res / s
                ratio_res = score_raw(h_res)
                return h_res, ratio_res
        except Exception:
            pass
        return None, 0.0

    def polish(h_init, max_iter=500, ftol=1e-12):
        """
        Final polish using SLSQP with only active constraints.
        """
        h_init = np.asarray(h_init, dtype=float)
        h_init = np.clip(h_init, 0, None)
        s = np.sum(h_init)
        if s <= 0:
            return None, 0.0
        h_init = h_init / s
        
        c_init = np.convolve(h_init, h_init)
        cmax = np.max(c_init)
        
        # Identify active lags (within tolerance of max)
        tol = 1e-6 * cmax
        active_lags = np.where(c_init >= cmax - tol)[0]
        
        if len(active_lags) == 0:
            active_lags = [np.argmax(c_init)]
        
        # Variables: h (length N) and M (scalar)
        def neg_ratio_obj(x):
            return x[N]

        def neg_ratio_obj_grad(x):
            grad = np.zeros(N + 1)
            grad[N] = 1.0
            return grad

        def constraint_sum(x):
            return np.sum(x[:N]) - 1.0

        def constraint_sum_grad(x):
            grad = np.zeros(N + 1)
            grad[:N] = 1.0
            return grad

        def constraints_active(x):
            h = x[:N]
            M = x[N]
            c = np.convolve(h, h)
            return M - c[active_lags]

        def constraints_active_grad(x):
            h = x[:N]
            grad = np.zeros((len(active_lags), N + 1))
            for idx, k in enumerate(active_lags):
                derivs = np.zeros(N)
                for i in range(N):
                    j = k - i
                    if 0 <= j < N:
                        derivs[i] = 2.0 * h[j]
                grad[idx, :N] = -derivs
                grad[idx, N] = 1.0
            return grad

        M_init = cmax
        x0 = np.concatenate([h_init, [M_init]])
        bounds = [(0.0, None)] * N + [(0.0, None)]
        
        cons = [
            {'type': 'eq', 'fun': constraint_sum, 'jac': constraint_sum_grad},
            {'type': 'ineq', 'fun': constraints_active, 'jac': constraints_active_grad}
        ]
        
        try:
            res = minimize(
                neg_ratio_obj, 
                x0, 
                method='SLSQP', 
                jac=neg_ratio_obj_grad,
                bounds=bounds, 
                constraints=cons,
                options={'maxiter': max_iter, 'ftol': ftol}
            )
            
            if res.success:
                h_res = res.x[:N]
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

    best_h = None
    best_score = 0.0

    # 1. Start with uniform distribution
    h_uniform = np.ones(N) / N
    h_opt, s_opt = optimize_smooth(h_uniform)
    if h_opt is not None and s_opt > best_score:
        best_score = s_opt
        best_h = h_opt.copy()

    # 2. Try sparse random starts with smooth optimization
    num_restarts = 30
    for i in range(num_restarts):
        if time.time() > deadline - 15:
            break
            
        K = random.randint(3, 15)
        support = np.random.choice(N, K, replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / K
        
        h_opt, s_opt = optimize_smooth(h_rand, max_iter=200)
        if h_opt is not None and s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()

    # 3. Try structured sets
    for base in [2, 3]:
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
            h_opt, s_opt = optimize_smooth(h_geom, max_iter=200)
            if h_opt is not None and s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()

    # 4. Polish the best solution with active-set SLSQP
    if best_h is not None and time.time() < deadline - 10:
        h_polished, s_polished = polish(best_h, max_iter=500, ftol=1e-12)
        if h_polished is not None and s_polished > best_score:
            best_score = s_polished
            best_h = h_polished.copy()

    # 5. If we still have time, do more random restarts
    while time.time() < deadline - 5:
        K = random.randint(2, 20)
        support = np.random.choice(N, K, replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / K
        
        h_opt, s_opt = optimize_smooth(h_rand, max_iter=150)
        if h_opt is not None and s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
            
            # Quick polish
            if time.time() < deadline - 5:
                h_polished, s_polished = polish(best_h, max_iter=200, ftol=1e-12)
                if h_polished is not None and s_polished > best_score:
                    best_score = s_polished
                    best_h = h_polished.copy()

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
