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

    def optimize_full_h(h_init, max_iter=500, ftol=1e-8):
        """
        Optimize h directly using SLSQP with epigraph formulation.
        Variables: x = [h_0, ..., h_{N-1}, M]
        Minimize M subject to sum(h)=1, M >= c_k for all k, h >= 0
        """
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

        def constraints_autoconv(x):
            h = x[:N]
            M = x[N]
            c = np.convolve(h, h)
            return M - c

        def constraints_autoconv_grad(x):
            h = x[:N]
            grad = np.zeros((2*N - 1, N + 1))
            for k in range(2*N - 1):
                derivs = np.zeros(N)
                for i in range(N):
                    j = k - i
                    if 0 <= j < N:
                        derivs[i] = 2.0 * h[j]
                grad[k, :N] = -derivs
                grad[k, N] = 1.0
            return grad

        h_init = np.asarray(h_init, dtype=float)
        c_init = np.convolve(h_init, h_init)
        M_init = np.max(c_init)
        x0 = np.concatenate([h_init, [M_init]])
        
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
            
            if res.success or res.fun < np.max(c_init):
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

    def polish(h_init, max_iter=2000, ftol=1e-10):
        """High-precision polish pass with tighter tolerance."""
        return optimize_full_h(h_init, max_iter=max_iter, ftol=ftol)

    def find_good_sidon_support(target_size):
        """Greedy construction of a near-Sidon set to minimize pairwise sum collisions."""
        if target_size <= 0:
            return []
        
        support = [0]
        used_sums = {0: 1}
        
        for _ in range(target_size - 1):
            best_candidate = None
            best_score = -1
            
            for candidate in range(N):
                if candidate in support:
                    continue
                
                # Count new collisions
                new_collisions = 0
                for s in support:
                    pair_sum = candidate + s
                    if pair_sum in used_sums:
                        new_collisions += 1
                
                # We want to minimize collisions, so maximize (target - collisions)
                score = target_size - new_collisions
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
            
            if best_candidate is not None:
                support.append(best_candidate)
                for s in support:
                    pair_sum = best_candidate + s
                    used_sums[pair_sum] = used_sums.get(pair_sum, 0) + 1
        
        return support

    best_h = None
    best_score = 0.0

    # 1. Start with uniform distribution
    h_uniform = np.ones(N) / N
    h_opt, s_opt = optimize_full_h(h_uniform)
    if h_opt is not None and s_opt > best_score:
        best_score = s_opt
        best_h = h_opt.copy()

    # 2. Try Sidon-like structured supports
    for target_k in [5, 6, 7, 8, 9, 10]:
        if time.time() > deadline - 15:
            break
        
        support = find_good_sidon_support(target_k)
        if len(support) >= 3:
            h_sidon = np.zeros(N)
            h_sidon[support] = 1.0 / len(support)
            h_opt, s_opt = optimize_full_h(h_sidon)
            if h_opt is not None and s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()

    # 3. Try random sparse starts with varying support sizes
    num_restarts = 15
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        K = random.randint(3, 12)
        support = np.random.choice(N, K, replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / K
        
        h_opt, s_opt = optimize_full_h(h_rand)
        if h_opt is not None and s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()

    # 4. Try geometric progressions
    for base in [2, 3]:
        if time.time() > deadline - 10:
            break
        support = []
        val = 0
        while val < N:
            support.append(val)
            val = val * base + 1
        if len(support) > 2:
            h_geom = np.zeros(N)
            h_geom[support] = 1.0 / len(support)
            h_opt, s_opt = optimize_full_h(h_geom)
            if h_opt is not None and s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()

    # 5. More random restarts if time allows
    while time.time() < deadline - 8:
        K = random.randint(2, 15)
        support = np.random.choice(N, K, replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / K
        
        h_opt, s_opt = optimize_full_h(h_rand, max_iter=300)
        if h_opt is not None and s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()

    # 6. Final polish pass on the best solution found
    if best_h is not None and time.time() < deadline - 3:
        h_polished, s_polished = polish(best_h, max_iter=2000, ftol=1e-10)
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
