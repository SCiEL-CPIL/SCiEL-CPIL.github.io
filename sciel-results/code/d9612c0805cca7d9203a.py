import numpy as np
import time
import random
from scipy.optimize import minimize

N = 50

def find_sidon_set(N, K, attempts=1000):
    """
    Attempts to find a Sidon set of size K in [0, N).
    Returns the set if found, None otherwise.
    """
    for _ in range(attempts):
        candidates = list(range(N))
        random.shuffle(candidates)
        
        S = []
        sums = set()
        
        for cand in candidates:
            if len(S) == K:
                break
            
            conflict = False
            new_sums = set()
            
            if cand + cand in sums:
                conflict = True
            else:
                new_sums.add(cand + cand)
                
            if not conflict:
                for s in S:
                    sm = cand + s
                    if sm in sums or sm in new_sums:
                        conflict = True
                        break
                    new_sums.add(sm)
            
            if not conflict:
                S.append(cand)
                sums.update(new_sums)
        
        if len(S) == K:
            return S
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    def calculate_ratio(h):
        if np.sum(h) == 0:
            return 0.0
        l1 = np.sum(h)
        l2_sq = np.sum(h**2)
        c = np.convolve(h, h)
        peak = np.max(c)
        if peak == 0:
            return 0.0
        return l2_sq / (l1 * peak)

    # 1. Initial Best: Uniform
    best_h = np.ones(N) / N
    best_ratio = calculate_ratio(best_h)
    
    # 2. Find best Sidon set (Sparse structure)
    # Larger K yields higher ratio (R ~ K/2 for pure Sidon)
    best_sidon_h = None
    for K in range(12, 4, -1):
        if time.time() > deadline - 20.0:
            break
        attempts = 200 if K > 8 else 500
        S = find_sidon_set(N, K, attempts)
        if S is not None:
            h = np.zeros(N)
            h[S] = 1.0 / K
            r = calculate_ratio(h)
            if r > best_ratio:
                best_ratio = r
                best_h = h
                best_sidon_h = h
            # Since we iterate K downwards, the first found is the largest.
            # We can break if we found a reasonably large K, as smaller K will likely be worse
            # unless the geometry is significantly better, but pure Sidon K/2 scaling is strong.
            # However, let's allow a bit more time to try slightly smaller if large fails quickly?
            # No, if we found K=12, R~6. K=11, R~5.5. So we should stop.
            if K >= 10:
                break

    # 3. Optimization Phase: SLSQP to refine weights
    # We minimize M subject to M >= c_k and sum(h)=1, h>=0.
    # This is equivalent to minimizing the peak of the autoconvolution for fixed norm.
    
    def autoconv(h):
        return np.convolve(h, h)

    def neg_ratio_obj(x):
        # x = [h_0, ..., h_{N-1}, M]
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

    def constraints_autoconv(x):
        h = x[:N]
        M = x[N]
        c = autoconv(h)
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

    bounds = [(0.0, None)] * N + [(0.0, None)]
    cons = [
        {'type': 'eq', 'fun': constraint_sum, 'jac': constraint_sum_grad},
        {'type': 'ineq', 'fun': constraints_autoconv, 'jac': constraints_autoconv_grad}
    ]

    # Initial guesses for SLSQP
    initial_guesses = []
    
    # 1. Current best (Uniform or Sidon)
    if best_h is not None:
        c_best = autoconv(best_h)
        M_best = np.max(c_best)
        x_best = np.concatenate([best_h, [M_best]])
        initial_guesses.append(x_best)
        
    # 2. Uniform
    h_uni = np.ones(N) / N
    c_uni = autoconv(h_uni)
    M_uni = np.max(c_uni)
    initial_guesses.append(np.concatenate([h_uni, [M_uni]]))
    
    # 3. Random sparse guesses (Sidon-like or random)
    for _ in range(30):
        if time.time() > deadline - 10.0:
            break
        K_rand = np.random.randint(5, 15)
        support = np.random.choice(N, K_rand, replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / K_rand
        c_rand = autoconv(h_rand)
        M_rand = np.max(c_rand)
        initial_guesses.append(np.concatenate([h_rand, [M_rand]]))
        
        # Also try a few Gaussian-like bumps
        if _ % 2 == 0:
            center = np.random.randint(0, N)
            sigma = np.random.uniform(2, 8)
            x = np.arange(N)
            h_gauss = np.exp(-((x - center)**2) / (2 * sigma**2))
            h_gauss = h_gauss / np.sum(h_gauss)
            c_gauss = autoconv(h_gauss)
            M_gauss = np.max(c_gauss)
            initial_guesses.append(np.concatenate([h_gauss, [M_gauss]]))

    # Run SLSQP on each guess
    for x0 in initial_guesses:
        if time.time() > deadline - 5.0:
            break
        try:
            res = minimize(
                neg_ratio_obj, 
                x0, 
                method='SLSQP', 
                jac=neg_ratio_obj_grad,
                bounds=bounds, 
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-9}
            )
            
            if res.success:
                h_res = res.x[:N]
                s = np.sum(h_res)
                if s > 0:
                    h_res = h_res / s
                else:
                    continue
                h_res = np.clip(h_res, 0, None)
                h_res = h_res / np.sum(h_res)
                
                ratio_res = calculate_ratio(h_res)
                if ratio_res > best_ratio:
                    best_ratio = ratio_res
                    best_h = h_res.copy()
                    
        except Exception:
            continue

    # Final cleanup
    best_h = np.clip(best_h, 0, None)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return list(best_h)
