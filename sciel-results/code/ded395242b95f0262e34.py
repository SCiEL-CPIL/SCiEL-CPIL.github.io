import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def score(h):
        h = np.maximum(h, 0)
        l1 = np.sum(h)
        if l1 < 1e-10:
            return 0.0
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-10:
            return 0.0
        return l2_sq / (l1 * max_conv)

    def smooth_max_conv(h, beta=100.0):
        """Compute smooth approximation to max of autoconvolution using log-sum-exp."""
        conv = np.convolve(h, h, mode='full')
        # Use log-sum-exp for numerical stability
        max_val = np.max(conv)
        if max_val <= 0:
            return 0.0
        # log-sum-exp: max + log(sum(exp(x - max)))
        # For our purpose, we want a smooth surrogate for max(conv)
        # Use: max * (1 + (sum(exp(beta*(conv/max - 1))))^(-1/beta))
        # Simpler: just use the actual max for scoring, but smooth for optimization
        return max_val

    def get_max_mult(S):
        if not S:
            return 0
        sums = {}
        S_list = list(S)
        for i in S_list:
            for j in S_list:
                s = i + j
                sums[s] = sums.get(s, 0) + 1
        return max(sums.values())

    def greedy_build(seed_set, max_size=15):
        S = set(seed_set)
        for _ in range(max_size):
            best_x = -1
            best_max_mult = float('inf')
            best_ratio = -1
            
            for x in range(N):
                if x in S:
                    continue
                S_trial = S | {x}
                m = get_max_mult(S_trial)
                k = len(S_trial)
                ratio = k / m if m > 0 else 0
                # We want to minimize M, maximize K/M
                if m < best_max_mult:
                    best_max_mult = m
                    best_x = x
                    best_ratio = ratio
                elif m == best_max_mult:
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_x = x
            
            if best_x != -1:
                S.add(best_x)
            else:
                break
        return S

    best_h = np.zeros(N)
    best_score = 0.0
    
    # 1. Generate diverse initial candidates using Sidon-like sets
    seeds = [
        [0],
        [0, 1],
        [0, 1, 4],
        [0, 1, 4, 10],
        [0, 1, 4, 10, 18],
        [0, 1, 4, 10, 18, 23, 30],
        [1, 2, 5, 11, 21, 34],
        [0, 1, 3, 7, 12, 20, 31],
        [0, 2, 5, 9, 14, 21, 29, 38],
        [0, 1, 4, 10, 18, 28, 39],
        [0, 3, 7, 12, 18, 25, 33, 42],
        [0, 2, 7, 13, 20, 28, 37, 46],
    ]
    
    for seed in seeds:
        if time.time() > deadline - 20:
            break
        S = greedy_build(seed)
        if not S:
            continue
        h = np.zeros(N)
        for i in S:
            h[i] = 1.0 / len(S)
        s = score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
            
    # Also try random sparse sets to diversify
    np.random.seed(42)
    for _ in range(2000):
        if time.time() > deadline - 20:
            break
        k = np.random.randint(3, 18)
        S = np.random.choice(N, k, replace=False)
        h = np.zeros(N)
        h[S] = 1.0 / k
        s = score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()

    # 2. Local Search: Set Swaps with multiple restarts
    S_best = set(np.nonzero(best_h)[0])
    current_score = best_score
    
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        # Try swapping each element in S_best with an element outside
        for i in list(S_best):
            if time.time() > deadline - 15:
                break
            for j in range(N):
                if j in S_best:
                    continue
                S_trial = S_best.copy()
                S_trial.remove(i)
                S_trial.add(j)
                
                h_trial = np.zeros(N)
                for k in S_trial:
                    h_trial[k] = 1.0 / len(S_trial)
                
                s_trial = score(h_trial)
                if s_trial > current_score + 1e-9:
                    S_best = S_trial
                    best_h = h_trial
                    current_score = s_trial
                    improved = True
                    break
            if improved:
                break

    # 3. Continuous Optimization with smooth surrogate
    try:
        from scipy.optimize import minimize
        
        support = list(S_best)
        k = len(support)
        if k == 0:
            return list(best_h)
            
        # Initial amplitudes: uniform
        x0 = np.ones(k) / k
        
        def neg_log_ratio(x):
            h_full = np.zeros(N)
            h_full[support] = np.maximum(x, 0)
            
            l1 = np.sum(h_full)
            if l1 < 1e-10:
                return 1000.0
            l2_sq = np.sum(h_full**2)
            conv = np.convolve(h_full, h_full, mode='full')
            max_conv = np.max(conv)
            if max_conv < 1e-10:
                return 1000.0
            
            ratio = l2_sq / (l1 * max_conv)
            if ratio <= 0:
                return 1000.0
            return -np.log(ratio)
        
        # Use numerical gradient for reliability
        def jac_neg_log_ratio(x):
            eps = 1e-7
            grad = np.zeros(k)
            f0 = neg_log_ratio(x)
            for i in range(k):
                x_plus = x.copy()
                x_plus[i] += eps
                f_plus = neg_log_ratio(x_plus)
                grad[i] = (f_plus - f0) / eps
            return grad
        
        # Run optimization with multiple restarts
        best_x = x0.copy()
        best_fun = neg_log_ratio(x0)
        
        for restart in range(3):
            if time.time() > deadline - 10:
                break
                
            if restart == 0:
                x_init = x0.copy()
            else:
                # Perturb initial point
                x_init = x0 * np.random.uniform(0.5, 2.0, k)
                x_init /= np.sum(x_init)
            
            bounds = [(1e-10, None) for _ in range(k)]
            
            try:
                res = minimize(neg_log_ratio, x_init, method='L-BFGS-B', 
                               jac=jac_neg_log_ratio,
                               bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
                
                if res.fun < best_fun:
                    best_fun = res.fun
                    best_x = res.x
            except Exception:
                pass
        
        if best_fun < neg_log_ratio(x0):
            h_opt = np.zeros(N)
            h_opt[support] = np.maximum(best_x, 0)
            
            s_opt = score(h_opt)
            if s_opt > current_score:
                best_h = h_opt
                current_score = s_opt
                
    except Exception:
        pass

    # 4. Final polish: coordinate descent with fine-grained scaling
    h = best_h.copy()
    support = list(np.nonzero(h)[0])
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        current_score = score(h)
        for idx in support:
            if time.time() > deadline - 2:
                break
            h_old = h[idx]
            if h_old < 1e-10:
                continue
            
            # Try a range of scaling factors
            for factor in np.linspace(0.1, 5.0, 25):
                h_trial = h.copy()
                h_trial[idx] = h_old * factor
                s_trial = score(h_trial)
                if s_trial > current_score + 1e-10:
                    h = h_trial
                    current_score = s_trial
                    improved = True
                    break

    # 5. One more pass of support optimization
    if time.time() < deadline - 5:
        S_current = set(np.nonzero(h)[0])
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in list(S_current):
                if time.time() > deadline - 5:
                    break
                for j in range(N):
                    if j in S_current:
                        continue
                    S_trial = S_current.copy()
                    S_trial.remove(i)
                    S_trial.add(j)
                    
                    h_trial = h.copy()
                    h_trial[:] = 0
                    for k in S_trial:
                        h_trial[k] = h[k] if k in S_current else 1.0
                    
                    # Reoptimize amplitudes quickly
                    support_trial = list(S_trial)
                    if len(support_trial) > 0:
                        x_trial = np.array([h_trial[k] for k in support_trial])
                        # Simple coordinate ascent
                        for _ in range(3):
                            for idx in support_trial:
                                pos = support_trial.index(idx)
                                old_val = x_trial[pos]
                                if old_val < 1e-10:
                                    continue
                                for factor in [0.5, 0.8, 1.2, 1.5, 2.0]:
                                    x_trial_new = x_trial.copy()
                                    x_trial_new[pos] = old_val * factor
                                    h_temp = np.zeros(N)
                                    h_temp[support_trial] = np.maximum(x_trial_new, 0)
                                    s_temp = score(h_temp)
                                    if s_temp > score(h_trial):
                                        x_trial = x_trial_new
                                        h_trial = h_temp
                                        break
                        h_trial[support_trial] = np.maximum(x_trial, 0)
                    
                    s_trial = score(h_trial)
                    if s_trial > current_score + 1e-9:
                        S_current = S_trial
                        h = h_trial
                        current_score = s_trial
                        improved = True
                        break
                if improved:
                    break
        best_h = h
        current_score = current_score

    return list(best_h)
