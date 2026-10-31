"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 50

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    
    def compute_score(heights):
        """Compute the ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
        h = np.array(heights, dtype=np.float64)
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        if l1 < 1e-12:
            return 0.0
        # Autoconvolution (linear, full mode)
        conv = np.convolve(h, h, mode='full')
        linf = np.max(conv)
        if linf < 1e-12:
            return 0.0
        return l2_sq / (l1 * linf)
    
    def find_sidon_set(n, target_size):
        """Find a Sidon set (B2 sequence) of size target_size within range(n) using greedy."""
        sidon = set()
        for i in range(n):
            new_set = sidon | {i}
            sums = set()
            valid = True
            lst = sorted(new_set)
            for idx_a in range(len(lst)):
                for idx_b in range(idx_a, len(lst)):
                    s = lst[idx_a] + lst[idx_b]
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            if valid:
                sidon.add(i)
                if len(sidon) >= target_size:
                    break
        
        if len(sidon) < target_size:
            for start in range(1, 15):
                sidon = set()
                for i in range(start, n):
                    new_set = sidon | {i}
                    sums = set()
                    valid = True
                    lst = sorted(new_set)
                    for idx_a in range(len(lst)):
                        for idx_b in range(idx_a, len(lst)):
                            s = lst[idx_a] + lst[idx_b]
                            if s in sums:
                                valid = False
                                break
                            sums.add(s)
                    if not valid:
                        continue
                    sidon.add(i)
                    if len(sidon) >= target_size:
                        break
                if len(sidon) >= target_size:
                    break
        
        return list(sidon)
    
    def local_search_support(support, n, max_iters=3000):
        """Local search: try swaps, additions, and removals to maximize score."""
        current = set(support)
        h = np.zeros(n)
        for i in current:
            h[i] = 1.0
        current_score = compute_score(h)
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 15:
            improved = False
            iters += 1
            
            # Try all swaps
            for elem in list(current):
                for new_elem in range(n):
                    if new_elem in current:
                        continue
                    trial = set(current)
                    trial.discard(elem)
                    trial.add(new_elem)
                    h_trial = np.zeros(n)
                    for i in trial:
                        h_trial[i] = 1.0
                    trial_score = compute_score(h_trial)
                    if trial_score > current_score + 1e-10:
                        current = trial
                        current_score = trial_score
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            
            # Try adding one element
            for new_elem in range(n):
                if new_elem in current:
                    continue
                trial = set(current)
                trial.add(new_elem)
                h_trial = np.zeros(n)
                for i in trial:
                    h_trial[i] = 1.0
                trial_score = compute_score(h_trial)
                if trial_score > current_score + 1e-10:
                    current = trial
                    current_score = trial_score
                    improved = True
                    break
            if improved:
                continue
            
            # Try removing one element (only if size > 3)
            if len(current) > 3:
                for elem in list(current):
                    trial = set(current)
                    trial.discard(elem)
                    h_trial = np.zeros(n)
                    for i in trial:
                        h_trial[i] = 1.0
                    trial_score = compute_score(h_trial)
                    if trial_score > current_score + 1e-10:
                        current = trial
                        current_score = trial_score
                        improved = True
                        break
        
        return list(current), current_score
    
    def continuous_optimize(support, n, max_iters=5000, tol=1e-12):
        """Nelder-Mead optimization of heights on a fixed support."""
        from scipy.optimize import minimize
        
        support_arr = np.array(support)
        k = len(support_arr)
        if k < 2:
            return None
        
        def neg_score(x):
            h = np.zeros(n)
            h[support_arr] = x
            return -compute_score(h)
        
        # Multiple starts
        best_x = None
        best_val = np.inf
        
        for start_iter in range(3):
            if time.time() > deadline - 10:
                break
            if start_iter == 0:
                x0 = np.ones(k)
            elif start_iter == 1:
                x0 = np.random.dirichlet(np.ones(k))
            else:
                x0 = np.random.exponential(1.0, k)
                x0 /= x0.sum()
            
            res = minimize(neg_score, x0, method='Nelder-Mead',
                          options={'maxiter': max_iters, 'xatol': tol, 'fatol': tol, 'adaptive': True})
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        
        if best_x is None:
            return None
        
        h = np.zeros(n)
        h[support_arr] = best_x
        score = compute_score(h)
        return h, score
    
    # Target Sidon set sizes to try
    best_heights = np.ones(N)
    best_score = compute_score(best_heights)
    
    # Try various Sidon set sizes
    for target_size in [6, 8, 10, 12, 14, 16, 18, 20]:
        if time.time() > deadline - 30:
            break
        
        sidon = find_sidon_set(N, target_size)
        if len(sidon) < 4:
            continue
        
        # Local search to improve the support
        improved_support, sidon_score = local_search_support(sidon, N)
        
        if sidon_score > best_score:
            best_score = sidon_score
            h = np.zeros(N)
            for i in improved_support:
                h[i] = 1.0
            best_heights = h.copy()
        
        # Also try random sparse sets with local search
        for trial in range(3):
            if time.time() > deadline - 25:
                break
            mask = np.zeros(N, dtype=bool)
            mask[random.sample(range(N), min(target_size, N))] = True
            h = mask.astype(float)
            s = compute_score(h)
            if s > best_score:
                best_score = s
                best_heights = h.copy()
                support = list(np.where(mask)[0])
            else:
                support = list(np.where(mask)[0])
            
            # Quick local search on this random support
            if time.time() < deadline - 20:
                improved_sup, sup_score = local_search_support(support, N, max_iters=500)
                if sup_score > best_score:
                    best_score = sup_score
                    h = np.zeros(N)
                    for i in improved_sup:
                        h[i] = 1.0
                    best_heights = h.copy()
    
    # Continuous optimization on the best support found
    if time.time() < deadline - 10:
        support = list(np.where(best_heights > 0)[0])
        if len(support) >= 2:
            result = continuous_optimize(support, N)
            if result is not None:
                h_opt, score_opt = result
                if score_opt > best_score:
                    best_score = score_opt
                    best_heights = h_opt.copy()
    
    # Try continuous optimization on a few alternative supports
    if time.time() < deadline - 8:
        # Try a few more support sizes
        for ts in [8, 12, 16]:
            if time.time() > deadline - 5:
                break
            sidon = find_sidon_set(N, ts)
            if len(sidon) < 4:
                continue
            result = continuous_optimize(sidon, N, max_iters=2000)
            if result is not None:
                h_opt, score_opt = result
                if score_opt > best_score:
                    best_score = score_opt
                    best_heights = h_opt.copy()
    
    # Ensure nonnegative and valid
    best_heights = np.maximum(best_heights, 0)
    if np.sum(best_heights) < 1e-12:
        return [1.0] * N
    
    return list(best_heights)
# EVOLVE-BLOCK-END
