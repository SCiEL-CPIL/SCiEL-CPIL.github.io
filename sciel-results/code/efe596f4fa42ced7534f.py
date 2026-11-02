import numpy as np
import time
import random
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def score(h):
        h = np.asarray(h)
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

    def get_max_mult(S):
        if not S:
            return 0
        sums = {}
        S_list = list(S)
        for i in S_list:
            for j in S_list:
                s = i + j
                sums[s] = sums.get(s, 0) + 1
        return max(sums.values()) if sums else 0

    def local_search_support(S_init, max_iters=5000):
        """
        Local search to minimize M_max (max multiplicity of sums) for a fixed size set.
        We allow swapping elements to reduce M_max.
        """
        S = set(S_init)
        k = len(S)
        current_m = get_max_mult(S)
        
        # Precompute sums for current S
        # Actually, just recalculating is fast enough for k <= 20, N=50
        # 20*20 = 400 ops per eval. 5000 iters = 2M ops. Fine.
        
        for _ in range(max_iters):
            if time.time() > deadline - 20:
                break
            improved = False
            # Try all possible swaps
            # To speed up, we can sample swaps or iterate through all
            # Let's iterate through all i in S, j not in S
            S_list = list(S)
            random.shuffle(S_list)
            candidates = [x for x in range(N) if x not in S]
            random.shuffle(candidates)
            
            # Limit candidates to check per iteration to save time if needed, 
            # but N=50 is small.
            for i in S_list:
                for j in candidates:
                    S_trial = S.copy()
                    S_trial.remove(i)
                    S_trial.add(j)
                    m_trial = get_max_mult(S_trial)
                    if m_trial < current_m:
                        S = S_trial
                        current_m = m_trial
                        improved = True
                        # Restart search with new S
                        break
                if improved:
                    break
            
            if not improved:
                # If no improvement, do a random perturbation (simulated annealing step)
                # or break if we want pure hill climbing. 
                # Let's try a random swap to escape local minima.
                if random.random() < 0.1:
                    i = random.choice(list(S))
                    j = random.choice(candidates)
                    S.remove(i)
                    S.add(j)
                    current_m = get_max_mult(S)
                    # Don't update best immediately, just move
                else:
                    break
        return S

    def optimize_amplitudes(S):
        """
        Optimize amplitudes for a given support S.
        """
        if not S:
            return np.zeros(N), 0.0
        support = list(S)
        k = len(support)
        
        # Initial guess: uniform
        x0 = np.ones(k) / k
        
        def neg_log_ratio(x):
            h_full = np.zeros(N)
            h_full[support] = x
            # Enforce non-negativity in objective? 
            # L-BFGS-B handles bounds, so x should be >= 0.
            l1 = np.sum(h_full)
            if l1 < 1e-10:
                return 0.0
            l2_sq = np.sum(h_full**2)
            conv = np.convolve(h_full, h_full, mode='full')
            max_conv = np.max(conv)
            if max_conv < 1e-10:
                return 0.0
            ratio = l2_sq / (l1 * max_conv)
            if ratio <= 0:
                return 0.0
            return -np.log(ratio)
        
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

        bounds = [(1e-8, None) for _ in range(k)]
        
        try:
            res = minimize(neg_log_ratio, x0, method='L-BFGS-B', jac=jac_neg_log_ratio, 
                           bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12})
            if res.success or res.fun < neg_log_ratio(x0):
                x_opt = res.x
                h_opt = np.zeros(N)
                h_opt[support] = x_opt
                h_opt = np.maximum(h_opt, 0)
                s_opt = score(h_opt)
                return h_opt, s_opt
        except Exception:
            pass
            
        # Fallback: uniform
        h_uniform = np.zeros(N)
        for i in support:
            h_uniform[i] = 1.0 / k
        s_uniform = score(h_uniform)
        return h_uniform, s_uniform

    best_h = np.zeros(N)
    best_score = 0.0
    
    # 1. Generate diverse initial supports
    # We want to explore different sizes k
    # Theoretical best is k ~ sqrt(N) for Sidon sets.
    # Let's try k from 5 to 20.
    
    np.random.seed(42)
    random.seed(42)
    
    candidate_supports = []
    
    # Structured seeds
    # Small Sidon-like sets
    candidate_supports.append([0, 1, 4, 10, 18, 23, 30]) # k=7
    candidate_supports.append([0, 1, 4, 10, 18, 23, 30, 38]) # k=8
    candidate_supports.append(list(range(5))) # k=5
    
    # Random sets of various sizes
    for k in range(5, 21):
        for _ in range(20):
            S = np.random.choice(N, k, replace=False)
            candidate_supports.append(list(S))
            
    # Process candidates
    # We will evaluate the uniform score first to filter, then local search, then optimize.
    
    processed = []
    for S_init in candidate_supports:
        if time.time() > deadline - 30:
            break
        S = set(S_init)
        # Quick score of uniform vector
        h_uniform = np.zeros(N)
        for i in S:
            h_uniform[i] = 1.0 / len(S)
        s_uniform = score(h_uniform)
        
        # If uniform score is already very good, we might skip local search?
        # No, local search on support might find better M_max.
        
        # Local search to improve M_max
        S_improved = local_search_support(S, max_iters=2000)
        
        # Score uniform on improved support
        h_uniform_improved = np.zeros(N)
        for i in S_improved:
            h_uniform_improved[i] = 1.0 / len(S_improved)
        s_uniform_improved = score(h_uniform_improved)
        
        # Keep track of best uniform score for this support size/type
        # We want to select top K supports for amplitude optimization
        processed.append((s_uniform_improved, S_improved, h_uniform_improved))
        
    # Sort by uniform score (proxy for potential)
    processed.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 10 supports for detailed optimization
    top_supports = processed[:10]
    
    for s_proxy, S, h_proxy in top_supports:
        if time.time() > deadline - 10:
            break
        h_opt, s_opt = optimize_amplitudes(S)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt
            
    # If we haven't found anything better than 1.0, return the single spike
    if best_score <= 1.0:
        h = [0.0] * N
        h[N // 2] = 1.0
        return h
        
    return list(best_h)

# EVOLVE-BLOCK-END
