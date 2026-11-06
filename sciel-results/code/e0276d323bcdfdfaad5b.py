import numpy as np
import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # Precompute all arithmetic progressions as index lists (0-based)
    progs = []
    for a in range(n):
        for s in range(1, n):
            idx = list(range(a, n, s))
            if len(idx) >= 1:
                progs.append(idx)
    
    Np = len(progs)
    
    # Build dense incidence matrix M where M[i, p] = 1 if i in prog p
    M = np.zeros((n, Np), dtype=np.float32)
    for pid, idx in enumerate(progs):
        for i in idx:
            M[i, pid] = 1.0
    
    def compute_sums(signs):
        return M.T @ signs
    
    def max_abs_sum(sums):
        return np.max(np.abs(sums))
    
    def soft_max_abs(sums, beta=10.0):
        abs_sums = np.abs(sums)
        max_val = np.max(abs_sums)
        if max_val == 0:
            return 0.0
        exp_sums = np.exp(beta * (abs_sums - max_val))
        return max_val + np.log(np.sum(exp_sums)) / beta
    
    # Initialize with random signs
    rng = np.random.RandomState(42)
    signs = rng.choice([-1, 1], size=n).astype(np.float32)
    sums = compute_sums(signs)
    
    best_signs = signs.copy()
    best_score = max_abs_sum(sums)
    
    # Phase 1: SA with smooth surrogate (soft max) for faster convergence
    T_start1 = 5.0
    T_end1 = 0.01
    beta = 10.0
    n_iters_phase1 = 50000
    
    current_signs = signs.copy()
    current_sums = sums.copy()
    current_score = best_score
    
    for it in range(n_iters_phase1):
        if time.time() > deadline - 30:
            break
            
        T = T_start1 * (1.0 - it / float(n_iters_phase1)) + T_end1
        
        i = rng.randint(n)
        
        sign_i = current_signs[i]
        delta = -2.0 * sign_i * M[i, :]
        new_sums = current_sums + delta
        
        current_soft = soft_max_abs(current_sums, beta)
        new_soft = soft_max_abs(new_sums, beta)
        
        delta_soft = new_soft - current_soft
        
        if delta_soft <= 0 or rng.rand() < np.exp(-delta_soft / T):
            current_signs[i] = -sign_i
            current_sums = new_sums
            current_score = max_abs_sum(new_sums)
            
            if current_score < best_score:
                best_score = current_score
                best_signs = current_signs.copy()
    
    # Phase 2: Greedy local search polish
    current_signs = best_signs.copy()
    current_sums = compute_sums(current_signs)
    current_score = best_score
    
    improved = True
    while improved:
        if time.time() > deadline - 10:
            break
        
        improved = False
        for i in range(n):
            sign_i = current_signs[i]
            delta = -2.0 * sign_i * M[i, :]
            new_sums = current_sums + delta
            new_score = max_abs_sum(new_sums)
            
            if new_score < current_score:
                current_signs[i] = -sign_i
                current_sums = new_sums
                current_score = new_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_signs = current_signs.copy()
    
    # Phase 3: SA with exact max for fine-tuning
    current_signs = best_signs.copy()
    current_sums = compute_sums(current_signs)
    current_score = best_score
    
    T_start3 = 2.0
    T_end3 = 0.001
    n_iters_phase3 = 30000
    
    for it in range(n_iters_phase3):
        if time.time() > deadline - 5:
            break
            
        T = T_start3 * (1.0 - it / float(n_iters_phase3)) + T_end3
        
        i = rng.randint(n)
        
        sign_i = current_signs[i]
        delta = -2.0 * sign_i * M[i, :]
        new_sums = current_sums + delta
        new_score = max_abs_sum(new_sums)
        
        delta_score = new_score - current_score
        
        if delta_score <= 0 or rng.rand() < np.exp(-delta_score / T):
            current_signs[i] = -sign_i
            current_sums = new_sums
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_signs = current_signs.copy()
    
    # Phase 4: Final greedy polish
    current_signs = best_signs.copy()
    current_sums = compute_sums(current_signs)
    current_score = best_score
    
    improved = True
    while improved:
        if time.time() > deadline - 1:
            break
        
        improved = False
        for i in range(n):
            sign_i = current_signs[i]
            delta = -2.0 * sign_i * M[i, :]
            new_sums = current_sums + delta
            new_score = max_abs_sum(new_sums)
            
            if new_score < current_score:
                current_signs[i] = -sign_i
                current_sums = new_sums
                current_score = new_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_signs = current_signs.copy()
    
    return [int(x) for x in best_signs]
