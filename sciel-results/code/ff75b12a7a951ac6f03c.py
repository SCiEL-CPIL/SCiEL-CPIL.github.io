import numpy as np
from scipy.optimize import minimize
import time
import warnings

N = 50

# EVOLVE-BLOCK-START
def construct():
    DEADLINE = time.time() + 140  # 140s out of 150s limit for safety
    
    def score(h):
        h = np.array(h)
        if np.any(h < 0) or np.sum(h) == 0:
            return 0.0
        h_norm = h / np.sum(h)
        l2_sq = np.sum(h_norm ** 2)
        conv = np.convolve(h_norm, h_norm, mode='full')
        max_conv = np.max(conv)
        if max_conv == 0:
            return 0.0
        return l2_sq / max_conv

    def smooth_score(h, temp=1e-4):
        """Smooth surrogate using log-sum-exp for the max in autoconvolution."""
        h = np.array(h)
        if np.any(h < 0) or np.sum(h) == 0:
            return 0.0
        h_norm = h / np.sum(h)
        l2_sq = np.sum(h_norm ** 2)
        conv = np.convolve(h_norm, h_norm, mode='full')
        # Use log-sum-exp for smooth max
        m = np.max(conv)
        if m <= 0:
            return 0.0
        # Soft max: m + temp * log(sum(exp((conv - m)/temp)))
        logsumexp = m + temp * np.log(np.sum(np.exp((conv - m) / temp)))
        if logsumexp <= 0:
            return 0.0
        return l2_sq / logsumexp

    # Initial guesses
    best_h = np.zeros(N)
    best_score = 0.0
    
    candidates = []
    
    # Candidate 1: [2/3, 1/3, 0...]
    h1 = np.zeros(N)
    h1[0] = 2/3
    h1[1] = 1/3
    candidates.append(h1)
    
    # Candidate 2: [2/3, 1/3, 1/4, 0...]
    h2 = np.zeros(N)
    h2[0] = 2/3
    h2[1] = 1/3
    h2[2] = 1/4
    h2 = h2 / np.sum(h2)
    candidates.append(h2)
    
    # Candidate 3: [1, 0.5, 0.25, 0.125, ...] geometric
    h3 = np.zeros(N)
    val = 1.0
    for i in range(min(10, N)):
        h3[i] = val
        val *= 0.5
    h3 = h3 / np.sum(h3)
    candidates.append(h3)
    
    # Candidate 4: [0.5, 0.5, 0, ...]
    h4 = np.zeros(N)
    h4[0] = 0.5
    h4[1] = 0.5
    candidates.append(h4)
    
    # Candidate 5: [0.4, 0.3, 0.2, 0.1, 0, ...]
    h5 = np.zeros(N)
    h5[0] = 0.4
    h5[1] = 0.3
    h5[2] = 0.2
    h5[3] = 0.1
    candidates.append(h5)
    
    # Candidate 6: [0.6, 0.2, 0.1, 0.1, 0, ...]
    h6 = np.zeros(N)
    h6[0] = 0.6
    h6[1] = 0.2
    h6[2] = 0.1
    h6[3] = 0.1
    candidates.append(h6)
    
    # Candidate 7: [0.7, 0.15, 0.1, 0.05, 0, ...]
    h7 = np.zeros(N)
    h7[0] = 0.7
    h7[1] = 0.15
    h7[2] = 0.1
    h7[3] = 0.05
    candidates.append(h7)
    
    # Evaluate all candidates
    for h_c in candidates:
        s = score(h_c)
        if s > best_score:
            best_score = s
            best_h = h_c.copy()
    
    # Objective function using smooth surrogate for optimization
    def objective_smooth(x, temp=1e-4):
        h_full = np.zeros(N)
        h_full[:len(x)] = np.abs(x)
        if np.sum(h_full) == 0:
            return 0.0
        return -smooth_score(h_full, temp)
    
    def objective_exact(x):
        h_full = np.zeros(N)
        h_full[:len(x)] = np.abs(x)
        if np.sum(h_full) == 0:
            return 0.0
        return -score(h_full)
    
    # Optimize on first 12 dimensions (more flexible support)
    opt_dim = 12
    start_time = time.time()
    
    # Phase 1: Coarse optimization with smooth objective
    x0 = best_h[:opt_dim].copy()
    try:
        if time.time() < DEADLINE - 30:
            res = minimize(objective_smooth, x0, method='Nelder-Mead', 
                          options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.success:
                h_opt = np.zeros(N)
                h_opt[:len(res.x)] = np.abs(res.x)
                s_opt = score(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt.copy()
    except Exception:
        pass
    
    # Phase 2: Random restarts with smooth objective
    np.random.seed(42)
    n_restarts = 10
    for i in range(n_restarts):
        if time.time() > DEADLINE - 20:
            break
        # Random sparse vector with small support
        support_size = np.random.randint(2, 7)
        indices = np.random.choice(opt_dim, support_size, replace=False)
        indices.sort()
        h_rand = np.zeros(opt_dim)
        h_rand[indices] = np.random.rand(support_size)
        h_rand = h_rand / np.sum(h_rand)
        
        try:
            res = minimize(objective_smooth, h_rand, method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-7})
            if res.success:
                h_opt = np.zeros(N)
                h_opt[:len(res.x)] = np.abs(res.x)
                s_opt = score(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Phase 3: High-precision polish with exact objective
    if time.time() < DEADLINE - 10:
        x0_polish = best_h[:opt_dim].copy()
        try:
            res = minimize(objective_exact, x0_polish, method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.success:
                h_opt = np.zeros(N)
                h_opt[:len(res.x)] = np.abs(res.x)
                s_opt = score(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Phase 4: Try even tighter polish if time remains
    if time.time() < DEADLINE - 5:
        x0_final = best_h[:opt_dim].copy()
        try:
            res = minimize(objective_exact, x0_final, method='Nelder-Mead', 
                          options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.success:
                h_opt = np.zeros(N)
                h_opt[:len(res.x)] = np.abs(res.x)
                s_opt = score(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final check and return
    best_h = np.abs(best_h)
    if np.sum(best_h) == 0:
        best_h[0] = 1.0
    else:
        best_h = best_h / np.sum(best_h)
        
    return best_h
# EVOLVE-BLOCK-END
