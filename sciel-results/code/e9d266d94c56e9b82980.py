import numpy as np
import time

def heuristics(prizes, weights):
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Ensure correct shape handling
    # Task says shape (5, 100)
    if prizes.ndim == 1:
        P = prizes
        w = weights[None, :] if weights.ndim == 1 else weights
        m = w.shape[0]
        n = P.shape[0]
    else:
        # Sum prizes across constraints to get total prize per item
        P = prizes.sum(axis=0)
        w = weights
        m = w.shape[0]
        n = P.shape[0]
        
    # Sanitize inputs
    P = np.where(np.isfinite(P) & (P > 0), P, 1e-9)
    w = np.where(np.isfinite(w) & (w > 0), w, 1e-9)
    
    # Time guard
    start_time = time.time()
    deadline = start_time + 290.0 # 10s safety margin
    
    # Precompute weight metrics
    w_max = w.max(axis=0)
    w_sum = w.sum(axis=0)
    w_mean = w.mean(axis=0)
    
    # Candidates
    # Using lambdas to delay computation
    candidates = [
        lambda: P / (1.0 + w_max),
        lambda: P / (1.0 + w_max)**2,
        lambda: P / (1.0 + w_sum),
        lambda: P / (1.0 + w_sum)**2,
        lambda: P / (w_max + 1e-9),
        lambda: P / (w_sum + 1e-9),
        lambda: P * np.exp(-w_max),
        lambda: P * np.exp(-w_sum),
        lambda: P / (1.0 + 0.5 * w_max),
        lambda: P / (1.0 + 2.0 * w_max),
        lambda: P / (1.0 + w_mean),
        lambda: P / (1.0 + w_max / 25.0),
        lambda: P / (1.0 + w_sum / (25.0 * m)),
        lambda: np.sqrt(P) / (1.0 + w_max),
        lambda: P**0.5 / (1.0 + w_sum),
    ]
    
    def fast_colony_sim(h, seed):
        rng = np.random.default_rng(seed)
        ph = np.ones(n)
        global_best = 0.0
        
        for _it in range(30):
            iter_best = 0.0
            for _a in range(20):
                rem = np.full(m, 25.0)
                used = np.zeros(n, dtype=bool)
                total = 0.0
                
                # Inner loop
                while True:
                    # Check fits
                    # w is (m, n), rem is (m,)
                    # fits[i] = all(w[j, i] <= rem[j])
                    fits = np.all(w <= rem[:, None], axis=0)
                    eligible = fits & ~used
                    if not np.any(eligible):
                        break
                    
                    p = ph * h
                    p = np.where(eligible, p, 0.0)
                    p_sum = p.sum()
                    if p_sum <= 0:
                        break
                    
                    r = rng.random() * p_sum
                    idx = np.searchsorted(p.cumsum(), r)
                    if idx >= n:
                        idx = n - 1
                        
                    used[idx] = True
                    rem -= w[:, idx]
                    total += P[idx]
                    
                if total > iter_best:
                    iter_best = total
                if total > global_best:
                    global_best = total
                    
            ph = ph * 0.9 + iter_best / 10.0
            
        return global_best

    best_h = candidates[0]()
    best_score = -1.0
    seeds = [0, 1, 2, 3, 4]
    
    for i, gen in enumerate(candidates):
        if time.time() > deadline:
            break
            
        h = gen()
        # Sanitize h
        h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        h = np.clip(h, 1e-9, 1e6)
        
        score_sum = 0.0
        for seed in seeds:
            if time.time() > deadline:
                break
            score_sum += fast_colony_sim(h, seed)
            
        avg_score = score_sum / len(seeds)
        
        if avg_score > best_score:
            best_score = avg_score
            best_h = h
            
    return best_h
