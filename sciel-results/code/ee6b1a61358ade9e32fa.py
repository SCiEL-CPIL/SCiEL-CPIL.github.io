import numpy as np
import time

def heuristics(prizes, weights):
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Ensure shapes are correct
    if prizes.ndim == 1:
        P = prizes
        w = weights.reshape(5, -1) if weights.ndim == 1 else weights
    else:
        P = prizes.sum(axis=0)
        w = weights
        
    n = P.shape[0]
    C = w.shape[0] # 5
    cap = 25.0
    
    # Time management: 330s total wall-clock, 3 calls expected
    t_start = time.time()
    deadline = t_start + 105.0  # Slightly more aggressive per-call deadline
    
    # Precompute features
    w_sum = w.sum(axis=0)
    w_max = w.max(axis=0)
    w_mean = w.mean(axis=0)
    w_min = w.min(axis=0)
    
    P_max = P.max()
    if P_max == 0:
        return np.ones(n)
    
    P_norm = P / P_max
    
    # Fast low-fidelity ACO simulator for candidate evaluation
    def simulate_aco_fast(h, seed=42, n_iter=10, n_ants=5):
        h = np.asarray(h, dtype=float)
        h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        
        rng = np.random.RandomState(seed)
        pheromone = np.ones(n)
        best_total = 0.0
        
        for it in range(n_iter):
            ant_totals = np.zeros(n_ants)
            
            for a in range(n_ants):
                remaining = np.full(C, cap)
                used = np.zeros(n, dtype=bool)
                total = 0.0
                
                while True:
                    fits_mask = np.all(w <= remaining[:, None], axis=0) & ~used
                    if not fits_mask.any():
                        break
                    
                    fit_indices = np.where(fits_mask)[0]
                    probs = pheromone[fit_indices] * h[fit_indices]
                    prob_sum = probs.sum()
                    if prob_sum == 0:
                        break
                    probs /= prob_sum
                    
                    idx = rng.choice(fit_indices, p=probs)
                    used[idx] = True
                    remaining -= w[:, idx]
                    total += P[idx]
                
                ant_totals[a] = total
            
            iter_best = ant_totals.max()
            if iter_best > best_total:
                best_total = iter_best
            
            pheromone = pheromone * 0.9 + iter_best / 10.0
        
        return best_total
    
    # Full-fidelity ACO simulator for final verification
    def simulate_aco_full(h, seed=42, n_iter=30, n_ants=20):
        h = np.asarray(h, dtype=float)
        h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        
        rng = np.random.RandomState(seed)
        pheromone = np.ones(n)
        best_total = 0.0
        
        for it in range(n_iter):
            ant_totals = np.zeros(n_ants)
            
            for a in range(n_ants):
                remaining = np.full(C, cap)
                used = np.zeros(n, dtype=bool)
                total = 0.0
                
                while True:
                    fits_mask = np.all(w <= remaining[:, None], axis=0) & ~used
                    if not fits_mask.any():
                        break
                    
                    fit_indices = np.where(fits_mask)[0]
                    probs = pheromone[fit_indices] * h[fit_indices]
                    prob_sum = probs.sum()
                    if prob_sum == 0:
                        break
                    probs /= prob_sum
                    
                    idx = rng.choice(fit_indices, p=probs)
                    used[idx] = True
                    remaining -= w[:, idx]
                    total += P[idx]
                
                ant_totals[a] = total
            
            iter_best = ant_totals.max()
            if iter_best > best_total:
                best_total = iter_best
            
            pheromone = pheromone * 0.9 + iter_best / 10.0
        
        return best_total
    
    best_h = None
    best_score = -1.0
    rng_search = np.random.RandomState(12345)
    
    # Candidate 1: Simple ratio
    h_base = P_norm / (w_sum / C + 1e-6)
    h_base = h_base ** 1.0
    score = simulate_aco_fast(h_base)
    if score > best_score:
        best_score = score
        best_h = h_base.copy()
        
    # Expanded search: more diverse metrics and parameter ranges
    metrics = [
        w_sum / C,
        w_max,
        w_mean,
        w_min,
        (w_sum + w_max) / (2 * C),
        w_sum / (w_max + 1e-6),
        w_sum * w_max,
        (w_sum / C) * (w_max / (w_mean + 1e-6))
    ]
    
    for m_idx, metric in enumerate(metrics):
        metric_safe = metric + 1e-6
        metric_norm = metric_safe / (metric_safe.max() + 1e-6)
        
        # Try a grid of alpha values combined with random beta
        for alpha in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
            if time.time() > deadline - 5.0:
                break
                
            for beta in [0.5, 1.0, 1.5, 2.0, 3.0]:
                if time.time() > deadline - 5.0:
                    break
                    
                h_candidate = (P_norm / metric_norm ** beta) ** alpha
                h_candidate = np.clip(h_candidate, 1e-9, 1e6)
                
                score = simulate_aco_fast(h_candidate)
                if score > best_score:
                    best_score = score
                    best_h = h_candidate.copy()
        
        # Random search for this metric
        for _ in range(20):
            if time.time() > deadline - 5.0:
                break
                
            beta = 10 ** rng_search.uniform(-0.5, 2.0)
            alpha = 10 ** rng_search.uniform(-0.5, 1.5)
            
            h_candidate = (P_norm / metric_norm ** beta) ** alpha
            h_candidate = np.clip(h_candidate, 1e-9, 1e6)
            
            score = simulate_aco_fast(h_candidate)
            if score > best_score:
                best_score = score
                best_h = h_candidate.copy()
    
    # Additional diverse candidates: combinations and transformations
    if best_h is not None and time.time() < deadline - 10.0:
        # Try blending with P_norm
        for blend in [0.2, 0.3, 0.5, 0.7, 0.8]:
            h_candidate = blend * best_h + (1 - blend) * P_norm
            h_candidate = np.clip(h_candidate, 1e-9, 1e6)
            
            score = simulate_aco_fast(h_candidate)
            if score > best_score:
                best_score = score
                best_h = h_candidate.copy()
        
        # Try log-transformed version
        h_candidate = np.log1p(best_h)
        h_candidate = np.clip(h_candidate, 1e-9, 1e6)
        
        score = simulate_aco_fast(h_candidate)
        if score > best_score:
            best_score = score
            best_h = h_candidate.copy()
        
        # Try inverse transform
        h_candidate = 1.0 / (1.0 + best_h)
        h_candidate = np.clip(h_candidate, 1e-9, 1e6)
        
        score = simulate_aco_fast(h_candidate)
        if score > best_score:
            best_score = score
            best_h = h_candidate.copy()
    
    # Final high-fidelity verification of the best candidate
    if best_h is not None and time.time() < deadline - 20.0:
        full_score = simulate_aco_full(best_h)
        # If full-fidelity score is significantly worse, fall back to base
        if full_score < best_score * 0.5:
            best_h = P_norm
    
    # Fallback if time ran out or no improvement
    if best_h is None:
        best_h = P_norm
    
    # Ensure valid output
    best_h = np.where(np.isfinite(best_h) & (best_h > 0), best_h, 1e-9)
    
    return best_h
