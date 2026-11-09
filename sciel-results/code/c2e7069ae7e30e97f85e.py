import numpy as np
import time


def heuristics(prizes, weights):
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle potential shape issues - expect (5, 100)
    if prizes.ndim == 1:
        P = prizes
        w = weights[None, :] if weights.ndim == 1 else weights
    else:
        P = prizes.sum(axis=0)
        w = weights
    
    n = P.shape[0]
    
    # Track elapsed time for safety
    if not hasattr(heuristics, "_t0"):
        heuristics._t0 = time.time()
    elapsed = time.time() - heuristics._t0
    deadline = heuristics._t0 + 315.0  # safety margin under 330s kill
    
    # Compute various weight metrics
    wsum = w.sum(axis=0)
    wmax = w.max(axis=0)
    w2 = (w * w).sum(axis=0)
    vol_sum = wsum / 25.0
    vol_max = wmax / 25.0
    eps = 1e-9
    
    def sanitize(h):
        h = np.asarray(h, dtype=float)
        h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        return np.clip(h, 1e-9, 1e6)
    
    # Base heuristic: prize-to-max-weight ratio squared
    base_heuristic = sanitize((P / (wmax + eps)) ** 2)
    
    # If we have time, try to improve via local colony simulation
    if elapsed < deadline - 60.0:
        def colony_sim(h, seed):
            rng = np.random.RandomState(seed)
            n_items = h.shape[0]
            pheromone = np.ones(n_items)
            best_total = 0.0
            
            for _ in range(30):
                # Each ant tries to build a solution
                ant_totals = []
                for _ in range(20):
                    remaining_cap = np.full(5, 25.0)
                    used = np.zeros(n_items, dtype=bool)
                    total_prize = 0.0
                    
                    while True:
                        # Find items that still fit
                        fits = []
                        for i in range(n_items):
                            if not used[i]:
                                if np.all(w[:, i] <= remaining_cap):
                                    fits.append(i)
                        
                        if not fits:
                            break
                        
                        # Compute probabilities
                        probs = np.array([pheromone[i] * h[i] for i in fits])
                        probs = probs / probs.sum()
                        
                        # Select item
                        idx = rng.choice(fits, p=probs)
                        used[idx] = True
                        remaining_cap -= w[:, idx]
                        total_prize += P[idx]
                    
                    ant_totals.append(total_prize)
                
                best_total = max(best_total, max(ant_totals))
                
                # Update pheromone
                pheromone = pheromone * 0.9 + max(ant_totals) / 10.0
            
            return best_total
        
        # Try several heuristic variations
        candidates = [
            base_heuristic,
            sanitize((P / (wsum + eps)) ** 2),
            sanitize((P / (w2 + eps)) ** 2),
            sanitize(P / (vol_sum + eps)),
            sanitize(P / (vol_max + eps)),
            sanitize((P / (wmax + eps)) * (1.0 / (1.0 + vol_sum))),
            sanitize((P / (wmax + eps)) * (1.0 / (1.0 + wsum / 5.0))),
        ]
        
        best_h = base_heuristic
        best_score = -1.0
        
        for h in candidates:
            if time.time() > deadline - 10.0:
                break
            score = colony_sim(h, 42)
            if score > best_score:
                best_score = score
                best_h = h
        
        return best_h
    
    return base_heuristic
