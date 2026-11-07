"""Bootstrap portfolio search for Erdos minimum overlap."""
import time
import numpy as np

N = 100
DEADLINE = 150.0

# EVOLVE-BLOCK-START
import numpy as np

def _score(h):
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _is_valid(h):
    if not np.all(np.isfinite(h)):
        return False
    if np.any(h < -1e-9) or np.any(h > 1 + 1e-9):
        return False
    if abs(np.sum(h) - N / 2) > 1e-9 * N:
        return False
    return True

def _fix_sum(h):
    # Adjust h to sum to N/2 while keeping in [0,1]
    target = N / 2
    current = np.sum(h)
    diff = target - current
    if abs(diff) < 1e-12:
        return h
    
    # Distribute diff to elements not at bounds
    # Simple iterative approach
    for _ in range(1000):
        if diff > 0:
            mask = h < 1.0
            if not np.any(mask):
                break
            idx = np.where(mask)[0]
            # Add to random indices
            amt = min(diff, 0.1)
            if amt > 0:
                chosen = np.random.choice(idx, size=min(len(idx), int(amt / 0.01) + 1), replace=False)
                for i in chosen:
                    add = min(amt, 1.0 - h[i])
                    h[i] += add
                    amt -= add
                    if amt < 1e-9:
                        break
        else:
            mask = h > 0.0
            if not np.any(mask):
                break
            idx = np.where(mask)[0]
            amt = min(-diff, 0.1)
            if amt > 0:
                chosen = np.random.choice(idx, size=min(len(idx), int(amt / 0.01) + 1), replace=False)
                for i in chosen:
                    sub = min(amt, h[i])
                    h[i] -= sub
                    amt -= sub
                    if amt < 1e-9:
                        break
        diff = target - np.sum(h)
        if abs(diff) < 1e-9:
            break
            
    # Final exact fix
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        # Put all diff in one element
        if diff > 0:
            idx = np.argmin(h)
            h[idx] += diff
        else:
            idx = np.argmax(h)
            h[idx] += diff
            
    return np.clip(h, 0, 1)

def _hill_climb(h0, max_iters=5000, step_size=0.05):
    h = h0.copy()
    best_score = _score(h)
    current_score = best_score
    rng = np.random.RandomState(42)
    
    for i in range(max_iters):
        # Perturb a few elements
        num_perturb = rng.randint(1, 4)
        indices = rng.choice(N, size=num_perturb, replace=False)
        
        h_new = h.copy()
        for idx in indices:
            h_new[idx] += rng.uniform(-step_size, step_size)
        
        h_new = np.clip(h_new, 0, 1)
        h_new = _fix_sum(h_new)
        
        if not _is_valid(h_new):
            continue
            
        new_score = _score(h_new)
        if new_score < current_score:
            current_score = new_score
            h = h_new
            if current_score < best_score:
                best_score = current_score
        else:
            # Accept with probability if worse (Simulated Annealing style)
            # But simple hill climb is safer for time
            pass
            
    return h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 5.0
    
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    np.random.seed(42)
    
    # Generate initial candidates
    candidates = [best_h]
    
    # Sinusoidal candidates
    for P in [10, 20, 50, 100, 20, 30]:
        for amp in [0.3, 0.4, 0.5]:
            h = 0.5 + amp * np.cos(2 * np.pi * np.arange(N) / P)
            h = np.clip(h, 0, 1)
            h = _fix_sum(h)
            if _is_valid(h):
                candidates.append(h)
                
    # Random continuous
    for _ in range(100):
        h = np.random.uniform(0, 1, N)
        h = _fix_sum(h)
        if _is_valid(h):
            candidates.append(h)
            
    # Evaluate candidates
    for h in candidates:
        if time.time() > deadline:
            break
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
            
    # Hill climb from best few
    # Sort candidates by score
    scored_candidates = [( _score(h), h) for h in candidates if _is_valid(h)]
    scored_candidates.sort(key=lambda x: x[0])
    
    top_candidates = [h for _, h in scored_candidates[:5]]
    
    for h0 in top_candidates:
        if time.time() > deadline - 10:
            break
        h_climbed, s_climbed = _hill_climb(h0, max_iters=2000, step_size=0.05)
        if s_climbed < best_score:
            best_score = s_climbed
            best_h = h_climbed
            
    # Final validation
    if not _is_valid(best_h):
        best_h = np.full(N, 0.5)
        
    return best_h
# EVOLVE-BLOCK-END
