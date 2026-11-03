import numpy as np
import time
import itertools

N = 50

def _score_raw(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-10):
        return 0.0
    h = np.clip(h, 0, None)
    norm1 = np.sum(h)
    if norm1 < 1e-12:
        return 0.0
    norm2_sq = np.sum(h ** 2)
    # Compute autoconvolution
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv < 1e-12:
        return 0.0
    return norm2_sq / (norm1 * max_conv)

def _greedy_sidon_set(n, max_size=8):
    """Find a Sidon-like set of size up to max_size in {0, ..., n-1}."""
    # A Sidon set has all pairwise sums distinct (except trivial ones)
    # We'll use a greedy approach
    best_set = None
    best_score = -1
    
    for _ in range(20):  # Try multiple random initializations
        # Start with 0
        selected = [0]
        used_sums = {0}
        
        for candidate in range(1, n):
            # Check if adding candidate keeps it Sidon-like
            valid = True
            for existing in selected:
                s = candidate + existing
                if s in used_sums and s != 2 * candidate:
                    valid = False
                    break
                # Also check 2*candidate
                if 2 * candidate in used_sums and 2 * candidate != candidate + existing:
                    valid = False
                    break
            
            if valid:
                selected.append(candidate)
                used_sums.add(candidate)
                for existing in selected[:-1]:
                    used_sums.add(candidate + existing)
                used_sums.add(2 * candidate)
                
                if len(selected) >= max_size:
                    break
        
        if len(selected) > (len(best_set) if best_set else 0):
            best_set = selected[:]
    
    if best_set is None:
        best_set = [0]
    
    return best_set

def _optimize_weights(support, time_budget=10.0):
    """Optimize weights on a given support using coordinate descent."""
    n = len(support)
    h = np.zeros(N)
    h[support] = 1.0 / n  # Start with uniform weights
    
    best_score = _score_raw(h)
    start_time = time.time()
    
    for iteration in range(500):
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        for i in range(n):
            old_val = h[support[i]]
            # Try different values
            for factor in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
                new_val = old_val * factor
                if new_val < 0:
                    continue
                h[support[i]] = new_val
                score = _score_raw(h)
                if score > best_score:
                    best_score = score
                    improved = True
                else:
                    h[support[i]] = old_val
        
        if not improved:
            break
    
    return h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave safety margin
    
    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_score = _score_raw(best_h)
    
    # Try different support sizes
    for support_size in [4, 5, 6, 7, 8, 9, 10]:
        if time.time() > deadline:
            break
        
        # Find a good Sidon-like set
        support = _greedy_sidon_set(N, support_size)
        if len(support) < support_size:
            continue
        
        # Optimize weights on this support
        h, score = _optimize_weights(support, time_budget=5.0)
        
        if score > best_score:
            best_score = score
            best_h = h.copy()
    
    # Final local search
    if time.time() < deadline - 20:
        # Try random perturbations
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            
            h_trial = best_h.copy()
            # Randomly modify a few positions
            for _ in range(3):
                idx = np.random.randint(0, N)
                h_trial[idx] *= np.random.uniform(0.5, 2.0)
            
            score = _score_raw(h_trial)
            if score > best_score:
                best_score = score
                best_h = h_trial.copy()
    
    return list(best_h)

# EVOLVE-BLOCK-END
