import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    # Safety margin: return 10s before the 150s limit
    deadline = start_time + 140.0
    
    # We search over discrete signals.
    # Let h be composed of integers.
    # To ensure sum(h) != 0 and to handle scale invariance, 
    # we can fix the sum to be a specific non-zero integer, e.g., 1, 
    # or just normalize at the end.
    # However, for the SA, it's better to work with raw integers 
    # and normalize the final candidate.
    
    # Initial state: A random integer signal with non-zero sum.
    # To get a good start, we can try a few structured initializations.
    # 1. All ones (Score 2.0)
    # 2. Blocky: First half 1s, second half -1s (Sum might be 0, need adjustment)
    # 3. Random integers
    
    def compute_score(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return float('inf')
        # Normalize to sum=1 for consistent comparison? 
        # The value is scale invariant: V(ah) = V(h).
        # So we can just compute the raw value.
        c = np.convolve(h, h)
        max_c = np.max(c)
        return abs(2.0 * N * max_c / (s * s))

    # Generate a few diverse starting points
    candidates = []
    
    # 1. Constant 1
    h1 = np.ones(N, dtype=np.int64)
    candidates.append(h1)
    
    # 2. Ramp
    h2 = np.arange(1, N+1, dtype=np.int64)
    candidates.append(h2)
    
    # 3. Blocky: 1 for first 300, -1 for next 100. Sum = 200.
    h3 = np.zeros(N, dtype=np.int64)
    h3[:300] = 1
    h3[300:] = -1
    candidates.append(h3)
    
    # 4. Blocky: 1 for first 200, -1 for next 100, 0 for last 100. Sum = 100.
    h4 = np.zeros(N, dtype=np.int64)
    h4[:200] = 1
    h4[200:300] = -1
    candidates.append(h4)
    
    # 5. Random integers in {-1, 0, 1}
    rng = np.random.default_rng(42)
    h5 = rng.choice([-1, 0, 1], size=N)
    if np.sum(h5) == 0:
        h5[0] = 1
    candidates.append(h5)
    
    best_h = None
    best_score = float('inf')
    
    for h in candidates:
        s = compute_score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Simulated Annealing
    # State: h (integer array)
    # Move: Pick a random index j, change h[j] by +1 or -1 (or set to 0/1/-1)
    # Cost: Score
    # We want to minimize Cost.
    
    h = best_h.copy()
    current_score = best_score
    best_score_sa = current_score
    best_h_sa = h.copy()
    
    T = 1.0 # Initial temperature
    cooling = 0.999
    
    # Precompute convolution cost is O(N^2) ~ 160k ops. 
    # With SA, we can do many iterations.
    # 140 seconds / (time per iteration)
    # Let's estimate time per iteration.
    
    iter_count = 0
    last_check = time.time()
    
    while time.time() < deadline:
        # Check time periodically
        if iter_count % 100 == 0:
            if time.time() > deadline:
                break
            
        # Generate a neighbor
        # Strategy: Change one element randomly
        j = rng.integers(0, N)
        old_val = h[j]
        
        # Perturb: try adding -1, 0, +1 or setting to a random small integer
        # To keep the "blocky" nature, maybe just flip sign or increment/decrement?
        # Let's try: new_val = old_val + rng.choice([-1, 1])
        # Or: new_val = rng.choice([-1, 0, 1])
        # The latter allows "erasing" parts of the signal, which might be good.
        
        new_val = rng.choice([-1, 0, 1])
        if new_val == old_val:
            continue
            
        h[j] = new_val
        
        # Check if sum is zero
        s = np.sum(h)
        if abs(s) < 1e-12:
            # Revert
            h[j] = old_val
            continue
            
        new_score = compute_score(h)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0:
            # Improvement
            current_score = new_score
            if new_score < best_score_sa:
                best_score_sa = new_score
                best_h_sa = h.copy()
        else:
            # Worse, accept with probability exp(-delta/T)
            # Note: scores are around 1-2. Delta might be small.
            # If T is too high, we accept everything. If too low, we get stuck.
            # Scale T relative to typical score differences.
            prob = np.exp(-delta / T)
            if rng.random() < prob:
                current_score = new_score
            else:
                # Revert
                h[j] = old_val
        
        # Cool down
        T *= cooling
        
        # If T is very small, we are effectively doing greedy hill climbing.
        # To escape local minima, we can occasionally reheat or restart.
        if T < 0.01:
            # Reheat
            T = 0.1
            # Or restart from best
            # h = best_h_sa.copy()
            # current_score = best_score_sa
            # T = 0.1
            
        iter_count += 1
        
    # Final check: ensure the best found is valid
    s_best = np.sum(best_h_sa)
    if abs(s_best) < 1e-12:
        # Fallback to seed
        return np.ones(N)
        
    # The evaluator expects a sequence of 400 signed step heights.
    # It computes value = abs(2 * N * max(conv(h,h)) / sum(h)^2).
    # We can return best_h_sa directly. 
    # Normalizing to sum=1 doesn't change the score, but might help numerical stability?
    # No, integer arithmetic is exact. 
    # However, the problem allows "real" heights. 
    # Returning integers is fine.
    
    return best_h_sa

