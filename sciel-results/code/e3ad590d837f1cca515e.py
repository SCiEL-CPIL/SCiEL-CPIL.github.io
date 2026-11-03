import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    def get_overlap(h):
        # h is array of length N
        # Compute linear correlation of h and (1-h)
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        # Exclude shift 0 (index N-1)
        # The max overlap is the maximum of all other shifts
        # Note: The problem asks for "maximum overlap of h with shifted 1-h".
        # This corresponds to max(c) for k != 0.
        c[0] = -np.inf
        c[-1] = -np.inf # Wait, 'full' mode has 2N-1 elements.
        # Index N-1 is shift 0.
        c[N-1] = -np.inf
        return np.max(c)
    
    def valid_h(h):
        return np.all(h >= 0) and np.all(h <= 1) and abs(np.sum(h) - N/2) < 1e-9

    def random_initial():
        # Generate random h with sum N/2
        # Use Dirichlet-like or simple random + normalize + clip
        # Simple: Random uniform, normalize to sum N/2, clip to [0,1]
        # This clipping breaks sum. 
        # Better: Generate N-1 random numbers, sort, use differences? No.
        # Let's use a "random partition" approach or just iterative repair.
        h = np.random.rand(N)
        h *= (N/2) / np.sum(h)
        # This might exceed 1.
        # If any h_i > 1, we need to redistribute.
        # Given N=100 and sum=50, average is 0.5.
        # Probability of any h_i > 1 in uniform random normalized is low but non-zero.
        # Let's just clip and adjust others.
        for _ in range(10):
            mask = h > 1
            if not np.any(mask): break
            excess = np.sum(h[mask] - 1)
            h[mask] = 1
            # Distribute excess to non-masked elements
            non_mask = ~mask
            if np.sum(non_mask) == 0:
                # All are 1? Sum would be N. But sum is N/2. So this shouldn't happen.
                break
            h[non_mask] -= excess / np.sum(non_mask)
            
        mask_low = h < 0
        if np.any(mask_low):
            deficit = np.sum(-h[mask_low])
            h[mask_low] = 0
            non_mask = ~mask_low
            h[non_mask] += deficit / np.sum(non_mask)
            
        # Final exact sum fix
        diff = N/2 - np.sum(h)
        h += diff / N
        h = np.clip(h, 0, 1)
        # One more sum fix
        diff = N/2 - np.sum(h)
        h += diff / N
        return h

    best_h = random_initial()
    best_score = get_overlap(best_h)
    
    # Simulated Annealing
    T_start = 1.0
    T_end = 0.001
    cooling_rate = 0.999
    steps_per_temp = 100
    
    # We can run multiple restarts
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if r == 0:
            h = best_h.copy()
        else:
            h = random_initial()
        
        current_score = get_overlap(h)
        T = T_start
        
        while T > T_end:
            if time.time() - start_time > time_limit:
                break
                
            for _ in range(steps_per_temp):
                if time.time() - start_time > time_limit:
                    break
                    
                # Pick two indices
                i, j = np.random.choice(N, 2, replace=False)
                # Max possible delta
                # h[i] + delta <= 1 => delta <= 1 - h[i]
                # h[j] - delta >= 0 => delta <= h[j]
                max_delta = min(1 - h[i], h[j])
                if max_delta <= 0:
                    continue
                
                # Choose delta from uniform distribution scaled by T?
                # Or fixed small range?
                # Let's use delta proportional to T to allow large moves early, small later.
                # But h is in [0,1].
                delta = np.random.uniform(-max_delta, max_delta)
                
                new_h = h.copy()
                new_h[i] += delta
                new_h[j] -= delta
                
                # Check bounds (should be satisfied by max_delta logic, but floating point)
                if new_h[i] < 0 or new_h[i] > 1 or new_h[j] < 0 or new_h[j] > 1:
                    continue
                
                new_score = get_overlap(new_h)
                delta_E = new_score - current_score
                
                # Accept if better, or with probability exp(-delta_E / T)
                if delta_E < 0:
                    accept = True
                else:
                    accept = np.random.random() < np.exp(-delta_E / T)
                
                if accept:
                    h = new_h
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_h = h.copy()
            
            T *= cooling_rate
            
    # Final projection to ensure exact sum and bounds
    # The moves preserve sum, but floating point errors might accumulate.
    diff = N/2 - np.sum(best_h)
    best_h += diff / N
    best_h = np.clip(best_h, 0, 1)
    
    # One final local polish with small random steps to fix sum/bounds if clip changed sum
    for _ in range(1000):
        s = np.sum(best_h)
        if abs(s - N/2) < 1e-12:
            break
        # Adjust a random element to fix sum
        idx = np.random.randint(N)
        # We need to change best_h[idx] by diff.
        # But we must keep it in [0,1].
        # If diff > 0, we need to increase best_h[idx].
        # If best_h[idx] + diff > 1, we can't do it in one step.
        # This simple fix might fail if sum is way off.
        # However, SA moves preserve sum exactly (h[i]+=d, h[j]-=d).
        # Only initial random or clip breaks it.
        # So best_h from SA should have sum N/2.
        pass
        
    return best_h
# EVOLVE-BLOCK-END
