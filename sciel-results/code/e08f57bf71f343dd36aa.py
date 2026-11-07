"""
Erdos minimum overlap: minimize max(correlate(h, 1-h, full))/N*2.

Strategy: Binary Discrete Simulated Annealing.
1. Reformulate h as a binary vector b in {0, 1}^N with sum(b) = N/2 = 50.
   This is equivalent to h being 0 or 1, but since the constraint is sum(h)=N/2,
   and h in [0,1], the optimal solution for minimizing max correlation 
   in many combinatorial settings is binary (or close to it).
   Actually, the problem allows h in [0,1]. However, the objective 
   max(correlate(h, 1-h)) is convex in h? No, it's the max of quadratic forms.
   But let's look at the structure.
   If we restrict h to {0, 1}, then 1-h is also {0, 1}.
   correlate(h, 1-h) at lag k is the number of indices i such that h_i=1 and h_{i+k}=0.
   We want to minimize the maximum number of such "1 followed by 0" pairs over all lags.
   This is equivalent to minimizing the maximum number of transitions from 1 to 0 
   in any cyclic shift (if we consider periodicity) or linear shift.
   
   For a binary sequence with 50 ones and 50 zeros, the number of 1->0 transitions 
   in the linear sequence is at most 50 (if they alternate perfectly, it's 50 
   transitions in a linear sense? No. 
   Let's check: 1010... has 50 ones and 50 zeros.
   Lag 0: sum h_i(1-h_i) = 0.
   Lag 1: h_i=1, h_{i+1}=0. In 1010..., every 1 is followed by 0. There are 50 such pairs.
   Lag 2: h_i=1, h_{i+2}=1. h_i(1-h_{i+2}) = 0.
   So max correlation is 50. Value = 50/100 * 2 = 1.0.
   
   Can we do better than 1.0?
   If we use a different binary pattern, e.g., 11001100...
   Lag 1: 1->0 transitions. 1100 has one 1->0 at index 1->2. 
   Total 1->0 transitions in 11001100... (50 ones, 50 zeros)
   Each block 1100 has one 1->0. There are 25 blocks. So 25 transitions.
   Lag 2: 1->0? 1 at 0, 0 at 2? No, h_0=1, h_2=0. Yes.
   Let's count carefully.
   Sequence: 1, 1, 0, 0, 1, 1, 0, 0 ...
   Lag 1: pairs (1,0) at (1,2), (5,6), ... -> 25 pairs.
   Lag 2: pairs (1,0) at (0,2), (4,6)? No, (0,2) is (1,0). (1,3) is (1,0).
   Indices i where h_i=1 and h_{i+2}=0.
   i=0: 1,0 (ok)
   i=1: 1,0 (ok)
   i=4: 1,0 (ok)
   i=5: 1,0 (ok)
   So 50 pairs for lag 2?
   Wait, h = [1,1,0,0, 1,1,0,0 ...]
   h_0=1, h_2=0.
   h_1=1, h_3=0.
   h_4=1, h_6=0.
   h_5=1, h_7=0.
   Yes, 50 pairs.
   Lag 3: h_i=1, h_{i+3}=0?
   i=0: 1,0 (h_3=0). Yes.
   i=1: 1,1 (h_4=1). No.
   i=4: 1,0. Yes.
   i=5: 1,1. No.
   So 25 pairs.
   Lag 4: h_i=1, h_{i+4}=1. 0 pairs.
   
   So max is 50. Value 1.0.
   
   Is it possible to get max < 50?
   The sum of correlations over all lags is related to the total number of 1s and 0s.
   Sum_{k=0}^{N-1} c_k = Sum_{i} h_i (1 - h_i) * (N - something)?
   Actually, Sum_{k} correlate(h, 1-h)_k = Sum_{i,j} h_i (1-h_j) delta_{j, i+k} ...
   Let's use the property: Sum_{k} c_k = Sum_{i=0}^{N-1} h_i * (1 - Sum_{j=0}^{N-1-i} h_{i+k} ... no.
   
   Let's just run Simulated Annealing on the binary space.
   If binary doesn't beat 1.0, we can try a "soft" binary (values close to 0 and 1) 
   or just stick to binary. The current best is 1.29, so 1.0 is a huge improvement.
   Even if we can't reach 1.0, SA on binary should find good patterns.
   
   We will also allow a "refinement" step where we allow values to move slightly 
   away from 0 and 1 if it helps reduce the max correlation, but start with binary.
"""

import numpy as np
import time

N = 100
DEADLINE = 135.0
_START = time.time()


def remaining_time():
    return DEADLINE - (time.time() - _START)


def compute_value(h):
    """Compute the raw value (lower is better)."""
    if not np.all(np.isfinite(h)):
        return float('inf')
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0


def construct():
    """
    Strategy: Binary Simulated Annealing.
    1. Initialize h as a random binary vector with 50 ones and 50 zeros.
    2. Use Simulated Annealing to minimize max(correlate(h, 1-h)).
    3. Moves: Swap a 1 and a 0 in the vector.
    4. After SA, perform a local search with small continuous perturbations 
       to see if "softening" the binary values helps.
    """
    
    # --- Phase 1: Binary Simulated Annealing ---
    
    # Initial state: Random permutation of 50 ones and 50 zeros
    h = np.zeros(N)
    ones_indices = np.random.choice(N, N // 2, replace=False)
    h[ones_indices] = 1.0
    
    current_val = compute_value(h)
    best_h = h.copy()
    best_val = current_val
    
    # SA Parameters
    T = 1.0
    T_min = 1e-5
    cooling_rate = 0.999
    
    # We want to minimize the max correlation.
    # The objective is discrete and can have plateaus.
    
    while T > T_min and remaining_time() > 10:
        # Generate a move: pick two indices, one with 1, one with 0, and swap
        ones = np.where(h == 1.0)[0]
        zeros = np.where(h == 0.0)[0]
        
        if len(ones) == 0 or len(zeros) == 0:
            break
            
        i = ones[np.random.randint(len(ones))]
        j = zeros[np.random.randint(len(zeros))]
        
        # Apply move
        h[i], h[j] = h[j], h[i]
        new_val = compute_value(h)
        
        delta = new_val - current_val
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            # Revert
            h[i], h[j] = h[j], h[i]
            
        T *= cooling_rate
        
        # To speed up, we can cool faster if no improvement for a while
        # But simple geometric cooling is fine for 100 steps.
        
    # --- Phase 2: Local Refinement (Continuous) ---
    # Try to nudge the binary values slightly to reduce the max correlation further.
    # This is a local search on the continuous domain, starting from the best binary solution.
    
    h_ref = best_h.copy()
    val_ref = best_val
    
    # Use a simple gradient-free local search or small random perturbations
    # Since the landscape is likely rugged, small random perturbations with projection 
    # to [0,1] and sum constraint might help.
    
    # We will use a "shake" approach: add small noise, project, and keep if better.
    noise_scale = 0.1
    for _ in range(1000):
        if remaining_time() < 5:
            break
            
        h_noisy = h_ref + np.random.randn(N) * noise_scale
        # Project to [0, 1]
        h_noisy = np.clip(h_noisy, 0.0, 1.0)
        # Adjust sum to N/2
        target = N / 2.0
        diff = target - h_noisy.sum()
        h_noisy += diff / N
        h_noisy = np.clip(h_noisy, 0.0, 1.0)
        
        # One more projection step to be safe
        diff = target - h_noisy.sum()
        h_noisy += diff / N
        h_noisy = np.clip(h_noisy, 0.0, 1.0)
        
        val_new = compute_value(h_noisy)
        
        if val_new < val_ref:
            val_ref = val_new
            h_ref = h_noisy.copy()
            best_val = val_ref
            best_h = h_ref.copy()
        else:
            # Decay noise scale
            noise_scale *= 0.99
            
    # Final check and return
    best_h = np.clip(best_h, 0.0, 1.0)
    # Ensure sum is exactly N/2 within tolerance
    diff = N / 2.0 - best_h.sum()
    best_h += diff / N
    best_h = np.clip(best_h, 0.0, 1.0)
    
    # Final sanity check
    assert np.all(np.isfinite(best_h))
    assert np.all(best_h >= -1e-10) and np.all(best_h <= 1.0 + 1e-10)
    assert abs(best_h.sum() - N / 2.0) < 1e-9 * N
    
    return best_h


# EVOLVE-BLOCK-START
# (construct is already defined above)
# EVOLVE-BLOCK-END
