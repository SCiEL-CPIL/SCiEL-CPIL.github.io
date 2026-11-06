import time
import random
import numpy as np

def construct():
    start_time = time.time()
    timeout = 85.0
    
    n = 128
    # Vector of size 129 (indices 0 to 128)
    # v[i] = 1 if i in A
    
    # Target range 128 to 256
    # Convolution result index k corresponds to sum k
    # We care about indices 128 to 256
    
    def get_score_vec(v):
        # v is numpy array of 0/1
        # conv = v * v
        conv = np.convolve(v, v, mode='full')
        # conv[t] is multiplicity of sum t
        # We only care about t in [128, 256]
        target_counts = conv[128:257]
        
        # Check coverage
        if np.any(target_counts == 0):
            # Count uncovered
            uncovered = np.sum(target_counts == 0)
            return 10000 + uncovered
        
        max_mult = np.max(target_counts)
        sum_sq = np.sum(target_counts ** 2)
        
        return max_mult * 100 + sum_sq

    # Initial A
    # Seed
    v_seed = np.zeros(129, dtype=int)
    v_seed[64:129] = 1
    best_v = v_seed.copy()
    best_score = get_score_vec(best_v)
    
    # Random inits
    for _ in range(10):
        size = random.randint(20, 35)
        indices = random.sample(range(129), size)
        v_rand = np.zeros(129, dtype=int)
        v_rand[indices] = 1
        s = get_score_vec(v_rand)
        if s < best_score:
            best_score = s
            best_v = v_rand.copy()
            
    current_v = best_v.copy()
    current_score = best_score
    
    # Local Search with Numpy
    # Toggle bit i
    # This requires recalculating conv.
    # np.convolve is fast, but 10,000 calls might be slow?
    # Let's check. 129 length. Very fast.
    
    no_improve = 0
    max_no_improve = 1000
    
    while time.time() - start_time < timeout:
        improved = False
        i = random.randint(0, 128)
        
        # Toggle
        new_v = current_v.copy()
        new_v[i] = 1 - new_v[i]
        
        new_score = get_score_vec(new_v)
        
        if new_score < current_score:
            current_v = new_v
            current_score = new_score
            improved = True
            no_improve = 0
            if current_score < best_score:
                best_score = current_score
                best_v = current_v.copy()
        else:
            no_improve += 1
            
        if no_improve > max_no_improve:
            # Random restart/kick
            for _ in range(random.randint(1, 5)):
                k = random.randint(0, 128)
                current_v[k] = 1 - current_v[k]
            current_score = get_score_vec(current_v)
            no_improve = 0
            
    # Extract set from best_v
    A = [i for i in range(129) if best_v[i] == 1]
    
    # Final validation
    # Just in case
    if not A:
        return list(range(64, 129))
        
    return A
