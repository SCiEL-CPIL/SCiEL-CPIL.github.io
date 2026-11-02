"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_overlap_max(h):
    """
    Compute the maximum overlap of h with shifted 1-h.
    h: array of length N, sum is N/2.
    Overlap I(s) = sum_i h_i * (1 - h_{i-s}) * (2/N)
    We want max over s.
    """
    h_comp = 1.0 - h
    # Cross correlation of h and h_comp
    # C[k] = sum_i h[i] * h_comp[(i-k) % N]
    # This is ifft(fft(h) * conj(fft(h_comp)))
    
    H = np.fft.fft(h)
    Hc = np.fft.fft(h_comp)
    corr = np.fft.ifft(H * np.conj(Hc)).real
    
    # The overlap integral for shift k is corr[k] * (2/N)
    # We want the maximum of this.
    max_overlap = np.max(corr) * (2.0 / N)
    return max_overlap

def _local_search(h, deadline, rng):
    """
    Perform local search using cyclic shift-exchange moves.
    Moves: 
    1. Shift a contiguous block of indices by +1 or -1.
    2. Swap two individual indices (special case of block size 1).
    
    Constraint: Sum must remain N/2.
    Since we only permute existing values, the sum is preserved exactly.
    """
    h_curr = h.copy()
    current_score = _compute_overlap_max(h_curr)
    best_h = h_curr.copy()
    best_score = current_score
    
    # Initial randomization to escape local minima if starting from uniform
    # If starting from uniform, all scores are identical, so we need a kick.
    if np.allclose(h_curr, 0.5):
        # Random permutation
        h_curr = np.random.permutation(h_curr)
        current_score = _compute_overlap_max(h_curr)
        best_h = h_curr.copy()
        best_score = current_score

    no_improve_count = 0
    max_no_improve = 500
    temp = 0.01  # Simulated annealing temperature
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Restart temperature or break
            temp *= 0.9
            no_improve_count = 0
            if temp < 1e-6:
                break
        
        # Generate a random move
        # Choose a random start index and length
        start = rng.randint(0, N - 1)
        length = rng.randint(1, 10)  # Small blocks
        
        # Create the move: shift block [start, start+length) by 1 position
        # We need to define the cyclic nature carefully.
        # Let's try swapping a block with its neighbor.
        # Move: Swap h[start:start+length] with h[start+length:start+2*length] (cyclic)
        # Or simpler: Swap two random indices.
        # Or: Rotate a segment.
        
        # Let's use a "2-opt" style swap for segments.
        # Pick two random segments of equal length? No, just swap two single elements first, 
        # then maybe larger blocks.
        
        # Simplest effective move: Swap two random indices.
        i1 = rng.randint(0, N - 1)
        i2 = rng.randint(0, N - 1)
        
        if i1 == i2:
            continue
            
        h_new = h_curr.copy()
        h_new[i1], h_new[i2] = h_curr[i2], h_curr[i1]
        
        new_score = _compute_overlap_max(h_new)
        
        # Accept if better, or with probability exp(-delta/temp) if worse
        if new_score < current_score:
            h_curr = h_new
            current_score = new_score
            no_improve_count = 0
            
            if new_score < best_score:
                best_score = new_score
                best_h = h_new.copy()
        else:
            delta = current_score - new_score # Positive if worse
            if delta < 0:
                # Should not happen if new_score < current_score handled above
                pass 
            # Accept worse move with probability
            if rng.random() < np.exp(-delta / temp):
                h_curr = h_new
                current_score = new_score
            else:
                no_improve_count += 1
                
        # Occasionally try a larger block swap to escape local minima
        if rng.random() < 0.1:
            start = rng.randint(0, N - 1)
            length = rng.randint(2, 20)
            # Rotate the block [start, start+length) by 1
            # This is a valid permutation.
            h_new = h_curr.copy()
            # Cyclic shift of the segment
            # Extract segment
            # Handle wrap-around
            if start + length <= N:
                seg = h_curr[start:start+length]
                h_new[start:start+length] = np.roll(seg, 1)
            else:
                # Wrap around
                # This is complex, let's just use simple swaps for robustness
                # Or handle wrap:
                len1 = N - start
                len2 = length - len1
                seg1 = h_curr[start:]
                seg2 = h_curr[:len2]
                # Roll combined? 
                # Let's just stick to simple swaps for the main loop and 
                # use a different move type: "Block Swap" where we swap two blocks of same size.
                # For now, fallback to single swap if wrap is complex
                i1 = start
                i2 = rng.randint(0, N - 1)
                h_new = h_curr.copy()
                h_new[i1], h_new[i2] = h_curr[i2], h_curr[i1]
            
            new_score = _compute_overlap_max(h_new)
            if new_score < current_score:
                h_curr = h_new
                current_score = new_score
                no_improve_count = 0
                if new_score < best_score:
                    best_score = new_score
                    best_h = h_new.copy()
            else:
                delta = current_score - new_score
                if rng.random() < np.exp(-delta / temp):
                    h_curr = h_new
                    current_score = new_score
                else:
                    no_improve_count += 1

    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Initialize with a random permutation of 0s and 1s?
    # Sum must be N/2 = 50.
    # So we have 50 zeros and 50 ones.
    # This is a valid step height configuration.
    # The uniform 0.5 is also valid but likely a local minimum for max overlap.
    
    # Let's start with a random binary configuration.
    h_init = np.zeros(N)
    h_init[:N//2] = 1.0
    h_init = np.random.permutation(h_init)
    
    best_h = h_init.copy()
    best_score = _compute_overlap_max(best_h)
    
    # Run multiple restarts with different random seeds for the local search
    # and different initial configurations.
    
    rng = random.Random(42)
    np.random.seed(42)
    
    num_restarts = 20
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Create a new initial configuration
        # Mix of strategies:
        # 1. Random binary
        # 2. Sorted binary (0s then 1s)
        # 3. Alternating (0,1,0,1...)
        # 4. Random continuous values that sum to N/2
        
        if i % 4 == 0:
            h_curr = np.zeros(N)
            h_curr[:N//2] = 1.0
            h_curr = np.random.permutation(h_curr)
        elif i % 4 == 1:
            # Alternating
            h_curr = np.zeros(N)
            h_curr[::2] = 1.0
            if N % 2 != 0:
                # Adjust to ensure sum is N/2
                # N is 100, so sum is 50. Alternating 0,1,0,1... gives 50 ones.
                pass
            else:
                h_curr = np.random.permutation(h_curr)
        elif i % 4 == 2:
            # Sorted
            h_curr = np.zeros(N)
            h_curr[:N//2] = 1.0
        else:
            # Random continuous, projected to sum N/2
            h_curr = np.random.uniform(0, 1, N)
            h_curr *= (N / 2.0) / np.sum(h_curr)
            # Clip and adjust? 
            # For local search, we need to handle the sum constraint.
            # If we use swaps, sum is preserved.
            # But if we start with continuous values, swaps preserve sum.
            # However, the "step height" nature suggests discrete values might be better?
            # The problem says "step heights h on [0,2]". It doesn't restrict to 0/1.
            # But "step heights" often implies piecewise constant.
            # Let's stick to permutations of 0/1 for the main search as it's a strong prior
            # for minimizing overlap (spreading mass out).
            # If continuous is better, the local search on permutations won't find it.
            # But let's try a mix: start with continuous, but only use swaps?
            # Swaps on continuous values just permutes them.
            # So we are effectively searching for the best permutation of a fixed multiset of values.
            # This is a good strategy: fix the values, optimize the arrangement.
            
            # Generate a random multiset of values that sum to N/2
            # E.g., 50 values of 0.5? That's uniform.
            # E.g., 25 values of 0, 25 values of 1, 50 values of 0.5?
            # Let's try: 25 zeros, 25 ones, 50 halfs. Sum = 25*1 + 50*0.5 = 50.
            h_curr = np.zeros(N)
            h_curr[:25] = 0.0
            h_curr[25:50] = 1.0
            h_curr[50:] = 0.5
            h_curr = np.random.permutation(h_curr)
        
        # Run local search
        h_res, score_res = _local_search(h_curr, deadline, rng)
        
        if score_res < best_score:
            best_score = score_res
            best_h = h_res.copy()
            
        # Check time
        if time.time() > deadline - 10:
            break

    return best_h
# EVOLVE-BLOCK-END
