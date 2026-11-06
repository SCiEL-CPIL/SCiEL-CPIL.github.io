import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We use 145s.
    deadline = start_time + 145.0
    
    N = 256
    
    # 1. Precompute all 4-term arithmetic progressions (0-indexed)
    # A 4-AP is (a, a+d, a+2d, a+3d) with 0 <= a, a+3d < N
    # d >= 1
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(0, max_a):
            ap_list.append((a, a+d, a+2*d, a+3*d))
            
    if not ap_list:
        return [0] * N
        
    aps = np.array(ap_list, dtype=np.int32)
    num_aps = len(aps)
    
    # 2. Precompute adjacency: for each index i, which APs does it belong to?
    # We need this to calculate delta score efficiently.
    # However, for vectorized greedy search, we can compute the state of ALL APs 
    # and then determine the effect of flipping each bit.
    
    # Let's define the "contribution" of an AP to the score.
    # Score = sum of 1 if mono, 0 else.
    # If we flip index i, only APs containing i change state.
    
    # Strategy: 
    # We maintain the current coloring `c`.
    # We maintain the current score `S`.
    # To find the best flip, we need to know, for each index i, 
    # how many APs containing i are currently mono, and how many would become mono if flipped.
    
    # Let's precompute for each index i, the list of AP indices it participates in.
    # And for each AP, the positions of the 4 elements.
    
    # Actually, a faster way for N=256:
    # We can precompute a matrix M where M[i, j] is 1 if index i is part of AP j.
    # But sparse representation is better.
    
    # Let's stick to the list of lists for APs per index, but optimize the delta calc.
    # Wait, the vectorized approach suggested in the directive implies checking all flips at once.
    # For N=256, we can compute the delta for all 256 indices in parallel using numpy.
    
    # Let's create arrays for the APs
    ap_a = aps[:, 0]
    ap_b = aps[:, 1]
    ap_c = aps[:, 2]
    ap_d = aps[:, 3]
    
    # Helper function to calculate the score and the "potential" delta for all flips
    # This is the core of the speedup.
    
    def get_deltas(c):
        """
        Returns an array `deltas` of length N.
        deltas[i] is the change in score if we flip bit i.
        Negative delta means improvement.
        """
        # Current state of all APs: are they monochromatic?
        # mono[j] is True if AP j is mono.
        
        # We need to know, for each AP j, what happens if we flip i.
        # This is tricky to do in one vectorized pass without loops over APs or indices.
        
        # Alternative: 
        # For each AP j, it is mono if c[a]==c[b]==c[c]==c[d].
        # If we flip i, AP j changes state only if i is in {a,b,c,d}.
        
        # Let's compute the "monochromatic status" of all APs first.
        ca = c[ap_a]
        cb = c[ap_b]
        cc = c[ap_c]
        cd = c[ap_d]
        
        # An AP is mono if all 4 are equal.
        # (ca == cb) & (cb == cc) & (cc == cd)
        is_mono = (ca == cb) & (cb == cc) & (cc == cd)
        
        # Now, for each index i, we want to sum the delta over all APs containing i.
        # delta_i = sum_{j: i in AP_j} (new_mono_j - old_mono_j)
        
        # Let's iterate over the 4 positions. 
        # For position 'a' (index ap_a[j]):
        # If we flip ap_a[j], does AP j become mono or stop being mono?
        
        # This logic is hard to vectorize across all i simultaneously without 
        # constructing a huge sparse matrix or doing 4 passes.
        
        # Given N=256 and ~10k APs, a loop over the 4 positions might be fast enough 
        # if we vectorize the AP dimension.
        
        # Let's compute the delta contribution from position 0 (a), 1 (b), 2 (c), 3 (d).
        
        deltas = np.zeros(N, dtype=np.int32)
        
        # Helper to update deltas for a specific position
        def update_deltas_for_pos(pos_indices, other_indices, is_mono_current, c):
            """
            pos_indices: array of indices in the coloring that correspond to this position in the AP
            other_indices: tuple of 3 arrays of indices for the other positions
            is_mono_current: bool array for each AP
            """
            # If we flip pos_indices[k], the new values for that AP are:
            # new_val = 1 - c[pos_indices[k]]
            # other_vals = c[other_indices]
            
            # The AP is mono after flip if:
            # new_val == other1 == other2 == other3
            
            # Let's check condition for mono after flip.
            # This requires comparing new_val with the 3 others.
            
            # We can compute this vectorized over the APs (dimension num_aps).
            
            # But we need to scatter these results into `deltas` at indices `pos_indices`.
            # np.add.at is the way to do this.
            
            # Calculate if the AP would be mono after flipping pos_indices
            # new_val = 1 - c[pos_indices]
            new_val = 1 - c[pos_indices]
            
            o1 = c[other_indices[0]]
            o2 = c[other_indices[1]]
            o3 = c[other_indices[2]]
            
            # Mono after: new_val == o1 == o2 == o3
            is_mono_after = (new_val == o1) & (o1 == o2) & (o2 == o3)
            
            # Change in mono status for this AP if we flip pos_indices
            # +1 if becomes mono (was not), -1 if becomes non-mono (was mono), 0 otherwise
            # delta_j = is_mono_after - is_mono_current
            
            delta_for_aps = is_mono_after.astype(np.int32) - is_mono_current.astype(np.int32)
            
            # Scatter add to deltas
            np.add.at(deltas, pos_indices, delta_for_aps)
            
        # Position 0: index a
        # Others: b, c, d
        update_deltas_for_pos(ap_a, (ap_b, ap_c, ap_d), is_mono, c)
        
        # Position 1: index b
        # Others: a, c, d
        update_deltas_for_pos(ap_b, (ap_a, ap_c, ap_d), is_mono, c)
        
        # Position 2: index c
        # Others: a, b, d
        update_deltas_for_pos(ap_c, (ap_a, ap_b, ap_d), is_mono, c)
        
        # Position 3: index d
        # Others: a, b, c
        update_deltas_for_pos(ap_d, (ap_a, ap_b, ap_c), is_mono, c)
        
        return deltas, np.sum(is_mono)

    best_score = np.inf
    best_coloring = None
    
    # Multi-Start Local Search
    # We will run multiple independent greedy searches.
    # Each search starts from a random (or slightly structured) initial state.
    # It performs greedy flips (min delta) until no improvement is possible.
    # Then it "kicks" (randomly flips a few bits) to escape local minimum.
    
    num_restarts = 50  # Adjust based on speed
    max_steps_per_restart = 1000
    
    # Use a fixed seed for reproducibility in evaluation? 
    # The task doesn't specify, but standard practice for these tasks is often 
    # to allow randomness or use a fixed seed. 
    # Given "seed/value" score metric, it implies the seed program is the baseline.
    # We should probably use a random seed for the search to explore, 
    # but ensure we don't exceed time.
    
    rng = np.random.default_rng()
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialize coloring
        # Option 1: Pure random
        # Option 2: Pattern-based (e.g., blocks)
        # Random is usually fine for local search.
        c = rng.integers(0, 2, size=N)
        
        current_score = np.sum((c[ap_a] == c[ap_b]) & (c[ap_b] == c[ap_c]) & (c[ap_c] == c[ap_d]))
        
        if current_score < best_score:
            best_score = current_score
            best_coloring = c.copy()
            
        # Greedy Phase
        for step in range(max_steps_per_restart):
            if time.time() > deadline - 2:
                break
                
            deltas, current_score = get_deltas(c)
            
            # Find the index with the minimum delta (most negative is best)
            min_delta = np.min(deltas)
            
            if min_delta >= 0:
                # No improvement possible. Local minimum reached.
                break
                
            # Flip all indices that have this minimum delta? 
            # Or just one? 
            # Flipping all optimal moves simultaneously can be risky if they interact,
            # but often works well. Let's pick the first one or all?
            # Let's pick the first one to be safe, or random among the best.
            # Actually, if multiple bits have the same min delta, flipping them 
            # sequentially might invalidate the delta of the next one.
            # So we must flip one, recalculate, then flip the next.
            
            # Find indices with min_delta
            candidates = np.where(deltas == min_delta)[0]
            idx_to_flip = candidates[rng.integers(0, len(candidates))]
            
            # Flip
            c[idx_to_flip] = 1 - c[idx_to_flip]
            current_score += min_delta
            
            if current_score < best_score:
                best_score = current_score
                best_coloring = c.copy()
                
        # Kick Phase: Randomly perturb to escape local minimum
        # Flip 3-5 random bits
        num_kicks = rng.integers(3, 6)
        kick_indices = rng.choice(N, size=num_kicks, replace=False)
        c[kick_indices] = 1 - c[kick_indices]
        
        # Recalculate score after kick
        current_score = np.sum((c[ap_a] == c[ap_b]) & (c[ap_b] == c[ap_c]) & (c[ap_c] == c[ap_d]))
        
        if current_score < best_score:
            best_score = current_score
            best_coloring = c.copy()

    # Final check
    if best_coloring is None:
        # Fallback
        return [0] * N
        
    return best_coloring.tolist()
# EVOLVE-BLOCK-END
