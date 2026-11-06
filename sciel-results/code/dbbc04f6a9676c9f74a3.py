"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

N = 256
DEADLINE = 85.0  # Safety margin before 90s kill

def _precompute_aps():
    """
    Precompute all arithmetic progressions within [1, N].
    Returns a list of lists of indices (0-indexed).
    """
    aps = []
    # Singletons
    for i in range(N):
        aps.append([i])
    
    # Length >= 2
    # step s, start a, length L
    # indices: a, a+s, ..., a+(L-1)s
    # max index: a + (L-1)s < N
    
    # To be efficient, we iterate by step and start
    # For each step s, max length is roughly N/s
    for s in range(1, N):
        for a in range(N):
            # Calculate max length for this start and step
            # a + (L-1)s < N  =>  (L-1)s < N-a  =>  L-1 < (N-a)/s
            max_l_minus_1 = (N - 1 - a) // s
            if max_l_minus_1 < 1:
                continue
            # Generate all lengths from 2 to max_l_minus_1 + 1
            # Actually, we need ALL APs.
            # An AP is defined by start, step, length.
            # If we include all lengths, we have O(N^3) APs? No.
            # For fixed s, a, the APs are prefixes of the sequence a, a+s, ...
            # So for each (a,s) pair, there are ~N/s APs (lengths 2..max).
            # Total APs approx sum_{s=1}^N (N/s * N/s) ~ N^2 * sum(1/s^2) ~ O(N^2).
            # N=256, N^2 = 65536. Constant factor ~ 1.6. ~100k APs.
            
            seq = []
            idx = a
            while idx < N:
                seq.append(idx)
                idx += s
                if len(seq) > 1:
                    # Add this AP
                    aps.append(seq[:])
            
    return aps

def _compute_max_abs_sum(signs, aps):
    """
    Compute the maximum absolute signed sum over all APs.
    signs: list of +1/-1
    aps: list of index lists
    """
    max_abs = 0
    for ap in aps:
        s = 0
        for idx in ap:
            s += signs[idx]
        abs_s = abs(s)
        if abs_s > max_abs:
            max_abs = abs_s
    return max_abs

def construct():
    start_time = time.time()
    
    # 1. Precompute APs
    aps = _precompute_aps()
    
    # 2. Constructive Greedy with Randomized Lookahead
    # We build the sequence one element at a time.
    # At each step i, we try both +1 and -1.
    # We evaluate the "cost" of each choice based on the current partial sums 
    # of APs that include index i and end at or before i.
    # To keep it fast, we only update sums for APs affected by the new index.
    
    # However, a pure greedy might be too myopic. 
    # Let's use a "Greedy + Random Restart" approach.
    # Strategy:
    # 1. Generate several random initial sequences.
    # 2. For each, perform a fast "Greedy Repair" pass:
    #    Iterate through indices 0..N-1.
    #    For each index, try flipping it. If it reduces the *current* max abs sum 
    #    (or the sum of squares of all AP sums), accept.
    #    This is effectively a very fast hill climb that moves sequentially.
    
    # Better Strategy: 
    # Use the precomputed APs to maintain a vector of current sums for each AP.
    # This allows O(1) update per flip (if we know which APs contain the index).
    # We need a map: index -> list of AP indices containing it.
    
    # Build index -> APs map
    idx_to_aps = [[] for _ in range(N)]
    for ap_idx, ap in enumerate(aps):
        for idx in ap:
            idx_to_aps[idx].append(ap_idx)
    
    # Precompute AP sums for a given sign vector
    # sums[ap_idx] = sum(signs[i] for i in aps[ap_idx])
    
    best_signs = None
    best_score = float('inf')
    
    rng = random.Random(42)
    
    # Run multiple restarts
    num_restarts = 200 # Adjust based on time
    
    for r in range(num_restarts):
        if time.time() - start_time > DEADLINE - 5.0:
            break
            
        # Initialize random signs
        signs = [rng.choice([-1, 1]) for _ in range(N)]
        
        # Compute initial sums
        # sums is a numpy array for speed
        sums = np.zeros(len(aps), dtype=np.int32)
        for ap_idx, ap in enumerate(aps):
            s = 0
            for idx in ap:
                s += signs[idx]
            sums[ap_idx] = s
        
        current_max = np.max(np.abs(sums))
        
        # Local Search: Hill Climbing with random perturbations
        # We want to minimize the max abs sum.
        # Objective: min max(sums^2) is equivalent to min max(|sums|).
        # Let's minimize the sum of squares of the top K largest sums, or just the max.
        # Minimizing the max directly is non-differentiable and hard.
        # Let's minimize the sum of squares of all AP sums. This is a proxy 
        # that generally reduces the max.
        
        # Actually, minimizing the max is better.
        # Let's use a "Greedy Flip" approach:
        # Find the AP with the largest absolute sum.
        # Try flipping a sign within that AP to reduce its sum.
        
        improved = True
        steps = 0
        max_steps = 5000 # Limit steps per restart
        
        while improved and steps < max_steps:
            improved = False
            steps += 1
            
            if time.time() - start_time > DEADLINE - 2.0:
                break
                
            # Find AP with max abs sum
            # This is O(M) per step. M ~ 100k. 
            # 5000 steps * 100k = 500M ops. Too slow in Python.
            # We need to be smarter.
            
            # Alternative: Randomized Local Search
            # Pick a random index, flip it, check if max abs sum decreases.
            # Checking max abs sum is O(M).
            # So 1 flip = O(M).
            # We can do ~ 1000 flips in 5s if M=100k? 
            # 1000 * 100k = 100M ops. In Python, ~1s? Maybe 5s.
            # So ~1000 flips per restart.
            
            # Let's try 100 random flips per restart, evaluating full max each time.
            # This is too slow if we do it naively.
            
            # Optimized Evaluation:
            # Instead of recomputing ALL sums, we only update the sums for APs 
            # containing the flipped index.
            # Then we need to find the new max. 
            # If we maintain a sorted list or a heap of abs sums, we can find max in O(1).
            # But updating the heap when values change is tricky.
            
            # Let's stick to a simpler, faster heuristic:
            # Minimize the sum of squares of all AP sums.
            # This can be updated in O(k) where k is the number of APs containing the index.
            # avg k ~ M/N ~ 100k/256 ~ 400.
            # So 1 flip costs ~400 ops.
            # 10000 flips = 4M ops. Very fast.
            
            # Does minimizing sum of squares minimize the max?
            # Generally yes, it spreads out the error.
            
            # Let's implement Sum of Squares minimization via Hill Climbing.
            
            # Current Sum of Squares
            # We don't need to store it, just compare before/after.
            
            # Precompute: for each index i, list of APs containing i.
            # We already have idx_to_aps.
            
            # Let's do a simple randomized hill climb on Sum of Squares.
            
            # Calculate initial SoS
            sos = np.sum(sums * sums)
            
            # Local Search
            for _ in range(2000): # 2000 flips
                if time.time() - start_time > DEADLINE - 2.0:
                    break
                    
                i = rng.randint(0, N - 1)
                old_sign = signs[i]
                new_sign = -old_sign
                
                # Update sums for APs containing i
                affected_aps = idx_to_aps[i]
                
                # Calculate change in SoS
                # New sum for ap j: sums[j] + 2*new_sign (since we flip from old to new, diff is 2*new_sign)
                # Wait, sum_new = sum_old - old_sign + new_sign = sum_old - 2*old_sign
                # Or sum_old + 2*new_sign (since new = -old)
                
                delta = -2 * old_sign
                
                # Change in SoS = sum( (sums[j] + delta)^2 - sums[j]^2 ) for j in affected
                # = sum( sums[j]^2 + 2*delta*sums[j] + delta^2 - sums[j]^2 )
                # = sum( 2*delta*sums[j] + delta^2 )
                # = 2*delta * sum(sums[j]) + len(affected) * delta^2
                
                # We need sum(sums[j]) for j in affected.
                # We can compute this on the fly.
                
                sum_sums = 0
                for j in affected_aps:
                    sum_sums += sums[j]
                
                delta_sos = 2 * delta * sum_sums + len(affected_aps) * (delta * delta)
                
                if delta_sos < 0:
                    # Accept
                    signs[i] = new_sign
                    for j in affected_aps:
                        sums[j] += delta
                    sos += delta_sos
                    improved = True
                else:
                    # Reject
                    pass
            
            # After local search, evaluate the true objective: Max Abs Sum
            # We have the current `sums` array.
            current_max = np.max(np.abs(sums))
            
            if current_max < best_score:
                best_score = current_max
                best_signs = signs[:]
                
        # End of restart
        
    if best_signs is None:
        # Fallback
        best_signs = [1] * N
        
    return best_signs

# EVOLVE-BLOCK-END
