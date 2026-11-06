"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill

    # Precompute all arithmetic progressions as index arrays
    # An AP is (start, step) -> indices start, start+step, start+2*step, ... < n
    # We need the max |sum| over ALL APs.
    # Total number of APs is roughly sum over step of (n - step + 1) starts... 
    # Actually for each step s, starts are 0..s-1 (mod s) but within [0, n).
    # Number of APs: for step s, there are s possible residues, but we iterate start in 0..n-1.
    # Let's just list them.
    
    ap_indices = []
    for step in range(1, n + 1):
        for start in range(n):
            indices = list(range(start, n, step))
            if len(indices) >= 1:
                ap_indices.append(np.array(indices, dtype=np.int32))
    
    # Convert to a matrix for fast dot products? 
    # Number of APs: 
    # step=1: 256 APs (lengths 256..1)
    # step=2: 256 APs
    # ...
    # Total is sum_{s=1}^{256} 256 = 65536 APs.
    # Matrix would be 65536 x 256. That's 16M entries. int8 is 16MB. Feasible.
    # But constructing it takes time. Let's try a different approach:
    # Instead of a matrix, we can use the fact that flipping one sign x_i changes the sum
    # of every AP containing i by +/- 2*x_i (which is +/- 2).
    
    # Strategy: 
    # 1. Start with a good random initialization.
    # 2. Use local search (flip one bit) with a fast objective evaluation.
    # 3. To make objective evaluation fast, we can precompute for each position i, 
    #    the list of APs that contain i. But that's also large.
    
    # Alternative: 
    # The objective is max |sum(AP)|. 
    # If we maintain the current sum for each AP, flipping bit i changes all sums 
    # of APs containing i by +/- 2.
    # There are ~65k APs. Updating them all on each flip is O(65k).
    # A full local search pass (256 flips) would be 256 * 65k = 16M ops. Very fast.
    # We can do many passes.
    
    # Let's precompute: for each position i, which APs contain it?
    # ap_to_pos: list of lists. ap_to_pos[i] = list of AP indices containing i.
    # And ap_sums: array of current sums for each AP.
    
    # Precomputation:
    num_aps = len(ap_indices)
    ap_to_pos = [[] for _ in range(n)]
    for ap_idx, indices in enumerate(ap_indices):
        for pos in indices:
            ap_to_pos[pos].append(ap_idx)
    
    # Initialize signs
    rng = np.random.RandomState(42)
    signs = rng.choice([-1, 1], size=n).astype(np.int8)
    
    # Initialize AP sums
    ap_sums = np.zeros(num_aps, dtype=np.int16)
    for ap_idx, indices in enumerate(ap_indices):
        ap_sums[ap_idx] = np.sum(signs[indices])
    
    current_max = np.max(np.abs(ap_sums))
    best_signs = signs.copy()
    best_max = current_max
    
    # Local search
    # We will iterate: for each position i, try flipping it.
    # Flip i: 
    #   delta = 2 * signs[i]  (since new sign is -signs[i], sum changes by -2*signs[i])
    #   Wait: sum_new = sum_old - 2*signs[i] for APs containing i.
    #   Actually, if we flip signs[i] to -signs[i], the change in sum is -2*signs[i].
    #   Let's define: change = -2 * signs[i].
    #   For each AP containing i: ap_sums[ap] += change.
    #   We need to track the new max |ap_sums|.
    #   Instead of recomputing max over 65k elements each time, we can keep a histogram 
    #   or just recompute max if it's fast enough. 65k max is very fast in numpy.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            
            # Try flipping sign at i
            old_sign = signs[i]
            new_sign = -old_sign
            change = -2 * old_sign  # This is the change to add to AP sums
            
            # Apply flip
            signs[i] = new_sign
            affected_aps = ap_to_pos[i]
            ap_sums[affected_aps] += change
            
            # Compute new max
            new_max = np.max(np.abs(ap_sums))
            
            if new_max < current_max:
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
                improved = True
            else:
                # Revert
                signs[i] = old_sign
                ap_sums[affected_aps] -= change
    
    # Final check
    if best_max < current_max:
        # Restore best signs (though we already track best_signs)
        pass
    
    return best_signs.tolist()
# EVOLVE-BLOCK-END
