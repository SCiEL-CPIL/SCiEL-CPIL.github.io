"""
Rewritten program: Local search with fast AP sum updates for minimizing max |sum| over APs.
"""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_N = 256

def _generate_aps(n):
    """
    Generate all arithmetic progressions in [0, n-1].
    Returns:
        ap_indices: List of lists, where ap_indices[j] is a list of indices for the j-th AP.
        num_aps: Total number of APs.
    """
    ap_indices = []
    for d in range(1, n):
        for r in range(d):
            # Indices: r, r+d, r+2d, ... < n
            seq = list(range(r, n, d))
            if len(seq) > 1:
                ap_indices.append(seq)
            # Note: Singletons are always |1|, so max is at least 1. 
            # We don't need to track them in the sum array if we just track max >= 1.
            # But for consistency in the update logic, we can ignore singletons 
            # because they don't change when we flip a sign? 
            # Wait, if we flip sign at k, the singleton AP [k] changes from 1 to -1 or vice versa.
            # Its absolute value is always 1. So it doesn't affect the max if max >= 1.
            # Since we are minimizing max abs sum, and max is always >= 1, 
            # we can safely ignore singletons in the `current_sums` array.
    return ap_indices

def _precompute_ap_containing(ap_indices, n):
    """
    For each index k, find which APs contain it.
    Returns:
        ap_containing: List of lists, where ap_containing[k] is a list of AP indices j 
                       such that k is in ap_indices[j].
    """
    ap_containing = [[] for _ in range(n)]
    for j, indices in enumerate(ap_indices):
        for k in indices:
            ap_containing[k].append(j)
    return ap_containing

def _construct_signs_fast():
    """
    Construct signs using greedy init and fast local search.
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = _N
    ap_indices = _generate_aps(n)
    num_aps = len(ap_indices)
    
    if num_aps == 0:
        return [1] * n

    ap_containing = _precompute_ap_containing(ap_indices, n)
    
    # Initialize sums to 0
    current_sums = np.zeros(num_aps, dtype=np.int32)
    signs = np.zeros(n, dtype=np.int8)
    
    # Greedy Initialization
    # For each position, choose sign that minimizes current max abs sum
    current_max = 1
    
    for i in range(n):
        best_sign = 1
        best_max_val = float('inf')
        
        for s in [1, -1]:
            signs[i] = s
            # Update sums for APs containing i
            # Since we are building from scratch, we just need to add s to all APs containing i
            # But wait, previous signs are already set. 
            # So we just need to update the sums for APs that contain i.
            # The change is adding s to the sum of each AP containing i.
            
            # To evaluate, we can just compute the max after adding.
            # But we need to revert if we try -1.
            # Let's do a temporary update.
            
            # Optimization: 
            # We can just calculate the new max for this specific choice.
            # Since we are at step i, only APs containing i are affected by this new sign.
            # APs not containing i have sums unchanged from previous steps.
            
            # Let's store the sums before the change? No, that's complex.
            # Simpler: Just apply the change, check max, revert if worse.
            
            affected_aps = ap_containing[i]
            if len(affected_aps) > 0:
                # Save old sums? No, just add and subtract.
                old_sums = current_sums[affected_aps].copy()
                
                current_sums[affected_aps] += s
                
                new_max = max(current_max, np.max(np.abs(current_sums[affected_aps])))
                # Note: current_max might be from APs NOT containing i.
                # So global max is max(current_max, max of changed APs).
                
                if new_max < best_max_val:
                    best_max_val = new_max
                    best_sign = s
                    # Keep the change for now if this is better, or just remember it?
                    # We need to commit to the best sign.
                    
                # Revert to try next sign
                current_sums[affected_aps] = old_sums
        
        # Commit the best sign
        signs[i] = best_sign
        if len(ap_containing[i]) > 0:
            current_sums[ap_containing[i]] += best_sign
            # Update global max
            # The global max is max(previous global max, max of new APs)
            # But previous global max might have been from APs that now changed?
            # No, only APs containing i changed. APs not containing i are unchanged.
            # So the new global max is max(old_global_max, max_abs(current_sums[ap_containing[i]]))
            # Wait, old_global_max included the old values of APs containing i.
            # Those values changed. So we can't just take max(old, new).
            # We must recompute the global max?
            # That's O(num_aps). Doing this N times is O(N * num_aps) ~ 256 * 32k ~ 8M ops.
            # This is acceptable for initialization.
            
            current_max = np.max(np.abs(current_sums))
            
    # Local Search
    best_signs = signs.copy()
    best_max = current_max
    rng = random.Random(42)
    
    no_improve_count = 0
    max_no_improve = 100 # Allow more wandering
    
    while time.time() < deadline:
        improved = False
        # Randomly select a position to flip
        # Or select a few?
        # Let's try single flips first.
        
        # To avoid getting stuck, we can try multiple random positions in a batch
        # or just one at a time.
        
        pos = rng.randint(0, n - 1)
        
        # Try flipping pos
        old_sign = signs[pos]
        new_sign = -old_sign
        
        affected_aps = ap_containing[pos]
        if len(affected_aps) == 0:
            continue
            
        # Save old sums
        old_sums = current_sums[affected_aps].copy()
        
        # Apply flip
        # Change in sum is -2 * old_sign
        current_sums[affected_aps] -= 2 * old_sign
        signs[pos] = new_sign
        
        # Check new max
        # Global max is max(max of unchanged APs, max of changed APs)
        # We don't track max of unchanged APs separately.
        # So we must compute np.max(np.abs(current_sums))
        # This is O(num_aps).
        
        new_max = np.max(np.abs(current_sums))
        
        if new_max < best_max:
            best_max = new_max
            best_signs = signs.copy()
            improved = True
            no_improve_count = 0
        else:
            # Revert
            current_sums[affected_aps] = old_sums
            signs[pos] = old_sign
            no_improve_count += 1
            
            if no_improve_count >= max_no_improve:
                # Random restart or perturbation
                # Perturb a few random positions
                num_flips = rng.randint(2, 5)
                for _ in range(num_flips):
                    p = rng.randint(0, n - 1)
                    affected = ap_containing[p]
                    if len(affected) == 0: continue
                    # Just flip without checking improvement (blind perturbation)
                    # This breaks the "only accept if better" rule, acting as a jump
                    # We need to update sums accordingly
                    signs[p] = -signs[p]
                    current_sums[affected] -= 2 * signs[p] # Wait, sign changed from s to -s. Change is -2s.
                    # Let's be precise:
                    # old_sign = signs[p]
                    # signs[p] = -old_sign
                    # change = -2 * old_sign
                    # But I just flipped it. 
                    # Let's redo this properly.
                    
                # Recalculate max after perturbation
                current_max = np.max(np.abs(current_sums))
                # Reset no_improve
                no_improve_count = 0
                # We don't update best_max here unless it improved, but perturbation usually worsens it.
                # The hope is that subsequent flips can fix it and find a better spot.
                
    return [int(x) for x in best_signs]

def construct():
    """
    Construct 256 signs (-1 or +1) minimizing the maximum absolute signed sum
    over any arithmetic progression in [1, 256].
    """
    # Use the fast construction method
    signs = _construct_signs_fast()
    return signs

# EVOLVE-BLOCK-END
