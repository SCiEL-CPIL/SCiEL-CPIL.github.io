"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Precompute all arithmetic progressions
    # An AP is defined by (start, step) where terms are start, start+step, ... <= n
    # We store 0-based indices
    ap_indices = []
    for start in range(1, n + 1):
        for step in range(1, n + 1):
            terms = list(range(start - 1, n, step))  # 0-based
            if terms:
                ap_indices.append(terms)
    
    num_aps = len(ap_indices)
    
    # For each position, store which APs contain it
    pos_to_aps = defaultdict(list)
    for ap_id, idx in enumerate(ap_indices):
        for i in idx:
            pos_to_aps[i].append(ap_id)
    
    # Convert to numpy arrays for faster access
    pos_to_aps_arr = [np.array(pos_to_aps[i], dtype=np.int32) for i in range(n)]
    
    # Initialize with a good heuristic: Thue-Morse-like or random
    # Let's try a few different initializations and pick the best
    
    def evaluate(signs):
        """Compute max absolute sum over all APs."""
        ap_sums = np.zeros(num_aps, dtype=np.int32)
        for i in range(n):
            if signs[i] != 0:
                for ap_id in pos_to_aps_arr[i]:
                    ap_sums[ap_id] += signs[i]
        return np.max(np.abs(ap_sums)), ap_sums
    
    # Try multiple random initializations and local search on the best
    best_signs = None
    best_max = float('inf')
    
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Thue-Morse sequence
            signs = np.zeros(n, dtype=np.int8)
            for i in range(n):
                # Count number of 1s in binary representation
                signs[i] = 1 if bin(i).count('1') % 2 == 0 else -1
        elif restart == 1:
            # Alternating
            signs = np.array([1 if i % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
        elif restart == 2:
            # Block pattern
            signs = np.array([1 if (i // 16) % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
        else:
            # Random
            signs = np.array([np.random.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
        
        current_max, ap_sums = evaluate(signs)
        
        # Local search: flip positions that reduce max
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                
                old_sign = signs[i]
                new_sign = -old_sign
                signs[i] = new_sign
                
                # Update ap_sums
                for ap_id in pos_to_aps_arr[i]:
                    ap_sums[ap_id] += 2 * new_sign
                
                new_max = np.max(np.abs(ap_sums))
                
                if new_max < current_max:
                    current_max = new_max
                    improved = True
                else:
                    # Revert
                    signs[i] = old_sign
                    for ap_id in pos_to_aps_arr[i]:
                        ap_sums[ap_id] -= 2 * new_sign
        
        if current_max < best_max:
            best_max = current_max
            best_signs = signs.copy()
    
    # Final refinement with more aggressive search if time permits
    if best_signs is not None and time.time() < deadline - 5:
        signs = best_signs.copy()
        current_max, ap_sums = evaluate(signs)
        
        # Try flipping pairs of positions
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                
                old_i = signs[i]
                old_j = signs[j]
                signs[i] = -old_i
                signs[j] = -old_j
                
                for ap_id in pos_to_aps_arr[i]:
                    ap_sums[ap_id] += 2 * (-old_i)
                for ap_id in pos_to_aps_arr[j]:
                    ap_sums[ap_id] += 2 * (-old_j)
                
                new_max = np.max(np.abs(ap_sums))
                
                if new_max < current_max:
                    current_max = new_max
                    best_signs = signs.copy()
                    best_max = new_max
                else:
                    # Revert
                    signs[i] = old_i
                    signs[j] = old_j
                    for ap_id in pos_to_aps_arr[i]:
                        ap_sums[ap_id] -= 2 * (-old_i)
                    for ap_id in pos_to_aps_arr[j]:
                        ap_sums[ap_id] -= 2 * (-old_j)
    
    if best_signs is None:
        # Fallback to random
        best_signs = np.array([np.random.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
    
    return [int(s) for s in best_signs]
# EVOLVE-BLOCK-END
