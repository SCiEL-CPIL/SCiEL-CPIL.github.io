"""
Targeted local search: worst-AP-guided perturbation + repair.
Structural change: focus search on the APs that currently violate the bound,
using a different representation and optimization strategy than SA.
"""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

_N = 256

def _generate_aps(n):
    """
    Generate all arithmetic progressions in [0, n-1] with length >= 2.
    Returns list of index lists.
    """
    ap_indices = []
    for d in range(1, n):
        for r in range(d):
            seq = list(range(r, n, d))
            if len(seq) > 1:
                ap_indices.append(seq)
    return ap_indices

def _precompute_ap_containing(ap_indices, n):
    """
    For each index k, find which APs contain it.
    Returns list of lists of AP indices.
    """
    ap_containing = [[] for _ in range(n)]
    for j, indices in enumerate(ap_indices):
        for k in indices:
            ap_containing[k].append(j)
    return ap_containing

def _greedy_init(n):
    """
    Greedy constructive initialization: choose each sign to minimize the
    maximum absolute AP sum incrementally.
    """
    signs = np.zeros(n, dtype=np.int8)
    ap_indices = _generate_aps(n)
    ap_containing = _precompute_ap_containing(ap_indices, n)
    
    # Partial sums: for each AP, track the sum of assigned signs so far
    current_sums = np.zeros(len(ap_indices), dtype=np.int32)
    
    for i in range(n):
        # Try both signs and pick the one that minimizes the resulting max
        best_sign = 1
        best_max_after = float('inf')
        
        for s in [1, -1]:
            # Compute what the max would be if we set signs[i] = s
            # Only APs containing i are affected
            affected = ap_containing[i]
            if len(affected) == 0:
                max_after = int(np.max(np.abs(current_sums))) if len(current_sums) > 0 else 0
            else:
                # Temporarily add s to affected APs
                old_vals = current_sums[affected].copy()
                current_sums[affected] += s
                max_after = int(np.max(np.abs(current_sums)))
                # Revert
                current_sums[affected] = old_vals
            
            if max_after < best_max_after:
                best_max_after = max_after
                best_sign = s
        
        signs[i] = best_sign
        # Update sums
        for ap_idx in ap_containing[i]:
            current_sums[ap_idx] += best_sign
    
    return signs

def _compute_max_abs_sum(current_sums):
    """Compute max absolute sum from the AP sums array."""
    if len(current_sums) == 0:
        return 1
    return int(np.max(np.abs(current_sums)))

def _find_worst_aps(current_sums, num_aps, threshold=None):
    """
    Find APs with |sum| >= threshold (or top few if threshold is None).
    Returns list of AP indices.
    """
    abs_sums = np.abs(current_sums)
    if threshold is not None:
        return np.where(abs_sums >= threshold)[0].tolist()
    else:
        # Return top 10 worst APs
        k = min(10, num_aps)
        idx = np.argpartition(abs_sums, -k)[-k:]
        return idx.tolist()

def _targeted_local_search(signs, ap_indices, ap_containing, current_sums, 
                           start_time, deadline, rng):
    """
    Targeted local search: focus on worst APs, try flips that improve them.
    """
    n = _N
    num_aps = len(ap_indices)
    current_max = _compute_max_abs_sum(current_sums)
    best_signs = signs.copy()
    best_max = current_max
    
    # Phase 1: Pure local search - try single flips, accept only improvements
    # Focus on positions that appear in many high-discrepancy APs
    iteration = 0
    stall_count = 0
    max_stall = 500
    
    while time.time() < deadline:
        iteration += 1
        
        # Find worst APs
        worst_aps = _find_worst_aps(current_sums, num_aps, threshold=current_max)
        
        if len(worst_aps) == 0:
            # No APs at the max level, try to reduce the next level
            worst_aps = _find_worst_aps(current_sums, num_aps, threshold=max(current_max - 1, 1))
        
        if len(worst_aps) == 0:
            break
        
        # Collect positions that appear in worst APs, weighted by frequency
        pos_freq = {}
        for ap_idx in worst_aps:
            for pos in ap_indices[ap_idx]:
                pos_freq[pos] = pos_freq.get(pos, 0) + 1
        
        if not pos_freq:
            break
        
        # Try flipping positions in descending order of frequency
        # Shuffle to avoid getting stuck in a cycle
        positions = list(pos_freq.keys())
        rng.shuffle(positions)
        positions.sort(key=lambda p: pos_freq[p], reverse=True)
        
        improved = False
        for pos in positions:
            if time.time() >= deadline:
                break
            
            old_sign = signs[pos]
            affected = ap_containing[pos]
            if len(affected) == 0:
                continue
            
            old_sums = current_sums[affected].copy()
            signs[pos] = -old_sign
            current_sums[affected] -= 2 * old_sign
            new_max = _compute_max_abs_sum(current_sums)
            
            if new_max < current_max:
                current_max = new_max
                improved = True
                if new_max < best_max:
                    best_max = new_max
                    best_signs = signs.copy()
                stall_count = 0
            else:
                # Revert
                current_sums[affected] = old_sums
                signs[pos] = old_sign
        
        if not improved:
            stall_count += 1
            if stall_count >= max_stall:
                # Perturbation: flip 5-10 random positions to escape local minimum
                num_flips = rng.randint(5, 10)
                flip_positions = rng.sample(range(n), num_flips)
                for pos in flip_positions:
                    old_sign = signs[pos]
                    affected = ap_containing[pos]
                    signs[pos] = -old_sign
                    current_sums[affected] -= 2 * old_sign
                current_max = _compute_max_abs_sum(current_sums)
                stall_count = 0
        
        # If we've reached a good bound, do a final polish
        if current_max <= 2:
            break
    
    # Phase 2: Final polish - exhaustive single-flip check
    for _ in range(1000):
        if time.time() >= deadline:
            break
        pos = rng.randint(0, n - 1)
        old_sign = signs[pos]
        affected = ap_containing[pos]
        if len(affected) == 0:
            continue
        
        old_sums = current_sums[affected].copy()
        signs[pos] = -old_sign
        current_sums[affected] -= 2 * old_sign
        new_max = _compute_max_abs_sum(current_sums)
        
        if new_max < current_max:
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_signs = signs.copy()
        else:
            current_sums[affected] = old_sums
            signs[pos] = old_sign
    
    return [int(x) for x in best_signs], best_max

def _construct_signs_targeted():
    """
    Construct signs using greedy init + targeted local search.
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = _N
    ap_indices = _generate_aps(n)
    num_aps = len(ap_indices)
    
    if num_aps == 0:
        return [1] * n

    ap_containing = _precompute_ap_containing(ap_indices, n)
    
    # Greedy initialization
    signs = _greedy_init(n)
    
    # Compute initial sums
    current_sums = np.zeros(num_aps, dtype=np.int32)
    for i in range(n):
        if len(ap_containing[i]) > 0:
            current_sums[ap_containing[i]] += signs[i]
    
    rng = random.Random(42)
    
    # Run targeted local search
    result_signs, best_max = _targeted_local_search(
        signs, ap_indices, ap_containing, current_sums,
        start_time, deadline, rng
    )
    
    return result_signs

def construct():
    """
    Construct 256 signs (-1 or +1) minimizing the maximum absolute signed sum
    over any arithmetic progression in [1, 256].
    """
    signs = _construct_signs_targeted()
    return signs

# EVOLVE-BLOCK-END
