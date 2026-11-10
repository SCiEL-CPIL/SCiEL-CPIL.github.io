"""
Rewritten program: Local search with fast AP sum updates for minimizing max |sum| over APs.
Uses quadratic character initialization and efficient local search.
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
    Returns:
        ap_containing: List of lists, where ap_containing[k] is a list of AP indices j 
                       such that k is in ap_indices[j].
    """
    ap_containing = [[] for _ in range(n)]
    for j, indices in enumerate(ap_indices):
        for k in indices:
            ap_containing[k].append(j)
    return ap_containing

def _quadratic_character_init(n, p=257):
    """
    Initialize signs using quadratic character modulo prime p.
    For prime p, the Legendre symbol (a/p) is 1 if a is a quadratic residue mod p, -1 otherwise.
    """
    signs = np.zeros(n, dtype=np.int8)
    for i in range(n):
        # Use i+1 mod p for the quadratic character
        a = (i + 1) % p
        # Compute Legendre symbol using Euler's criterion
        if a == 0:
            signs[i] = 1
        else:
            # (a/p) = a^((p-1)/2) mod p
            if pow(a, (p - 1) // 2, p) == 1:
                signs[i] = 1
            else:
                signs[i] = -1
    return signs

def _construct_signs_fast():
    """
    Construct signs using quadratic character init and fast local search.
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = _N
    ap_indices = _generate_aps(n)
    num_aps = len(ap_indices)
    
    if num_aps == 0:
        return [1] * n

    ap_containing = _precompute_ap_containing(ap_indices, n)
    
    # Initialize using quadratic character
    signs = _quadratic_character_init(n)
    
    # Compute initial sums
    current_sums = np.zeros(num_aps, dtype=np.int32)
    for j, indices in enumerate(ap_indices):
        current_sums[j] = np.sum(signs[indices])
    
    current_max = int(np.max(np.abs(current_sums)))
    
    # Local Search with simulated annealing-like acceptance
    best_signs = signs.copy()
    best_max = current_max
    rng = random.Random(42)
    
    # Track which APs have the maximum for faster evaluation
    max_aps = np.where(np.abs(current_sums) == current_max)[0]
    
    no_improve_count = 0
    max_no_improve = 50
    
    # Temperature for simulated annealing
    temperature = 1.0
    cooling_rate = 0.995
    
    while time.time() < deadline:
        pos = rng.randint(0, n - 1)
        
        old_sign = signs[pos]
        new_sign = -old_sign
        
        affected_aps = ap_containing[pos]
        if len(affected_aps) == 0:
            continue
            
        # Save old sums for affected APs
        old_sums = current_sums[affected_aps].copy()
        
        # Apply flip: change is -2 * old_sign
        delta = -2 * old_sign
        current_sums[affected_aps] += delta
        signs[pos] = new_sign
        
        # Check new max efficiently
        # Only affected APs changed, so new max is max(max of unchanged, max of affected)
        # We can compute max of affected and compare with best_max
        new_affected_max = int(np.max(np.abs(current_sums[affected_aps])))
        
        # If no affected AP has |sum| > best_max, then new_max <= best_max
        # Otherwise, new_max = new_affected_max (since unchanged APs have |sum| <= best_max)
        if new_affected_max <= best_max:
            new_max = best_max
        else:
            new_max = new_affected_max
        
        # Accept if improvement, or with probability exp(-(new_max - best_max) / temperature) if worse
        if new_max < best_max:
            # Improvement
            best_max = new_max
            best_signs = signs.copy()
            no_improve_count = 0
        elif new_max == best_max:
            # Same max, accept with some probability to explore
            if rng.random() < 0.5:
                no_improve_count += 1
            else:
                # Revert
                current_sums[affected_aps] = old_sums
                signs[pos] = old_sign
                no_improve_count += 1
        else:
            # Worse, accept with probability exp(-(new_max - best_max) / temperature)
            acceptance_prob = np.exp(-(new_max - best_max) / max(temperature, 0.001))
            if rng.random() < acceptance_prob:
                # Accept worse solution to escape local minima
                no_improve_count += 1
            else:
                # Revert
                current_sums[affected_aps] = old_sums
                signs[pos] = old_sign
                no_improve_count += 1
        
        # Cool down
        temperature *= cooling_rate
        
        # If stuck, do a perturbation
        if no_improve_count >= max_no_improve:
            # Perturb a few random positions
            num_flips = rng.randint(3, 7)
            for _ in range(num_flips):
                p = rng.randint(0, n - 1)
                affected = ap_containing[p]
                if len(affected) == 0:
                    continue
                old_s = signs[p]
                signs[p] = -old_s
                current_sums[affected] += -2 * old_s
            
            # Recompute max after perturbation
            current_max = int(np.max(np.abs(current_sums)))
            no_improve_count = 0
            temperature = max(temperature * 2, 0.5)  # Reheat
    
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
