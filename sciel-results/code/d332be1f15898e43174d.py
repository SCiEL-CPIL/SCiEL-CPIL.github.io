import time
import random
import math
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 20
    max_val = 10000
    
    # Precompute all pairs (i, j) with i <= j for n=20
    pairs_i = np.array([i for i in range(n) for j in range(i, n)], dtype=np.int32)
    pairs_j = np.array([j for i in range(n) for j in range(i, n)], dtype=np.int32)
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A using NumPy vectorization."""
        A_arr = np.array(A, dtype=np.int64)
        sums = A_arr[pairs_i] + A_arr[pairs_j]
        prods = A_arr[pairs_i] * A_arr[pairs_j]
        sumset_size = len(np.unique(sums))
        prodset_size = len(np.unique(prods))
        return max(sumset_size, prodset_size)
    
    # Generate all 7-smooth numbers up to 10000
    smooths = set()
    for a in range(14):  # 2^13 > 8000
        p2 = 2 ** a
        if p2 > max_val: break
        for b in range(9): # 3^8 > 6000
            p3 = 3 ** b
            if p2 * p3 > max_val: break
            for c in range(6): # 5^5 > 3000
                p5 = 5 ** c
                if p2 * p3 * p5 > max_val: break
                for d in range(5): # 7^4 > 2400
                    p7 = 7 ** d
                    val = p2 * p3 * p5 * p7
                    if val > max_val: break
                    smooths.add(val)
    
    smooth_list = sorted(list(smooths))
    print(f"Found {len(smooth_list)} smooth numbers")
    
    # Map smooth numbers to indices for fast lookup
    smooth_index = {v: i for i, v in enumerate(smooth_list)}
    
    # Initial set: select 20 smooth numbers that are well-distributed
    # Strategy: pick powers of 2 and 3, then add 5s and 7s to fill gaps
    initial_candidates = []
    # Powers of 2
    for a in range(14):
        v = 2 ** a
        if v <= max_val:
            initial_candidates.append(v)
    # Powers of 3
    for b in range(9):
        v = 3 ** b
        if v <= max_val:
            initial_candidates.append(v)
    # Powers of 5
    for c in range(6):
        v = 5 ** c
        if v <= max_val:
            initial_candidates.append(v)
    # Powers of 7
    for d in range(5):
        v = 7 ** d
        if v <= max_val:
            initial_candidates.append(v)
    
    # Remove duplicates and sort
    initial_candidates = sorted(set(initial_candidates))
    
    # Select 20 from initial candidates, biased toward smaller numbers
    if len(initial_candidates) >= n:
        # Take first 15 small ones, then sample from the rest
        selected = initial_candidates[:15]
        remaining = initial_candidates[15:]
        random.shuffle(remaining)
        selected.extend(remaining[:5])
        current_set = sorted(selected)
    else:
        # If not enough, fill with random smooth numbers
        current_set = initial_candidates[:]
        while len(current_set) < n:
            v = random.choice(smooth_list)
            if v not in current_set:
                current_set.append(v)
        current_set = sorted(current_set)
    
    # Ensure exactly n distinct positive integers <= max_val
    current_set = sorted(list(set(current_set))[:n])
    while len(current_set) < n:
        v = random.choice(smooth_list)
        if v not in current_set:
            current_set.append(v)
        current_set = sorted(current_set[:n])
    
    current_score = evaluate(current_set)
    best_set = current_set[:]
    best_score = current_score
    
    # Local search on the smooth number lattice
    # Represent each element by its index in smooth_list
    current_indices = [smooth_index[v] for v in current_set]
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for idx in range(n):
            old_idx = current_indices[idx]
            old_val = smooth_list[old_idx]
            
            # Try neighboring smooth numbers (in terms of value, not index)
            # Look at a window around the current value
            val_min = max(1, old_val - 50)
            val_max = min(max_val, old_val + 50)
            
            # Find smooth numbers in this range
            candidates = [v for v in smooth_list if val_min <= v <= val_max]
            
            for new_val in candidates:
                if new_val == old_val:
                    continue
                if new_val in current_set:
                    continue
                
                new_idx = smooth_index[new_val]
                current_indices[idx] = new_idx
                current_set[idx] = new_val
                new_score = evaluate(current_set)
                
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_set = current_set[:]
                    improved = True
                else:
                    # Revert
                    current_indices[idx] = old_idx
                    current_set[idx] = old_val
    
    # Second phase: try larger jumps in the smooth lattice
    if time.time() < deadline - 10:
        for _ in range(50):  # 50 random restarts of local search
            if time.time() > deadline - 5:
                break
            
            # Perturb the current best set by replacing 1-3 elements with random smooth numbers
            perturbed_set = best_set[:]
            num_changes = random.randint(1, 3)
            for _ in range(num_changes):
                idx = random.randint(0, n - 1)
                old_val = perturbed_set[idx]
                # Pick a random smooth number far from old_val
                for _ in range(10):
                    new_val = random.choice(smooth_list)
                    if new_val != old_val and new_val not in perturbed_set:
                        perturbed_set[idx] = new_val
                        break
                else:
                    continue
            
            perturbed_set = sorted(perturbed_set)
            perturbed_score = evaluate(perturbed_set)
            
            if perturbed_score < best_score:
                best_score = perturbed_score
                best_set = perturbed_set[:]
            
            # Local search from the perturbed set
            current_set = perturbed_set[:]
            current_score = perturbed_score
            current_indices = [smooth_index[v] for v in current_set]
            
            improved_local = True
            while improved_local and time.time() < deadline - 5:
                improved_local = False
                for idx in range(n):
                    old_idx = current_indices[idx]
                    old_val = smooth_list[old_idx]
                    
                    val_min = max(1, old_val - 30)
                    val_max = min(max_val, old_val + 30)
                    candidates = [v for v in smooth_list if val_min <= v <= val_max]
                    
                    for new_val in candidates:
                        if new_val == old_val:
                            continue
                        if new_val in current_set:
                            continue
                        
                        new_idx = smooth_index[new_val]
                        current_indices[idx] = new_idx
                        current_set[idx] = new_val
                        new_score = evaluate(current_set)
                        
                        if new_score < current_score:
                            current_score = new_score
                            if current_score < best_score:
                                best_score = current_score
                                best_set = current_set[:]
                            improved_local = True
                        else:
                            current_indices[idx] = old_idx
                            current_set[idx] = old_val
    
    return sorted(best_set)
