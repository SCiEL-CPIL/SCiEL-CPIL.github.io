"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def _fast_is_sidon(lst):
    """Fast Sidon check using a boolean array for O(1) lookups."""
    n = len(lst)
    # Max possible sum is 2*128 = 256
    seen = [False] * 257
    for i in range(n):
        a = lst[i]
        for j in range(i, n):
            s = a + lst[j]
            if seen[s]:
                return False
            seen[s] = True
    return True

def _max_sidon_size(ambient):
    """Compute max Sidon subset size for sorted 12-element list."""
    n = len(ambient)
    # Try from largest k down to 2
    for k in range(min(n, 6), 1, -1):
        for combo in itertools.combinations(ambient, k):
            if _fast_is_sidon(list(combo)):
                return k
    return 2

def _random_ambient(rng):
    return sorted(rng.sample(range(129), 12))

def _local_search(ambient, rng, deadline, current_best_k, max_iters=300):
    """Local search with simulated annealing-style acceptance."""
    best = list(ambient)
    best_k = current_best_k
    no_improve = 0
    temperature = 0.5
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        if best_k <= 3:
            break
        
        idx = rng.randint(0, 11)
        new_val = rng.randint(0, 128)
        if new_val in best:
            no_improve += 1
            continue
        
        trial = best[:]
        trial[idx] = new_val
        trial.sort()
        k = _max_sidon_size(trial)
        
        if k < best_k:
            best = trial
            best_k = k
            no_improve = 0
            temperature = 0.5
        else:
            if k == best_k and rng.random() < 0.15:
                best = trial
                no_improve = 0
            else:
                no_improve += 1
                temperature *= 0.995
                if temperature < 0.01:
                    temperature = 0.01
        
        if no_improve >= 100:
            break
    
    return best, best_k

def _candidates():
    """Generate diverse initial candidates with high collision density."""
    cands = []
    # Arithmetic progressions (many pair sum collisions)
    for step in [2, 3, 4, 5, 6, 8, 10, 11, 12]:
        cand = list(range(0, min(129, step * 12), step))
        if len(cand) == 12 and cand[-1] <= 128:
            cands.append(cand)
    # Consecutive blocks
    cands.append(list(range(12)))
    cands.append(list(range(1, 13)))
    cands.append(list(range(2, 14)))
    cands.append(list(range(3, 15)))
    # Near-consecutive with one gap
    cands.append([0,1,2,3,4,5,6,7,8,9,10,12])
    cands.append([0,1,2,3,4,5,6,7,8,9,11,13])
    cands.append([0,1,2,3,4,5,6,7,8,9,10,13])
    cands.append([0,1,2,3,4,5,6,7,8,9,11,14])
    # Mixed patterns
    cands.append([0,2,4,6,8,10,12,14,16,18,20,22])
    cands.append([0,1,3,5,7,9,11,13,15,17,19,21])
    cands.append([0,1,2,4,6,8,10,12,14,16,18,20])
    cands.append([0,2,3,5,6,8,9,11,12,14,15,17])
    cands.append([0,1,3,4,6,7,9,10,12,13,15,16])
    cands.append([0,2,4,5,7,9,10,12,14,15,17,19])
    cands.append([0,1,4,5,8,9,12,13,16,17,20,21])
    # Quadratic-like (many sum collisions)
    cands.append([i*i for i in range(12)])
    cands.append([i*i + i for i in range(12)])
    cands.append([i*(i+1)//2 for i in range(12)])
    cands.append([i*(i+1)//2 + 1 for i in range(12)])
    # Fibonacci-like
    cands.append([0,1,2,3,5,8,13,21,34,55,89,128])
    cands.append([1,1,2,3,5,8,13,21,34,55,89,144][:12])
    # Exponential with clamping
    cands.append([min(2**i, 128) for i in range(12)])
    cands.append([min(2**i + i, 128) for i in range(12)])
    # Dense clusters
    cands.append([0,1,2,3,4,5,6,7,10,11,12,13])
    cands.append([0,1,2,3,4,5,8,9,10,11,12,13])
    cands.append([0,1,2,3,5,6,7,8,9,10,11,12])
    cands.append([0,1,2,4,5,6,7,8,9,10,11,12])
    # Symmetric patterns
    cands.append([0,1,2,3,4,5,6,5,4,3,2,1][:12])  # dedup
    cands.append([0,2,4,6,8,10,12,10,8,6,4,2][:12])  # dedup
    # More structured collision-prone sets
    cands.append([0,1,2,3,4,5,6,7,8,10,12,14])
    cands.append([0,1,2,3,4,5,6,7,9,11,13,15])
    cands.append([0,1,2,3,4,5,6,8,10,12,14,16])
    cands.append([0,1,2,3,4,5,7,9,11,13,15,17])
    cands.append([0,1,2,3,4,6,8,10,12,14,16,18])
    cands.append([0,1,2,3,5,7,9,11,13,15,17,19])
    cands.append([0,1,2,4,6,8,10,12,14,16,18,20])
    cands.append([0,1,3,5,7,9,11,13,15,17,19,21])
    cands.append([0,2,4,6,8,10,12,14,16,18,20,22])
    # Validate and deduplicate
    valid_cands = []
    seen = set()
    for c in cands:
        c = sorted(set(c))
        if len(c) == 12 and all(0 <= x <= 128 for x in c):
            t = tuple(c)
            if t not in seen:
                seen.add(t)
                valid_cands.append(c)
    return valid_cands

def _spectral_candidates():
    """Generate candidates using spectral graph clustering approach."""
    cands = []
    
    # Build a 129x129 matrix where entry (i,j) is 1 if i+j is a "frequent" sum
    # Sums in [0, 256]. The number of pairs (a,b) with a+b=s in [0,128] is:
    # For s in [0, 128]: s+1 pairs
    # For s in [129, 256]: 257-s pairs
    # We want to maximize collisions, so we want pairs that share frequent sums.
    
    # Count how many pairs produce each sum
    sum_counts = [0] * 257
    for s in range(257):
        if s <= 128:
            sum_counts[s] = s + 1
        else:
            sum_counts[s] = 257 - s
    
    # Build adjacency matrix: A[i,j] = 1 if i+j has high frequency
    # Use top 50% of sums as "frequent"
    sorted_sums = sorted(range(257), key=lambda x: sum_counts[x], reverse=True)
    frequent_sums = set(sorted_sums[:129])  # top half
    
    A = np.zeros((129, 129), dtype=float)
    for i in range(129):
        for j in range(129):
            if i + j in frequent_sums:
                A[i, j] = 1.0
    
    # Compute eigenvalues and eigenvectors
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(A)
    except Exception:
        return cands
    
    # The eigenvector corresponding to the smallest eigenvalue should identify
    # a cluster of nodes with high internal connectivity (many shared frequent sums)
    # We want the 12 integers corresponding to the most "central" part of the graph
    
    # Try different eigenvectors (smallest few eigenvalues)
    for eig_idx in range(min(5, len(eigenvalues))):
        vec = eigenvectors[:, eig_idx]
        # Take the top 12 by absolute value
        indices = np.argsort(np.abs(vec))[::-1][:12]
        cand = sorted([int(i) for i in indices])
        if len(cand) == 12 and all(0 <= x <= 128 for x in cand) and len(set(cand)) == 12:
            cands.append(cand)
    
    # Also try the largest eigenvalue's eigenvector (might identify a different structure)
    vec = eigenvectors[:, -1]
    indices = np.argsort(np.abs(vec))[::-1][:12]
    cand = sorted([int(i) for i in indices])
    if len(cand) == 12 and all(0 <= x <= 128 for x in cand) and len(set(cand)) == 12:
        cands.append(cand)
    
    # Validate and deduplicate
    valid_cands = []
    seen = set()
    for c in cands:
        c = sorted(set(c))
        if len(c) == 12 and all(0 <= x <= 128 for x in c):
            t = tuple(c)
            if t not in seen:
                seen.add(t)
                valid_cands.append(c)
    return valid_cands

def _select_best_ambient():
    start_time = time.time()
    deadline = start_time + 85
    
    best = None
    best_k = None
    
    # First try hand-crafted candidates
    for cand in _candidates():
        if time.time() > deadline - 2:
            break
        k = _max_sidon_size(cand)
        if best_k is None or k < best_k:
            best = cand
            best_k = k
            if best_k <= 3:
                break
    
    # Try spectral candidates
    if time.time() < deadline - 2:
        for cand in _spectral_candidates():
            if time.time() > deadline - 2:
                break
            k = _max_sidon_size(cand)
            if best_k is None or k < best_k:
                best = cand
                best_k = k
                if best_k <= 3:
                    break
    
    rng = random.Random(42)
    trials_done = 0
    max_trials = 1000
    
    while trials_done < max_trials:
        if time.time() > deadline - 3:
            break
        if best_k is not None and best_k <= 3:
            break
        
        cand = _random_ambient(rng)
        k = _max_sidon_size(cand)
        trials_done += 1
        
        if best_k is None or k < best_k:
            best = cand
            best_k = k
            if best_k <= 3:
                break
        
        if time.time() < deadline - 5:
            improved, ik = _local_search(cand, rng, deadline - 3, best_k if best_k is not None else 12)
            trials_done += 1
            if best_k is None or ik < best_k:
                best = improved
                best_k = ik
                if best_k <= 3:
                    break
    
    if best is not None and best_k > 3 and time.time() < deadline - 1:
        improved, ik = _local_search(best, rng, deadline - 0.5, best_k, max_iters=500)
        if ik < best_k:
            best = improved
            best_k = ik
    
    if best is None:
        return list(range(12))
    return best

def construct():
    return _select_best_ambient()
# EVOLVE-BLOCK-END
