import time
import math
import random
import numpy as np
from itertools import combinations, product

# Deadline guard: evaluator kills after 150s, we target 120s hard limit with margin
_START_TIME = time.time()
TIME_LIMIT = 115  # seconds, leave 5s margin for overhead

def _elapsed():
    return time.time() - _START_TIME

def _time_remaining():
    return TIME_LIMIT - _elapsed()

def _score_set(U):
    """
    Compute the lower bound: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)
    U must be a finite set of nonnegative integers containing 0.
    """
    if not U:
        return 0.0
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    # Compute U+U and U-U
    U_arr = np.array(U, dtype=np.int64)
    # Sumset: all a+b for a,b in U
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    # Difference set: all a-b for a,b in U
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    
    sum_size = len(sumset)
    diff_size = len(diffset)
    
    if sum_size == 0:
        return 0.0
    
    log_ratio = math.log(diff_size / sum_size)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom <= 0:
        return 0.0
    
    return 1.0 + log_ratio / log_denom

def _score_set_fast(U):
    """
    Faster scoring using numpy for larger sets.
    """
    if not U:
        return 0.0
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    U_arr = np.array(U, dtype=np.int64)
    n = len(U_arr)
    
    # Sumset via outer sum
    sums = U_arr[:, None] + U_arr[None, :]
    sumset_size = len(np.unique(sums))
    
    # Difference set via outer diff
    diffs = U_arr[:, None] - U_arr[None, :]
    diffset_size = len(np.unique(diffs))
    
    if sumset_size == 0:
        return 0.0
    
    log_ratio = math.log(diffset_size / sumset_size)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom <= 0:
        return 0.0
    
    return 1.0 + log_ratio / log_denom

def _build_geometric(ratio, n_terms):
    """Build a geometric progression set."""
    if ratio <= 1:
        return [0]
    U = [0]
    current = 1
    for i in range(1, n_terms):
        U.append(current)
        current = int(current * ratio)
    return U

def _build_binary(n_bits):
    """Build a set based on binary representation (powers of 2)."""
    return [0] + [1 << i for i in range(n_bits)]

def _build_mixed(base, step, n_terms):
    """Build a mixed arithmetic-geometric set."""
    U = [0]
    current = base
    for i in range(1, n_terms):
        U.append(current)
        current = current + step
    return U

def _local_search(U, max_iterations=200):
    """
    Optimized local search: 
    1. Uses heuristic candidate generation instead of linear scan.
    2. Focuses on adding elements that likely increase the difference set size.
    """
    U = sorted(set(U))
    if 0 not in U:
        U = list(set(U) | {0})
        U.sort()
    
    best_score = _score_set_fast(U)
    best_U = list(U)
    
    for it in range(max_iterations):
        if _time_remaining() < 5:
            break
            
        current_U = list(best_U)
        max_u = max(current_U)
        
        # Generate smart candidates
        candidates = set()
        # Sums of pairs (often create new structure)
        for a in current_U:
            for b in current_U:
                s = a + b
                if s > max_u:
                    candidates.add(s)
        # Multiples
        for a in current_U:
            for k in [2, 3, 5]:
                candidates.add(a * k)
        # Near max
        for d in range(1, 10):
            candidates.add(max_u + d)
        # Random far jumps
        for _ in range(10):
            candidates.add(max_u + random.randint(1, max(10, max_u // 2)))
            
        # Filter candidates already in U and > 0
        candidates = [c for c in candidates if c not in current_U and c > 0]
        
        if not candidates:
            break
            
        best_candidate = None
        best_candidate_score = best_score
        
        # Evaluate candidates
        for c in candidates:
            new_U = current_U + [c]
            s = _score_set_fast(new_U)
            if s > best_candidate_score:
                best_candidate_score = s
                best_candidate = c
                
        if best_candidate is not None:
            best_U.append(best_candidate)
            best_U.sort()
            best_score = best_candidate_score
        else:
            # No improvement by adding. 
            # Optional: try removing an element if we are stuck, but it's risky/slow.
            # We'll just break to save time for other seeds.
            break
            
    return best_U, best_score

def _multi_phase_search():
    """
    Multi-phase local search with different initializations.
    """
    best_U = [0]
    best_score = _score_set_fast([0])
    
    initializations = []
    
    # 1. Geometric (Powers) - Sparse sets
    for base in [2, 3, 4, 5, 10]:
        for n_terms in range(3, 15):
            U = [0]
            current = 1
            for i in range(1, n_terms):
                U.append(current)
                current *= base
            initializations.append(U)
            
    # 2. Quadratic growth (Golomb-like)
    for n_terms in range(3, 15):
        U = [0]
        for i in range(1, n_terms):
            U.append(i*i)
        initializations.append(U)
        
    # 3. Random sparse
    random.seed(123)
    for _ in range(30):
        n = random.randint(4, 12)
        max_val = random.randint(50, 200)
        U = [0] + random.sample(range(1, max_val), n-1)
        initializations.append(sorted(U))
        
    # 4. Mix of small and large
    for _ in range(20):
        U = [0, 1, 2]
        max_val = random.randint(20, 100)
        for _ in range(random.randint(3, 8)):
            U.append(random.randint(10, max_val))
        initializations.append(sorted(set(U)))

    # 5. Original structured ones (keep some for diversity)
    for ratio in [2, 3]:
        for n_terms in range(3, 12):
            initializations.append(_build_geometric(ratio, n_terms))

    # Evaluate and locally optimize each
    for init_U in initializations:
        if _time_remaining() < 10:
            break
        
        # Quick score to filter
        quick_score = _score_set_fast(init_U)
        # Be less aggressive with filtering to allow local search to fix bad seeds
        if quick_score < best_score - 0.2:
            continue
        
        optimized_U, optimized_score = _local_search(init_U, max_iterations=100)
        if optimized_score > best_score:
            best_score = optimized_score
            best_U = optimized_U
    
    return best_U, best_score

def _exhaustive_small(max_u, min_size=3, max_size=8):
    """
    Exhaustive search for small sets with max element <= max_u.
    """
    best_U = [0]
    best_score = _score_set_fast([0])
    
    candidates = list(range(0, max_u + 1))
    
    for size in range(min_size, max_size + 1):
        if _time_remaining() < 10:
            break
        
        for combo in combinations(candidates, size):
            if 0 not in combo:
                continue
            U = list(combo)
            score = _score_set_fast(U)
            if score > best_score:
                best_score = score
                best_U = U
    
    return best_U, best_score

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_U = [0, 1, 3]
    best_score = _score_set_fast(best_U)
    
    # Phase 1: Exhaustive search for small configurations
    try:
        small_U, small_score = _exhaustive_small(max_u=15, min_size=3, max_size=8)
        if small_score > best_score:
            best_score = small_score
            best_U = small_U
    except Exception:
        pass
    
    # Phase 2: Multi-phase local search with structured candidates
    try:
        multi_U, multi_score = _multi_phase_search()
        if multi_score > best_score:
            best_score = multi_score
            best_U = multi_U
    except Exception:
        pass
    
    # Phase 3: Further local search from the best found so far
    try:
        if _time_remaining() > 10:
            final_U, final_score = _local_search(best_U, max_iterations=1000)
            if final_score > best_score:
                best_score = final_score
                best_U = final_U
    except Exception:
        pass
    
    # Ensure 0 is in the set and return sorted list
    best_U = sorted(set(best_U) | {0})
    
    return best_U
