"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_DEADLINE = 90.0
_SAFETY_MARGIN = 5.0

def _collision_score(candidate_set):
    """Count the number of unordered pair-sum collisions (including diagonals)."""
    sums = {}
    for i in range(len(candidate_set)):
        for j in range(i, len(candidate_set)):
            s = candidate_set[i] + candidate_set[j]
            sums[s] = sums.get(s, 0) + 1
    collisions = sum(c - 1 for c in sums.values() if c > 1)
    return collisions

def _is_sidon_fast(subset):
    """Check if subset is Sidon with early termination."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_subset_size(ambient_set, max_size_to_check=6):
    """Returns the largest Sidon subset size found (up to max_size_to_check)."""
    n = len(ambient_set)
    for size in range(min(max_size_to_check, n), 2, -1):
        for combo in combinations(ambient_set, size):
            if _is_sidon_fast(combo):
                return size
    return 2

def _local_search_improve(ambient_set, time_limit):
    """Improve via single-element replacement local search."""
    best_set = sorted(ambient_set)
    best_ms = _max_sidon_subset_size(best_set, 5)
    best_cs = _collision_score(best_set)
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(len(best_set)):
            for new_val in range(129):
                if new_val in best_set and new_val != best_set[i]:
                    continue
                trial = list(best_set)
                trial[i] = new_val
                trial.sort()
                cs = _collision_score(trial)
                if cs > best_cs:
                    ms = _max_sidon_subset_size(trial, 5)
                    if ms < best_ms:
                        best_set = trial
                        best_ms = ms
                        best_cs = cs
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    
    return best_set, best_ms

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE - _SAFETY_MARGIN
    
    best_ambient = None
    best_max_sidon = 12
    
    random.seed(42)
    
    # Phase 1: Multiple greedy attempts with different strategies
    for seed_attempt in range(30):
        if time.time() > deadline:
            break
        
        ambient = []
        used = set()
        
        # Random initialization with 2 elements
        init_pool = list(range(129))
        random.shuffle(init_pool)
        for _ in range(2):
            e = init_pool.pop()
            ambient.append(e)
            used.add(e)
        
        # Greedily add remaining 10 elements maximizing collisions
        for _ in range(10):
            if time.time() > deadline:
                break
            
            candidates = [x for x in range(129) if x not in used]
            best_elem = None
            best_score = -1
            
            for cand in candidates:
                new_set = list(ambient) + [cand]
                score = _collision_score(new_set)
                if score > best_score:
                    best_score = score
                    best_elem = cand
            
            if best_elem is None:
                break
            
            ambient.append(best_elem)
            used.add(best_elem)
        
        if len(ambient) == 12:
            ambient.sort()
            ms = _max_sidon_subset_size(ambient, 5)
            if ms < best_max_sidon:
                best_max_sidon = ms
                best_ambient = list(ambient)
            
            if best_max_sidon <= 3:
                break
    
    # Phase 2: Local search improvement
    if best_ambient is not None and time.time() < deadline:
        remaining_time = deadline - time.time()
        if remaining_time > 2.0:
            improved_set, improved_ms = _local_search_improve(
                best_ambient, remaining_time - 1.0
            )
            if improved_ms < best_max_sidon:
                best_ambient = improved_set
                best_max_sidon = improved_ms
    
    # Fallback
    if best_ambient is None:
        best_ambient = list(range(12))
    
    return best_ambient
# EVOLVE-BLOCK-END
