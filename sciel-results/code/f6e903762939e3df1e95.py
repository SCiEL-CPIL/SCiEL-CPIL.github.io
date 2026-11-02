import math
import time
import random
from itertools import combinations

def score(U):
    """Compute the score for a finite set U containing 0."""
    if not U or 0 not in U:
        return 0
    U = sorted(set(U))
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    
    n_sums = len(sums)
    n_diffs = len(diffs)
    
    if n_sums == 0:
        return 0
        
    ratio = n_diffs / n_sums
    if ratio <= 1:
        return 1.0
        
    val = 1 + math.log(ratio) / math.log(2 * max_u + 1)
    return val

def _fast_score(U):
    """Optimized score calculation using sets."""
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    sumset = set()
    diffset = set()
    for a in U_set:
        for b in U_set:
            sumset.add(a + b)
            diffset.add(a - b)
            
    if not sumset:
        return 0.0
        
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    if ratio == 1.0:
        return 1.0
        
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
        
    return 1.0 + log_ratio / log_denom

def _search_small_sets(max_n=5, max_val=50, deadline=None):
    """Exhaustive search for small sets U."""
    best_score = 0.0
    best_U = [0, 1, 3]
    
    start_time = time.time()
    
    for n in range(3, max_n + 1):
        for combo in combinations(range(1, max_val + 1), n - 1):
            if deadline and time.time() > deadline:
                return best_U, best_score
                
            U = [0] + list(combo)
            U.sort()
            
            s = _fast_score(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
                
    return best_U, best_score

def _construct_heuristic_sets(deadline=None):
    """Construct sets based on known good structures."""
    candidates = []
    
    # Structure 1: Powers of 2 minus 1 (2^k - 1)
    # These have good difference/sum properties
    for k in range(2, 12):
        U = [2**i - 1 for i in range(k + 1)]
        if 0 in U:
            candidates.append(U)
    
    # Structure 2: Almost AP
    for k in range(3, 10):
        for M in range(k, 50):
            if deadline and time.time() > deadline:
                break
            U = list(range(k - 1)) + [M]
            if len(set(U)) == len(U):
                candidates.append(U)
                
    # Structure 3: Golomb rulers
    known_gr = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 22],
        [0, 1, 4, 10, 12, 17, 22, 29],
    ]
    candidates.extend(known_gr)
    
    # Structure 4: Randomized perturbations
    random.seed(42)
    for _ in range(50):
        if deadline and time.time() > deadline:
            break
        n = random.randint(3, 8)
        base_ap = list(range(n))
        i = random.randint(1, n - 1)
        new_val = random.randint(1, 60)
        U = base_ap[:]
        U[i] = new_val
        if len(set(U)) == len(U) and 0 in U:
            candidates.append(sorted(set(U)))
            
    return candidates

def _local_search(U, max_iters=200, deadline=None):
    """Improve set U via local search."""
    U = sorted(set(U))
    best_score = _fast_score(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 20
    
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        # Try moving one element
        for i in range(len(U)):
            old_val = U[i]
            if old_val == 0:
                continue
            for new_val in range(1, upper_bound + 1):
                if new_val == old_val:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                if len(set(new_U)) != len(new_U):
                    continue
                s = _fast_score(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
            
    return sorted(set(best_U)), best_score

def _hill_climbing(U, max_iters=100, deadline=None):
    """Hill climbing with add/remove/replace operations."""
    current_set = sorted(set(U))
    current_score = _fast_score(current_set)
    
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        improved = False
        max_u = max(current_set) if current_set else 0
        upper_bound = max_u * 2 + 10
        
        # Try adding a random element
        x = random.randint(1, upper_bound)
        if x not in current_set:
            new_set = current_set + [x]
            s = _fast_score(new_set)
            if s > current_score + 1e-9:
                current_set = new_set
                current_score = s
                improved = True
        
        # Try removing an element (if size > 1)
        if not improved and len(current_set) > 1:
            idx = random.randint(1, len(current_set) - 1)
            new_set = current_set[:]
            new_set.pop(idx)
            s = _fast_score(new_set)
            if s > current_score + 1e-9:
                current_set = new_set
                current_score = s
                improved = True
        
        # Try replacing an element
        if not improved and len(current_set) > 1:
            idx = random.randint(1, len(current_set) - 1)
            new_val = random.randint(1, upper_bound)
            if new_val not in current_set:
                new_set = current_set[:]
                new_set[idx] = new_val
                s = _fast_score(new_set)
                if s > current_score + 1e-9:
                    current_set = new_set
                    current_score = s
                    improved = True
        
        if not improved:
            # Try a few more random moves
            for _ in range(5):
                if deadline and time.time() > deadline:
                    break
                x = random.randint(1, upper_bound)
                if x not in current_set:
                    new_set = current_set + [x]
                    s = _fast_score(new_set)
                    if s > current_score + 1e-9:
                        current_set = new_set
                        current_score = s
                        improved = True
                        break
        
        if not improved:
            break
            
    return sorted(set(current_set)), current_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    best_score = 0
    best_set = [0, 1, 3]
    
    # 1. Exhaustive search for small sets
    try:
        U_sol, s_sol = _search_small_sets(max_n=5, max_val=50, deadline=deadline)
        if s_sol > best_score:
            best_score = s_sol
            best_set = U_sol
    except Exception:
        pass
        
    # 2. Heuristic candidates
    candidates = _construct_heuristic_sets(deadline=deadline)
    for U in candidates:
        if deadline and time.time() > deadline:
            break
        s = _fast_score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
            
    # 3. Local search on best found
    if time.time() < deadline:
        best_set, best_score = _local_search(best_set, max_iters=100, deadline=deadline)
        
    # 4. Hill climbing for further improvement
    if time.time() < deadline:
        best_set, best_score = _hill_climbing(best_set, max_iters=200, deadline=deadline)
        
    return sorted(set(best_set))
