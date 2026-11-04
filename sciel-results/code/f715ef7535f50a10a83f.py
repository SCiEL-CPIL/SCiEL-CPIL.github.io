import time
import math
import random
import itertools
from collections import defaultdict

def _compute_score(U):
    """Compute the score for a given set U."""
    U = sorted(set(U))
    if not U or U[0] != 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    # Compute U+U and U-U
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    n_sum = len(sumset)
    n_diff = len(diffset)
    if n_sum == 0:
        return 0.0
    log_ratio = math.log(n_diff / n_sum)
    log_denom = math.log(2 * max_u + 1)
    if log_denom <= 0:
        return 0.0
    return 1.0 + log_ratio / log_denom

def _try_set(U):
    """Quick score computation, returns 0 if invalid."""
    U = list(set(U))
    if 0 not in U or any(x < 0 for x in U):
        return 0.0
    return _compute_score(U)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Strategy 1: Try structured sets with geometric progressions and offsets
    # U = {a * r^i + b_i : i = 0..m}
    
    # Strategy 2: Try sets of form {i * M + f(i) : i = 0..k}
    
    # Strategy 3: Randomized local search
    
    # First, try some known good structures
    # Powers of 2 with perturbations
    for m in range(2, 12):
        if time.time() > deadline:
            break
        for r in [2, 3, 4, 5, 7, 10, 15, 20]:
            if time.time() > deadline:
                break
            # Pure geometric
            U = [r**i for i in range(m+1)]
            s = _try_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
            
            # Geometric with +1 at each step
            U = [r**i + (1 if i > 0 else 0) for i in range(m+1)]
            if 0 not in U:
                U[0] = 0
            s = _try_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    # Try two-scale sets: {0, 1, M, M+1, M^2, M^2+1, ...}
    for m in range(2, 8):
        if time.time() > deadline:
            break
        for M in [10, 20, 50, 100, 200, 500, 1000]:
            if time.time() > deadline:
                break
            U = set()
            U.add(0)
            U.add(1)
            power = M
            for i in range(m):
                U.add(power)
                U.add(power + 1)
                power *= M
            U = sorted(U)
            s = _try_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    # Try sets of form {i*M + j : i=0..k, j in S} where S is small
    for k in range(2, 15):
        if time.time() > deadline:
            break
        for M in [3, 5, 8, 13, 21, 34, 55]:
            if time.time() > deadline:
                break
            # S = {0, 1}
            U = set()
            for i in range(k+1):
                U.add(i * M)
                U.add(i * M + 1)
            U = sorted(U)
            s = _try_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
            
            # S = {0, 1, 2}
            U = set()
            for i in range(k+1):
                for j in range(3):
                    U.add(i * M + j)
            U = sorted(U)
            s = _try_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    # Randomized search: start with a base set and mutate
    random.seed(42)
    
    # Generate random candidate sets
    for trial in range(5000):
        if time.time() > deadline:
            break
        
        # Random size
        size = random.randint(3, 50)
        
        # Random growth pattern
        r = random.choice([1.2, 1.5, 2, 3, 5, 8])
        offset = random.choice([0, 1, 2])
        
        U = [0]
        current = 1
        for i in range(size - 1):
            U.append(current)
            current = int(current * r) + random.randint(0, offset)
        
        U = sorted(set(U))
        if 0 not in U:
            U.insert(0, 0)
        
        s = _try_set(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    # Local search: try adding/removing elements from best set
    if time.time() < deadline:
        current = best_set[:]
        current_score = best_score
        improved = True
        max_iter = 500
        iter_count = 0
        
        while improved and iter_count < max_iter and time.time() < deadline:
            improved = False
            iter_count += 1
            
            # Try adding a new element
            max_u = max(current) if current else 0
            candidates = []
            for d in range(1, min(100, max_u // 2 + 1)):
                candidates.append(max_u + d)
            # Also try small additions
            for d in range(1, 20):
                if d not in current:
                    candidates.append(d)
            # Try geometric extensions
            if max_u > 0:
                for mult in [2, 3, 5]:
                    candidates.append(max_u * mult)
                    candidates.append(max_u * mult + 1)
            
            random.shuffle(candidates)
            
            for cand in candidates[:20]:  # Limit per iteration
                if time.time() > deadline:
                    break
                new_set = current + [cand]
                new_set = sorted(set(new_set))
                s = _try_set(new_set)
                if s > current_score + 1e-10:
                    current = new_set
                    current_score = s
                    if s > best_score:
                        best_score = s
                        best_set = current[:]
                    improved = True
                    break
            
            # Try removing an element (not 0)
            if len(current) > 2:
                removable = [x for x in current if x != 0]
                random.shuffle(removable)
                for rem in removable[:10]:
                    if time.time() > deadline:
                        break
                    new_set = [x for x in current if x != rem]
                    s = _try_set(new_set)
                    if s > current_score + 1e-10:
                        current = new_set
                        current_score = s
                        if s > best_score:
                            best_score = s
                            best_set = current[:]
                        improved = True
                        break
    
    # Final: ensure 0 is in the set and return
    best_set = sorted(set(best_set))
    if 0 not in best_set:
        best_set.insert(0, 0)
    
    return best_set
