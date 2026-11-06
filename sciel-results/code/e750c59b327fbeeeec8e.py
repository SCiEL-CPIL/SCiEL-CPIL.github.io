import math
import time
import numpy as np

def _score_set(A):
    """Compute max(|A+A|, |A*A|) for a set/list of distinct positive integers."""
    if len(set(A)) != len(A):
        return float('inf')
    if any(x < 1 for x in A):
        return float('inf')
    
    # Convert to numpy array for vectorized operations
    arr = np.array(A, dtype=np.int64)
    
    # Sumset
    sset = np.unique(arr[:, None] + arr[None, :])
    
    # Productset
    pset = np.unique(arr[:, None] * arr[None, :])
    
    return max(len(sset), len(pset))

def construct():
    n = 20
    max_val = 10000
    best_score = float('inf')
    best_set = None
    
    candidates = []
    
    # 1. Structured candidates
    
    # 1..20
    candidates.append(list(range(1, n + 1)))
    
    # Arithmetic Progressions
    # a + i*d <= 10000
    for d in range(1, 20):
        for a in range(1, 100, d):
            A = [a + i * d for i in range(n)]
            if A[-1] <= max_val:
                candidates.append(A)
            else:
                break
    
    # Divisor Sets
    # Divisors of highly composite numbers
    for M in [2520, 5040, 840, 1260, 1680, 2520, 3360, 4200, 5040, 6720, 8400, 10080]:
        divs = [d for d in range(1, int(math.isqrt(M)) + 1) if M % d == 0]
        divs += [M // d for d in divs if d * d != M]
        divs = sorted(set(divs))
        if len(divs) >= n:
            # Take first n, last n, or middle n?
            # Taking small ones keeps sum small?
            candidates.append(divs[:n])
            candidates.append(divs[-n:])
            # Mix
            candidates.append(divs[:n//2] + divs[len(divs)//2:len(divs)//2+n//2])

    # Powers of 2 (padded)
    p2 = [2**i for i in range(n) if 2**i <= max_val]
    if len(p2) < n:
        remaining = sorted(set(range(1, max_val + 1)) - set(p2))
        p2 += remaining[:n - len(p2)]
    candidates.append(p2[:n])

    # Evaluate all structured candidates
    for cand in candidates:
        if len(cand) != n:
            continue
        if len(set(cand)) != n:
            continue
        if any(x < 1 or x > max_val for x in cand):
            continue
        s = _score_set(cand)
        if s < best_score:
            best_score = s
            best_set = cand[:]
    
    # 2. Local Search
    if best_set is None:
        best_set = list(range(1, n + 1))
    
    best_set.sort()
    current_score = best_score
    
    start_time = time.time()
    time_limit = 55
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                break
            old_val = best_set[i]
            candidates_for_swap = set()
            
            # Neighbors
            for d in range(1, 50):
                candidates_for_swap.add(old_val - d)
                candidates_for_swap.add(old_val + d)
            
            # Multiples/Divisors of other elements
            for j in range(n):
                if j == i: continue
                other = best_set[j]
                for d in range(1, int(math.isqrt(other)) + 1):
                    if other % d == 0:
                        candidates_for_swap.add(d)
                        candidates_for_swap.add(other // d)
                for m in range(2, 8):
                    if other * m <= max_val:
                        candidates_for_swap.add(other * m)
            
            valid_candidates = [c for c in candidates_for_swap if 1 <= c <= max_val and c != old_val and c not in best_set]
            valid_candidates.sort()
            
            best_candidate = None
            best_candidate_score = current_score
            
            for new_val in valid_candidates:
                best_set[i] = new_val
                s = _score_set(best_set)
                if s < best_candidate_score:
                    best_candidate_score = s
                    best_candidate = new_val
            
            if best_candidate is not None:
                best_set[i] = best_candidate
                current_score = best_candidate_score
                improved = True
            else:
                best_set[i] = old_val
                
    return best_set
