"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def _score(A):
    """Compute max(|A+A|, |A*A|) for set A."""
    A = list(A)
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return max(len(sums), len(prods))

def _is_valid(A, n=20, maximum=10000):
    """Check if A has n distinct positive integers <= maximum."""
    if len(A) != n:
        return False
    if len(set(A)) != n:
        return False
    if any(x <= 0 for x in A):
        return False
    if any(x > maximum for x in A):
        return False
    return True

def _local_search(A, time_limit, rng):
    """Improve A via local search."""
    A = list(A)
    current_score = _score(A)
    best_A = list(A)
    best_score = current_score
    deadline = time.time() + time_limit
    
    while time.time() < deadline:
        # Randomly pick an element to perturb
        idx = rng.randint(0, len(A) - 1)
        old_val = A[idx]
        
        # Try several candidate replacements
        for _ in range(50):
            if A[idx] > 10:
                new_val = max(1, A[idx] + rng.randint(-20, 20))
            else:
                new_val = rng.randint(1, 100)
            
            if new_val == old_val:
                continue
            if new_val > 10000:
                continue
            if new_val in A and new_val != old_val:
                continue
            
            A[idx] = new_val
            s = _score(A)
            if s < best_score:
                best_score = s
                best_A = list(A)
            
            # Accept with some probability if not worse (simulated annealing style)
            if s > current_score:
                # Revert
                A[idx] = old_val
            else:
                current_score = s
        
        A[idx] = old_val
    
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin before 90s kill
    
    rng = random.Random(42)
    
    # Start with a geometric-like sequence
    best_A = None
    best_score = float('inf')
    
    # Try several initial candidates
    initial_candidates = []
    
    # Powers of 2
    p2 = [2**i for i in range(20)]
    if max(p2) <= 10000:
        initial_candidates.append(p2)
    
    # Mix of small numbers and geometric
    for base in [2, 3, 4, 5]:
        cand = []
        val = 1
        for i in range(20):
            cand.append(val)
            val *= base
            if val > 10000:
                val = min(10000, val + rng.randint(1, 100))
        if len(set(cand)) == 20 and all(1 <= x <= 10000 for x in cand):
            initial_candidates.append(cand)
    
    # Random geometric with varying ratios
    for _ in range(10):
        ratio = rng.uniform(1.2, 2.5)
        start = rng.randint(1, 10)
        cand = []
        val = float(start)
        for i in range(20):
            iv = int(round(val))
            if iv < 1:
                iv = 1
            if iv > 10000:
                iv = 10000
            if iv not in cand:
                cand.append(iv)
            else:
                # Find next unique
                for k in range(iv+1, min(iv+100, 10001)):
                    if k not in cand:
                        cand.append(k)
                        break
                else:
                    break
            val *= ratio
        if len(cand) == 20 and all(1 <= x <= 10000 for x in cand):
            initial_candidates.append(cand)
    
    # Random sets
    for _ in range(5):
        cand = list(rng.sample(range(1, 10001), 20))
        initial_candidates.append(cand)
    
    # Evaluate and improve each candidate
    for cand in initial_candidates:
        if time.time() > deadline:
            break
        
        if not _is_valid(cand, 20, 10000):
            continue
        
        s = _score(cand)
        if s < best_score:
            best_score = s
            best_A = list(cand)
        
        # Local search
        if time.time() < deadline - 5:
            improved_A, improved_score = _local_search(cand, min(5, deadline - time.time() - 2), rng)
            if improved_score < best_score:
                best_score = improved_score
                best_A = improved_A
    
    if best_A is None:
        return list(range(1, 21))
    
    return best_A
# EVOLVE-BLOCK-END
