import math
import time
import random

def _score(U):
    if not U or 0 not in U:
        return 0
    U = sorted(set(U))
    if not U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0 # log(1/1)/log(1) is undefined, but limit is 1? 
        # Actually if U={0}, U+U={0}, U-U={0}. Ratio 1. Log 1 = 0. Score 1.
    
    sumset = set()
    diffset = set()
    for u in U:
        for v in U:
            sumset.add(u + v)
            diffset.add(u - v)
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
    return 1 + math.log(ratio) / denom

def _geometric_set(base, n):
    U = [0]
    val = 1
    for i in range(1, n):
        U.append(val)
        val *= base
    return U

def _local_search(U, max_iterations=1000, deadline=None):
    U = sorted(set(U))
    best_score = _score(U)
    best_U = U
    current_U = U
    current_score = best_score
    
    for i in range(max_iterations):
        if deadline and time.time() > deadline:
            break
        
        # Try adding a random element
        max_u = max(current_U)
        # Range for new element
        # Sometimes adding a small element helps, sometimes a large one
        if random.random() < 0.5:
            new_val = random.randint(1, max_u * 2)
        else:
            new_val = random.randint(1, max_u + 10)
        
        candidate = sorted(set(current_U + [new_val]))
        s = _score(candidate)
        if s > current_score:
            current_U = candidate
            current_score = s
            if s > best_score:
                best_score = s
                best_U = candidate
        
        # Try removing an element (not 0)
        if len(current_U) > 1:
            idx = random.randint(1, len(current_U) - 1)
            candidate = sorted(set(current_U[:idx] + current_U[idx+1:]))
            s = _score(candidate)
            if s > current_score:
                current_U = candidate
                current_score = s
                if s > best_score:
                    best_score = s
                    best_U = candidate
        else:
            # If only 0, add 1
            candidate = [0, 1]
            s = _score(candidate)
            if s > current_score:
                current_U = candidate
                current_score = s
                if s > best_score:
                    best_score = s
                    best_U = candidate

        # Try shifting an element
        if len(current_U) > 1:
            idx = random.randint(1, len(current_U) - 1)
            shift = random.randint(-5, 5)
            if shift == 0: continue
            new_val = current_U[idx] + shift
            if new_val >= 0:
                candidate = sorted(set([current_U[j] if j != idx else new_val for j in range(len(current_U))]))
                if len(candidate) < len(current_U):
                    # Collision, skip or handle
                    pass
                s = _score(candidate)
                if s > current_score:
                    current_U = candidate
                    current_score = s
                    if s > best_score:
                        best_score = s
                        best_U = candidate
                        
    return best_U

def construct():
    start_time = time.time()
    deadline = start_time + 100 # 20s safety margin
    
    best_U = [0, 1, 3]
    best_score = _score(best_U)
    
    # Try geometric sets
    for base in [2, 3, 4, 5, 6, 10]:
        for n in range(3, 8):
            if time.time() > deadline:
                break
            U = _geometric_set(base, n)
            s = _score(U)
            if s > best_score:
                best_score = s
                best_U = U
            
            # Refine with local search
            if s > best_score * 0.9: # Only refine promising ones
                refined = _local_search(U, max_iterations=500, deadline=deadline)
                s_r = _score(refined)
                if s_r > best_score:
                    best_score = s_r
                    best_U = refined
                    
    # Try some random sets
    for _ in range(10):
        if time.time() > deadline:
            break
        n = random.randint(3, 8)
        U = [0]
        max_val = random.randint(10, 100)
        for i in range(n-1):
            U.append(random.randint(1, max_val))
        U = sorted(set(U))
        s = _score(U)
        if s > best_score:
            best_score = s
            best_U = U
        
        refined = _local_search(U, max_iterations=500, deadline=deadline)
        s_r = _score(refined)
        if s_r > best_score:
            best_score = s_r
            best_U = refined

    return best_U
