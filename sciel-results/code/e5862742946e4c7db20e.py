import time
import random
import math

def _compute_score(A):
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return max(len(sums), len(prods))

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Good initial set: mix of powers of 2 and 3*2^i
    # 2^0..2^13 (14 elements) + 3*2^0..3*2^5 (6 elements) = 20
    A = [2**i for i in range(14)] + [3 * 2**i for i in range(6)]
    A = sorted(set(A))
    assert len(A) == 20, f"Expected 20, got {len(A)}"
    
    best_A = A[:]
    best_score = _compute_score(best_A)
    
    random.seed(123)
    temp = 50.0
    cooling = 0.9995
    
    # Precompute for speed
    A_set = set(best_A)
    
    while time.time() < deadline:
        # Pick an element to replace
        idx = random.randint(0, 19)
        old_val = best_A[idx]
        
        # Generate candidate new values
        # Try structured values: powers of 2, 3*2^i, 5*2^i, etc.
        candidates = set()
        for p in [2, 3, 5, 7]:
            for e in range(0, 15):
                v = p ** e
                if 1 <= v <= 10000:
                    candidates.add(v)
        # Also try near existing values
        for v in best_A:
            for delta in [-3, -2, -1, 1, 2, 3, 5, 10]:
                nv = v + delta
                if 1 <= nv <= 10000:
                    candidates.add(nv)
        # Remove current elements
        candidates -= set(best_A)
        candidates.discard(old_val)
        
        if not candidates:
            continue
        
        new_val = random.choice(list(candidates))
        
        # Create new set
        new_A = best_A[:]
        new_A[idx] = new_val
        new_A = sorted(set(new_A))
        
        if len(new_A) < 20:
            # Need to add back an element
            for v in range(1, 10001):
                if v not in new_A:
                    new_A.append(v)
                    break
            new_A = sorted(new_A[:20])
        
        if len(new_A) != 20 or len(set(new_A)) != 20:
            continue
        
        new_score = _compute_score(new_A)
        
        accept = False
        if new_score < best_score:
            accept = True
        elif new_score == best_score:
            accept = True
        else:
            delta = new_score - best_score
            if temp > 1e-10 and random.random() < math.exp(-delta / temp):
                accept = True
        
        if accept:
            best_A = new_A
            best_score = new_score
        
        temp *= cooling
    
    return best_A
