import time
import random

def _score_set(A):
    """Compute max(|A+A|, |A*A|) for a set of distinct positive integers."""
    n = len(A)
    sumset = set()
    prodset = set()
    for i in range(n):
        for j in range(i, n):
            sumset.add(A[i] + A[j])
            prodset.add(A[i] * A[j])
    return max(len(sumset), len(prodset))

def _local_search(A, max_time, rng):
    """Swap-based local search to reduce max(|A+A|,|A*A|)."""
    A = list(A)
    A.sort()
    n = len(A)
    current = _score_set(A)
    best = current
    best_A = list(A)
    
    # Candidate pool: numbers 1..10000
    # We'll try replacing one element with a random candidate
    # To be efficient, precompute candidate ranges
    
    deadline = max_time
    
    while time.time() < deadline:
        improved = False
        # Try all single-swap moves
        for i in range(n):
            old_val = A[i]
            # Try replacing A[i] with various candidates
            # Use a mix of nearby and random candidates
            candidates = set()
            # Nearby values
            for delta in range(-50, 51):
                v = old_val + delta
                if 1 <= v <= 10000 and v != old_val:
                    candidates.add(v)
            # Some random candidates
            for _ in range(30):
                v = rng.randint(1, 10000)
                if v != old_val:
                    candidates.add(v)
            # Some structured candidates (powers, multiples)
            for p in [2, 3, 5]:
                v = 1
                while v <= 10000:
                    if v != old_val:
                        candidates.add(v)
                    v *= p
            
            for v in candidates:
                if v in A:
                    continue
                # Check: replacing A[i] with v, keep sorted
                new_A = list(A)
                new_A[i] = v
                new_A.sort()
                # Check distinctness (already ensured by v not in A)
                s = _score_set(new_A)
                if s < current:
                    current = s
                    A = new_A
                    improved = True
                    if current < best:
                        best = current
                        best_A = list(A)
                    break  # restart scan after improvement
        
        if not improved:
            # Stuck: try random perturbation (kick)
            for _ in range(20):
                i = rng.randint(0, n-1)
                v = rng.randint(1, 10000)
                if v in A:
                    continue
                new_A = list(A)
                new_A[i] = v
                new_A.sort()
                s = _score_set(new_A)
                if s <= current + 2:  # accept slightly worse to escape local min
                    A = new_A
                    current = s
                    break
    
    return best_A, best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    rng = random.Random(42)
    
    # Multiple diverse initializations
    initializations = []
    
    # 1. Powers of 2 (geometric)
    initializations.append([2**i for i in range(20)])
    
    # 2. Small consecutive
    initializations.append(list(range(1, 21)))
    
    # 3. Powers of 3 (but capped at 10000)
    p3 = [3**i for i in range(20) if 3**i <= 10000]
    if len(p3) < 20:
        # Pad with nearby values
        last = p3[-1]
        while len(p3) < 20:
            last += 1
            if last not in p3:
                p3.append(last)
    initializations.append(p3[:20])
    
    # 4. Mixed: small numbers with multiplicative structure
    # Use numbers that are products of small primes
    base_primes = [2, 3, 5, 7]
    structured = set()
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    v = (2**a) * (3**b) * (5**c) * (7**d)
                    if 1 <= v <= 10000:
                        structured.add(v)
    structured_list = sorted(structured)
    if len(structured_list) >= 20:
        # Pick 20 with good spread
        step = len(structured_list) // 20
        initializations.append([structured_list[i*step] for i in range(20)])
    
    # 5. Random sets
    for _ in range(5):
        A = set()
        while len(A) < 20:
            A.add(rng.randint(1, 10000))
        initializations.append(sorted(A))
    
    # 6. Geometric with ratio ~1.5
    geo = [1]
    for i in range(1, 20):
        geo.append(int(geo[-1] * 1.5) + 1)
        if geo[-1] > 10000:
            geo[-1] = 10000
            # Ensure distinct
            while geo[-1] in geo[:-1]:
                geo[-1] -= 1
    initializations.append(geo)
    
    # 7. Numbers with many divisors (to create product collisions)
    # e.g., multiples of 60, 120, etc.
    div_rich = []
    for v in range(1, 10001):
        # Count divisors roughly
        d = 0
        for k in range(1, int(v**0.5)+1):
            if v % k == 0:
                d += 2
                if k*k == v:
                    d -= 1
        if d >= 12:
            div_rich.append(v)
    if len(div_rich) >= 20:
        initializations.append(div_rich[:20])
    
    best_A = None
    best_score = float('inf')
    
    for init in initializations:
        if time.time() > deadline - 5:
            break
        # Ensure valid: 20 distinct positive integers <= 10000
        init = list(init)
        # Remove duplicates and out-of-range
        valid = []
        seen = set()
        for x in init:
            if 1 <= x <= 10000 and x not in seen:
                valid.append(x)
                seen.add(x)
        while len(valid) < 20:
            v = rng.randint(1, 10000)
            if v not in seen:
                valid.append(v)
                seen.add(v)
        valid = valid[:20]
        valid.sort()
        
        # Quick score
        s = _score_set(valid)
        if s < best_score:
            best_score = s
            best_A = list(valid)
        
        # Local search with time budget
        remaining = deadline - time.time() - 2
        if remaining > 1:
            ls_A, ls_score = _local_search(valid, time.time() + remaining, rng)
            if ls_score < best_score:
                best_score = ls_score
                best_A = ls_A
    
    # Final polish: if we have time, do more local search on best
    remaining = deadline - time.time() - 1
    if remaining > 2 and best_A is not None:
        ls_A, ls_score = _local_search(best_A, time.time() + remaining, rng)
        if ls_score < best_score:
            best_score = ls_score
            best_A = ls_A
    
    if best_A is None:
        return list(range(1, 21))
    
    return best_A
