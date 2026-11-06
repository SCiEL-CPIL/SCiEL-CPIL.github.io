import time
import random
from collections import Counter

# Precompute smallest prime factor (SPF)
_MAX_SUM = 20000
_spf = list(range(_MAX_SUM + 1))
for _i in range(2, int(_MAX_SUM**0.5) + 1):
    if _spf[_i] == _i:
        for _j in range(_i * _i, _MAX_SUM + 1, _i):
            if _spf[_j] == _j:
                _spf[_j] = _i

# Precompute prime factor sets
_pf_cache = [None] * (_MAX_SUM + 1)
_pf_cache[1] = frozenset()
for _n in range(2, _MAX_SUM + 1):
    _factors = set()
    _x = _n
    while _x > 1:
        _p = _spf[_x]
        _factors.add(_p)
        while _x % _p == 0:
            _x //= _p
    _pf_cache[_n] = frozenset(_factors)

def _get_prime_factors(n):
    return _pf_cache[n]

def _objective(arr):
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            primes.update(_get_prime_factors(arr[i] + arr[j]))
    return len(primes)

def _local_search(arr, deadline):
    """
    Local search with incremental prime counting.
    Returns (best_arr, best_score).
    """
    current = list(arr)
    n = len(current)
    
    # Initialize prime counts
    prime_counts = Counter()
    for i in range(n):
        for j in range(i + 1, n):
            s = current[i] + current[j]
            for p in _get_prime_factors(s):
                prime_counts[p] += 1
    
    current_score = len(prime_counts)
    best_arr = list(current)
    best_score = current_score
    
    # Candidate pool for replacements
    # We want small numbers to keep sums small.
    # Multiples of 6 are good because sums are divisible by 6 (primes 2,3 guaranteed, others smaller).
    candidates = set()
    # Small multiples of 6
    for k in range(1, 200):
        candidates.add(6 * k)
    # Small even numbers
    for k in range(1, 100):
        candidates.add(2 * k)
    # Small multiples of 3
    for k in range(1, 100):
        candidates.add(3 * k)
    # Random small numbers
    for _ in range(100):
        candidates.add(random.randint(1, 200))
    
    candidates = [c for c in candidates if c <= 10000]
    random.shuffle(candidates)
    
    improved = True
    no_improve_count = 0
    max_no_improve = 50
    
    while improved and time.time() < deadline:
        improved = False
        no_improve_count += 1
        if no_improve_count > max_no_improve:
            break
            
        for i in range(n):
            if time.time() > deadline:
                break
            old_val = current[i]
            
            # Try to find a better value for position i
            best_new_val = None
            best_new_score = current_score
            best_new_counts = None
            
            # Sample a subset of candidates to save time, or iterate all if fast enough
            # With N=12, updating counts is 11 * (few primes) ops. Very fast.
            # Let's try all candidates in the pool.
            for new_val in candidates:
                if new_val == old_val:
                    continue
                if new_val in current and current[i] != new_val:
                    # Check if new_val is already in the array at another index
                    # Since we are swapping, if new_val is at index j != i, it's invalid unless we handle swaps.
                    # Simpler: just check if new_val is in the set of other elements.
                    # To be efficient, maintain a set of current elements?
                    # Or just check `new_val in current` and handle the case where it IS current[i] (skipped above).
                    # If new_val is in current at another index, we can't just replace i with it without moving the other one.
                    # So we require new_val NOT in current.
                    continue
                
                # Simulate replacing current[i] with new_val
                # Remove contributions of old_val
                temp_counts = prime_counts.copy() # This is expensive (dict copy)
                # Better: apply delta, check, rollback if not better.
                
                # Actually, copying a Counter of size ~10-20 is fast.
                # But let's do in-place with rollback for speed.
                
                # 1. Remove old_val contributions
                for j in range(n):
                    if j == i:
                        continue
                    s = old_val + current[j]
                    for p in _get_prime_factors(s):
                        temp_counts[p] -= 1
                        if temp_counts[p] == 0:
                            del temp_counts[p]
                
                # 2. Add new_val contributions
                for j in range(n):
                    if j == i:
                        continue
                    s = new_val + current[j]
                    for p in _get_prime_factors(s):
                        if p not in temp_counts:
                            temp_counts[p] = 1
                        else:
                            temp_counts[p] += 1
                
                new_score = len(temp_counts)
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_new_val = new_val
                    # We need to keep the counts if this is the best so far, 
                    # but if we find an even better one later, we discard these.
                    # Storing a copy of the Counter for the best candidate is O(1) relative to the loop.
                    best_new_counts = temp_counts.copy()
            
            if best_new_val is not None:
                # Commit the change
                current[i] = best_new_val
                prime_counts = best_new_counts
                current_score = best_new_score
                improved = True
                no_improve_count = 0
                
                if current_score < best_score:
                    best_score = current_score
                    best_arr = list(current)
    
    return best_arr, best_score

def _generate_initial_candidates():
    """Generate a diverse set of starting arrays."""
    candidates = []
    
    # 1. Consecutive multiples of 6
    for start in range(1, 20):
        arr = [6 * (start + i) for i in range(12)]
        if max(arr) <= 10000:
            candidates.append(arr)
            
    # 2. Small consecutive integers
    for start in range(1, 20):
        arr = list(range(start, start + 12))
        candidates.append(arr)
        
    # 3. Random multiples of 6
    for _ in range(50):
        arr = random.sample([6*i for i in range(1, 200)], 12)
        candidates.append(arr)
        
    # 4. Random multiples of 12
    for _ in range(50):
        arr = random.sample([12*i for i in range(1, 100)], 12)
        candidates.append(arr)
        
    # 5. Small evens
    for _ in range(50):
        arr = random.sample([2*i for i in range(1, 100)], 12)
        candidates.append(arr)
        
    # 6. Mix of small numbers
    for _ in range(50):
        arr = random.sample(range(1, 100), 12)
        candidates.append(arr)
        
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    best_arr = None
    best_score = float('inf')
    
    # Phase 1: Evaluate structured candidates
    initial_candidates = _generate_initial_candidates()
    
    # Sort by a heuristic score (e.g., number of distinct primes in sums) to find good starts quickly
    # But calculating exact score is fast enough for ~200 candidates.
    for arr in initial_candidates:
        if time.time() > deadline - 10:
            break
        score = _objective(arr)
        if score < best_score:
            best_score = score
            best_arr = list(arr)
            if best_score <= 5: # Very good, maybe stop early? No, let's keep searching a bit.
                pass
                
    if best_arr is None:
        best_arr = list(range(1, 13))
        best_score = _objective(best_arr)
        
    # Phase 2: Local search from the best candidate found
    if time.time() < deadline - 5:
        local_deadline = deadline - 5
        best_arr, best_score = _local_search(best_arr, local_deadline)
        
    # Phase 3: Random restarts with local search
    # Use remaining time for random restarts
    restarts = 0
    while time.time() < deadline - 2:
        restarts += 1
        
        # Generate a new random candidate
        r = random.random()
        if r < 0.2:
            arr = random.sample([6*i for i in range(1, 200)], 12)
        elif r < 0.4:
            arr = random.sample([12*i for i in range(1, 100)], 12)
        elif r < 0.6:
            arr = random.sample([2*i for i in range(1, 100)], 12)
        elif r < 0.8:
            arr = random.sample(range(1, 100), 12)
        else:
            # Arithmetic progression
            a = random.randint(1, 50)
            d = random.choice([2, 3, 4, 6, 10, 12])
            arr = [a + i*d for i in range(12)]
            if max(arr) > 10000:
                continue
            
        if len(set(arr)) != 12 or any(x < 1 or x > 10000 for x in arr):
            continue
            
        score = _objective(arr)
        
        # Only do local search if this candidate is promising (better than or close to best)
        # Or if we have lots of time left.
        # To save time, only do local search if score < best_score + 2 or if best_score is high.
        if score < best_score or (score < best_score + 3 and random.random() < 0.5):
             # Short local search
             # Allocate small time slice for this restart
             search_deadline = min(deadline - 2, time.time() + 2.0)
             if search_deadline > time.time() + 0.5:
                 local_arr, local_score = _local_search(arr, search_deadline)
                 if local_score < best_score:
                     best_score = local_score
                     best_arr = local_arr
        
    return best_arr

