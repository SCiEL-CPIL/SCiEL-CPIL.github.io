"""Optimized program: search for 12 distinct positive integers <=10000 minimizing distinct primes in pair sums."""
import time
import random
from itertools import combinations
from functools import lru_cache

# Precompute smallest-prime-factor table up to 20000 (max sum = 10000+9999=19999)
_MAX_SUM = 20000
_spf = list(range(_MAX_SUM + 1))
for _i in range(2, int(_MAX_SUM**0.5) + 1):
    if _spf[_i] == _i:
        for _j in range(_i * _i, _MAX_SUM + 1, _i):
            if _spf[_j] == _j:
                _spf[_j] = _i

# Precompute prime factor masks for all numbers up to _MAX_SUM
# Use a list where index n stores the set of prime factors as a frozenset
_pf_cache = {}
_prime_to_idx = {}
_idx_to_prime = []

def _get_prime_idx(p):
    """Get or create index for a prime."""
    if p not in _prime_to_idx:
        _prime_to_idx[p] = len(_idx_to_prime)
        _idx_to_prime.append(p)
    return _prime_to_idx[p]

# Precompute prime factors for all numbers up to _MAX_SUM
_pf_sets = [None] * (_MAX_SUM + 1)
for _n in range(2, _MAX_SUM + 1):
    factors = set()
    x = _n
    while x > 1:
        p = _spf[x]
        factors.add(p)
        while x % p == 0:
            x //= p
    _pf_sets[_n] = factors
_pf_sets[0] = set()
_pf_sets[1] = set()

def _prime_factors_cached(n):
    """Return set of prime factors of n using precomputed table."""
    if n <= _MAX_SUM:
        return _pf_sets[n]
    # Fallback for larger numbers (shouldn't happen with our constraints)
    if n in _pf_cache:
        return _pf_cache[n]
    factors = set()
    x = n
    while x > 1:
        p = _spf[x] if x <= _MAX_SUM else _factor_large(x)
        factors.add(p)
        while x % p == 0:
            x //= p
    _pf_cache[n] = factors
    return factors

def _factor_large(n):
    """Factor a large number > _MAX_SUM."""
    for p in range(2, int(n**0.5) + 1):
        if n % p == 0:
            return p
    return n

def _distinct_primes_in_sums(arr):
    """Count distinct primes dividing any sum of two distinct elements."""
    primes_set = set()
    n = len(arr)
    for i in range(n):
        for j in range(i+1, n):
            s = arr[i] + arr[j]
            primes_set.update(_prime_factors_cached(s))
    return primes_set

def _objective(arr):
    return len(_distinct_primes_in_sums(arr))

def _objective_incremental(arr, removed_idx, added_val):
    """
    Compute objective when replacing arr[removed_idx] with added_val.
    This is faster than recomputing from scratch.
    """
    old_val = arr[removed_idx]
    if old_val == added_val:
        return _objective(arr)
    
    # Start with all primes from pairs not involving removed_idx
    primes_set = set()
    n = len(arr)
    
    # Pairs not involving removed_idx
    for i in range(n):
        if i == removed_idx:
            continue
        for j in range(i+1, n):
            if j == removed_idx:
                continue
            s = arr[i] + arr[j]
            primes_set.update(_prime_factors_cached(s))
    
    # Pairs involving removed_idx (now with added_val)
    for i in range(n):
        if i == removed_idx:
            continue
        s = added_val + arr[i]
        primes_set.update(_prime_factors_cached(s))
    
    return len(primes_set)

def _generate_structured_candidates():
    """Generate diverse structured candidate arrays with high probability of low prime union."""
    structured = []
    
    # 1. Arithmetic Progressions: a + i*d
    for d in range(1, 200):
        for a in range(1, 200):
            arr = [a + i * d for i in range(12)]
            if max(arr) <= 10000 and len(set(arr)) == 12:
                structured.append(arr)
                
    # 2. Multiples of small bases
    for base in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 36, 40, 48, 60]:
        max_mult = 10000 // base
        if max_mult < 12:
            continue
        # Regular multiples
        for start in range(1, 50):
            if start + 11 > max_mult:
                break
            arr = [base * (start + i) for i in range(12)]
            if len(set(arr)) == 12:
                structured.append(arr)
        
        # Random multiples
        for _ in range(20):
            mults = random.sample(range(1, max_mult + 1), 12)
            arr = [base * m for m in mults]
            if len(set(arr)) == 12:
                structured.append(arr)

    # 3. Strong multiples-of-6 candidates
    # All numbers = 6*k, so sums = 6*(k1+k2), prime factors = {2,3} U primes(k1+k2)
    # We want k1+k2 to have few prime factors
    for _ in range(200):
        # Choose k values from a restricted range to keep sums small
        max_k = 10000 // 6
        if random.random() < 0.5:
            # Small k values
            k_vals = random.sample(range(1, 100), 12)
        else:
            # Medium k values  
            k_vals = random.sample(range(1, 500), 12)
        arr = [6 * k for k in k_vals]
        if len(set(arr)) == 12 and max(arr) <= 10000:
            structured.append(arr)
    
    # 4. Small integers
    for _ in range(100):
        arr = random.sample(range(1, 100), 12)
        structured.append(arr)
        
    for _ in range(100):
        arr = random.sample(range(2, 200, 2), 12)
        structured.append(arr)

    for _ in range(100):
        arr = random.sample(range(1, 200, 3), 12)
        structured.append(arr)

    for _ in range(100):
        candidates = list(range(2, 100, 2)) + list(range(3, 100, 3))
        candidates = list(set(candidates))
        arr = random.sample(candidates, 12)
        structured.append(arr)

    return structured

def _local_search(current, current_score, deadline, best_score, best_arr):
    """Local search with smart perturbations. Returns (best_score, best_arr)."""
    improved = True
    no_improve_count = 0
    while improved and time.time() < deadline:
        improved = False
        no_improve_count += 1
        if no_improve_count > 20:
            break
            
        # Try single-element changes using incremental evaluation
        for i in range(12):
            # Strategy 1: Random small value (bias towards small numbers)
            if random.random() < 0.7:
                new_val = random.randint(1, 500)
            else:
                new_val = random.randint(1, 10000)
                
            if new_val in current and new_val != current[i]:
                continue
            if new_val == current[i]:
                continue
            score = _objective_incremental(current, i, new_val)
            if score < current_score:
                current[i] = new_val
                current_score = score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_arr = list(current)
                break
            elif score == current_score and random.random() < 0.1:
                pass

            # Strategy 2: Multiple of a small number
            if not improved:
                base = random.choice([2, 3, 4, 5, 6, 10, 12, 15, 20, 30])
                m = random.randint(1, min(10000 // base, 500))
                new_val = base * m
                if new_val in current and new_val != current[i]:
                    continue
                if new_val == current[i]:
                    continue
                score = _objective_incremental(current, i, new_val)
                if score < current_score:
                    current[i] = new_val
                    current_score = score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_arr = list(current)
                    break

        # Try swapping two elements
        if not improved:
            for _ in range(10):
                i, j = random.sample(range(12), 2)
                cand = list(current)
                if random.random() < 0.5:
                    new_vi = random.randint(1, 200)
                else:
                    new_vi = random.choice([2,3,4,5,6,10,12,15,20,30]) * random.randint(1, 100)
                    
                if random.random() < 0.5:
                    new_vj = random.randint(1, 200)
                else:
                    new_vj = random.choice([2,3,4,5,6,10,12,15,20,30]) * random.randint(1, 100)
                
                if new_vi == new_vj:
                    continue
                cand[i] = new_vi
                cand[j] = new_vj
                if len(set(cand)) != 12 or max(cand) > 10000:
                    continue
                score = _objective(cand)
                if score < current_score:
                    current = cand
                    current_score = score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_arr = list(current)
                    break

    return best_score, best_arr

def _smooth_numbers_up_to(limit, num_primes=4):
    """Generate numbers that are smooth (only small prime factors)."""
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    if num_primes > len(small_primes):
        num_primes = len(small_primes)
    primes_to_use = small_primes[:num_primes]
    
    smooth = set([1])
    for p in primes_to_use:
        new_smooth = set()
        for s in smooth:
            x = s
            while x * p <= limit:
                x *= p
                new_smooth.add(x)
        smooth.update(new_smooth)
    
    return sorted([s for s in smooth if s <= limit])

def _generate_smooth_based_candidates():
    """Generate candidates where numbers are built from smooth multipliers."""
    candidates = []
    
    # Try different bases and smooth multipliers
    for base in [2, 3, 6, 10, 12, 15, 20, 30, 40, 60]:
        max_mult = 10000 // base
        if max_mult < 12:
            continue
        
        # Get smooth numbers up to max_mult
        smooth = _smooth_numbers_up_to(max_mult, num_primes=4)
        if len(smooth) < 12:
            continue
        
        # Try several random selections of smooth multipliers
        for _ in range(50):
            mults = random.sample(smooth, 12)
            arr = [base * m for m in mults]
            if len(set(arr)) == 12 and max(arr) <= 10000:
                candidates.append(arr)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # --- Phase 1: Structured Constructions ---
    best_arr = None
    best_score = float('inf')
    
    structured = _generate_structured_candidates()
    smooth_candidates = _generate_smooth_based_candidates()
    all_candidates = structured + smooth_candidates
    
    # Shuffle to avoid bias
    random.shuffle(all_candidates)
    
    for arr in all_candidates:
        if time.time() > deadline - 15:
            break
        arr = list(dict.fromkeys(arr))
        if len(arr) != 12:
            continue
        if any(x < 1 or x > 10000 for x in arr):
            continue
        score = _objective(arr)
        if score < best_score:
            best_score = score
            best_arr = list(arr)
            if best_score <= 4:
                break
    
    if best_arr is None:
        best_arr = list(range(1, 13))
        best_score = _objective(best_arr)
    
    # --- Phase 2: Local Search from Best ---
    if time.time() < deadline - 10:
        current = list(best_arr)
        current_score = best_score
        local_deadline = min(deadline - 5, start_time + 30)
        best_score, best_arr = _local_search(current, current_score, local_deadline, best_score, best_arr)
    
    # --- Phase 3: Random Restarts ---
    if time.time() < deadline - 5:
        restarts_done = 0
        while time.time() < deadline - 3 and restarts_done < 500:
            restarts_done += 1
            
            r = random.random()
            if r < 0.2:
                arr = random.sample(range(1, 100), 12)
            elif r < 0.4:
                arr = random.sample(range(2, 300, 2), 12)
            elif r < 0.5:
                arr = random.sample(range(3, 300, 3), 12)
            elif r < 0.6:
                arr = random.sample(range(6, 500, 6), 12)
            elif r < 0.7:
                # Smooth-based
                smooth = _smooth_numbers_up_to(1000, num_primes=4)
                if len(smooth) >= 12:
                    mults = random.sample(smooth, 12)
                    base = random.choice([2, 3, 5, 6, 10])
                    arr = [base * m for m in mults]
                    if max(arr) > 10000:
                        arr = random.sample(range(1, 100), 12)
                else:
                    arr = random.sample(range(1, 100), 12)
            else:
                base = random.choice([2, 3, 4, 5, 6, 8, 10, 12, 15, 20])
                max_m = 10000 // base
                if max_m < 12:
                    base = 2
                    max_m = 5000
                mults = random.sample(range(1, max_m + 1), 12)
                arr = [base * m for m in mults]
            
            if len(set(arr)) != 12 or any(x < 1 or x > 10000 for x in arr):
                continue
            
            score = _objective(arr)
            if score < best_score:
                best_score = score
                best_arr = list(arr)
            
            if time.time() < deadline - 3:
                cur = list(arr)
                cur_score = score
                search_deadline = min(deadline - 3, time.time() + 1.5)
                best_score, best_arr = _local_search(cur, cur_score, search_deadline, best_score, best_arr)

    return best_arr

# EVOLVE-BLOCK-END
