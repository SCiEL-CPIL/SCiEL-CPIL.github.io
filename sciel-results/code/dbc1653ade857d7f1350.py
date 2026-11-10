"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 40, 'maximum_size': 18}

# Precompute prime factorization vectors for all numbers 2..40
# Primes in [2, 40]: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
NUM_PRIMES = len(PRIMES)

def _factor_vector(num):
    """Return a tuple of exponents of PRIMES in num's factorization."""
    vec = [0] * NUM_PRIMES
    x = num
    for i, p in enumerate(PRIMES):
        while x % p == 0:
            vec[i] += 1
            x //= p
    return tuple(vec)

# Precompute factor vectors for all candidates
FACTOR_VECS = {}
for n in range(2, 41):
    FACTOR_VECS[n] = _factor_vector(n)

def _check_distinct_products_fast(nums):
    """
    Check if all 2^k subset products are distinct using prime exponent vectors.
    Two subset products are equal iff their exponent vectors are equal.
    We use a set of exponent vectors (tuples) to check for collisions.
    """
    k = len(nums)
    if k == 0:
        return True
    # Start with the empty product: zero vector
    seen = {tuple([0] * NUM_PRIMES)}
    # For each new number, add its vector to all existing subset vectors
    # This is O(2^k * k * NUM_PRIMES) which is fine for k <= 18
    # Actually, let's do it more efficiently by building incrementally
    
    # Better approach: use bitmask enumeration but with vector addition
    # For k up to 18, 2^18 = 262144, which is manageable
    
    # Precompute the vectors
    vecs = [FACTOR_VECS[n] for n in nums]
    
    # Use a set to track seen exponent vectors
    # Represent each vector as a tuple of small ints
    seen_vectors = set()
    seen_vectors.add(tuple([0] * NUM_PRIMES))  # empty subset
    
    # Incremental build: for each element, double the set of vectors
    # This is more memory efficient and faster for small k
    current_vectors = [tuple([0] * NUM_PRIMES)]
    
    for i in range(k):
        v = vecs[i]
        # For each existing vector, create a new one by adding v
        new_vectors = []
        for cv in current_vectors:
            nv = tuple(cv[j] + v[j] for j in range(NUM_PRIMES))
            if nv in seen_vectors:
                return False
            seen_vectors.add(nv)
            new_vectors.append(nv)
        current_vectors.extend(new_vectors)
    
    return True

def construct():
    deadline = time.time() + 80  # 80s safety margin before 90s limit
    start = time.time()
    
    # Strategy: We know that 12 primes work. We need to find 6 more.
    # Key insight: Numbers that share primes with already-chosen numbers can still work
    # as long as no two distinct subsets produce the same product.
    
    # Start with all 12 primes as the base
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Candidates: all composites in [4, 40]
    prime_set = set(primes)
    candidates = [n for n in range(4, 41) if n not in prime_set]
    
    # Sort candidates by a heuristic: prefer numbers with fewer shared prime factors
    # with the current set, and prefer numbers with larger "new" prime content
    def conflict_score(cand):
        pf = FACTOR_VECS[cand]
        # Count how many primes in cand are already in the prime_set
        shared = 0
        for i, p in enumerate(PRIMES):
            if pf[i] > 0 and p in prime_set:
                shared += 1
        # Also consider the magnitude - larger numbers with unique factors are better
        # But for now, just use shared count
        return shared
    
    candidates.sort(key=conflict_score)
    
    current = list(primes)
    
    # Greedily add candidates
    for cand in candidates:
        if len(current) >= 18:
            break
        if time.time() > deadline:
            break
        trial = current + [cand]
        if _check_distinct_products_fast(trial):
            current.append(cand)
            # Update prime set to include primes from cand
            pf = FACTOR_VECS[cand]
            for i, p in enumerate(PRIMES):
                if pf[i] > 0:
                    prime_set.add(p)
    
    # If greedy didn't reach 18, try backtracking
    if len(current) < 18:
        all_candidates = list(range(2, 41))
        
        # Use a smarter backtracking with pruning
        def backtrack(chosen, chosen_set, idx):
            if time.time() > deadline:
                return None
            if len(chosen) == 18:
                return chosen[:]
            
            # Pruning: not enough candidates left
            remaining = len(all_candidates) - idx
            if len(chosen) + remaining < 18:
                return None
            
            for i in range(idx, len(all_candidates)):
                cand = all_candidates[i]
                if cand in chosen_set:
                    continue
                
                trial = chosen + [cand]
                if _check_distinct_products_fast(trial):
                    result = backtrack(trial, chosen_set | {cand}, i + 1)
                    if result is not None:
                        return result
            
            return None
        
        # Try backtracking starting from empty set
        result = backtrack([], set(), 0)
        if result is not None:
            current = result
    
    # Final validation
    if len(current) >= 2:
        assert _check_distinct_products_fast(current), "Distinct products check failed"
    
    # Ensure we return Python ints in a list
    return [int(x) for x in current]

# EVOLVE-BLOCK-END
