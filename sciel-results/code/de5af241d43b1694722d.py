import time
import random
from itertools import combinations

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _factorize(n, primes):
    """Return dict of prime -> exponent for n."""
    factors = {}
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            exp = 0
            while n % p == 0:
                n //= p
                exp += 1
            factors[p] = exp
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _all_subset_products_distinct(candidates, primes):
    """Check if all 2^k subset products are distinct by checking if all subset sum vectors are distinct."""
    k = len(candidates)
    if k == 0:
        return True
    if k > 20:
        # Too many to check efficiently with naive approach
        return False
    
    # For each candidate, compute its prime factorization vector
    # Two subset products are equal iff their prime exponent vectors are equal
    # So we need all 2^k subset sum vectors to be distinct
    # This is equivalent to: no non-empty subset has product 1 (which is impossible since all >= 2)
    # Actually, we need: for any two distinct subsets S1, S2, product(S1) != product(S2)
    # This means product(S1) / product(S2) != 1, i.e., product(S1 \ S2) != product(S2 \ S1)
    # More directly: all 2^k subset products must be distinct.
    
    # Efficient check: compute all subset products and check for duplicates
    # But 2^18 = 262144, which is manageable.
    
    # Actually, a better approach: two subsets have the same product iff their symmetric difference
    # has product 1 in some sense... No, let's just enumerate.
    
    # For k <= 18, 2^18 = 262144 products. Each product can be huge but Python handles big ints.
    # However, comparing big ints is expensive. Let's use a smarter approach.
    
    # Two subsets S1 and S2 have the same product iff for every prime p, the sum of exponents
    # in S1 equals the sum of exponents in S2. So we can represent each subset by its exponent vector
    # and check if all 2^k vectors are distinct.
    
    # The exponent vector for a subset is the sum of the exponent vectors of its elements.
    # We need all 2^k such sums to be distinct.
    
    # This is equivalent to: the only subset with exponent vector (0,0,...,0) is the empty set.
    # Because if two subsets have the same vector, their difference (symmetric difference with signs)
    # gives a signed combination that sums to zero.
    
    # Actually, the cleanest check: enumerate all 2^k subset products as tuples of exponents and check uniqueness.
    
    num_primes = len(primes)
    
    # Build exponent vectors for each candidate
    exp_vecs = []
    for c in candidates:
        factors = _factorize(c, primes)
        vec = [0] * num_primes
        for p, e in factors.items():
            idx = primes.index(p) if p in primes else -1
            if idx >= 0:
                vec[idx] = e
        exp_vecs.append(vec)
    
    # Now enumerate all 2^k subset sums and check for duplicates
    seen = set()
    seen.add(tuple([0] * num_primes))
    
    # Use iterative approach
    current_sums = {tuple([0] * num_primes)}
    for i in range(k):
        new_sums = set()
        vec = exp_vecs[i]
        for s in current_sums:
            new_s = tuple(s[j] + vec[j] for j in range(num_primes))
            if new_s in seen:
                return False
            new_sums.add(new_s)
        seen.update(new_sums)
        current_sums = new_sums
    
    return True

def construct():
    """
    Returns up to 18 distinct integers in [2, 40] such that all 2^k subset products are distinct.
    
    Uses a greedy approach with prime-based factorization analysis.
    """
    start_time = time.time()
    deadline = start_time + 85  # 85 second deadline with 5s safety margin
    
    primes = [p for p in range(2, 41) if _is_prime(p)]
    
    # Start with the known good solution of 16 elements
    base_solution = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Verify base solution
    if _all_subset_products_distinct(base_solution, primes):
        current_best = list(base_solution)
    else:
        current_best = []
    
    # Try to add more elements
    # Candidate elements not in base_solution
    all_candidates = list(range(2, 41))
    for c in all_candidates:
        if c in current_best:
            continue
        if time.time() > deadline:
            break
        
        trial = current_best + [c]
        if _all_subset_products_distinct(trial, primes):
            current_best = trial
    
    # Try different orderings and combinations
    # Use randomized search to find larger valid sets
    random.seed(42)
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        # Randomly sample a set of size 17 or 18
        target_size = min(18, len(all_candidates))
        if random.random() < 0.5:
            target_size = 17
        else:
            target_size = 18
        
        if target_size > len(all_candidates):
            target_size = len(all_candidates)
        
        sample = random.sample(all_candidates, target_size)
        
        if _all_subset_products_distinct(sample, primes):
            if len(sample) > len(current_best):
                current_best = sample
    
    # Try greedy construction from scratch with different heuristics
    for heuristic in ['smallest_first', 'largest_first', 'primes_first']:
        if time.time() > deadline:
            break
        
        if heuristic == 'smallest_first':
            ordered = sorted(all_candidates)
        elif heuristic == 'largest_first':
            ordered = sorted(all_candidates, reverse=True)
        else:  # primes_first
            ordered = sorted(all_candidates, key=lambda x: (0 if _is_prime(x) else 1, x))
        
        greedy_set = []
        for c in ordered:
            if time.time() > deadline:
                break
            trial = greedy_set + [c]
            if _all_subset_products_distinct(trial, primes):
                greedy_set = trial
        
        if len(greedy_set) > len(current_best):
            current_best = greedy_set
    
    # Ensure return type is list of Python ints
    return [int(x) for x in current_best]
