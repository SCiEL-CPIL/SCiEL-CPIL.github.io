"""Direct construction for 'Large sets with distinct subset products'."""

import time
import itertools

DEADLINE = 82.0  # wall-clock seconds (90s limit, safety margin)
T_START = time.time()

def _check_distinct_subset_products(items):
    """Verify all 2^n subset products are distinct."""
    n = len(items)
    prods = set()
    for mask in range(1 << n):
        prod = 1
        for i in range(n):
            if mask & (1 << i):
                prod *= items[i]
        if prod in prods:
            return False
        prods.add(prod)
    return True

def _is_subset_sum_free(exponents):
    """Check no element equals subset-sum of other elements."""
    for a in exponents:
        others = [x for x in exponents if x != a]
        for r in range(len(others) + 1):
            for sub in itertools.combinations(others, r):
                if sum(sub) == a:
                    return False
    return True

def construct():
    global DEADLINE, T_START
    deadline = DEADLINE
    if deadline < 0:
        deadline = time.time() + 82.0

    def time_left():
        return deadline - time.time()

    # Primes in [2, 40]
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    # For each prime, find all maximal (by size) sum-free exponent subsets
    # Max exponents: p^m <= 40
    prime_exponents = {}
    for p in primes:
        m = 1
        while p ** (m + 1) <= 40:
            m += 1
        prime_exponents[p] = list(range(1, m + 1))

    # Strategy:
    # 1. For primes with max exponent >= 2 (2,3,5,7), try to use 2 exponents
    #    (sum-free pairs) to get 2 items per such prime
    # 2. For remaining primes, use exponent 1 (single item)
    # 3. Then try to add composite numbers (products of powers from different primes)
    #    that keep all subset products distinct

    # Find best sum-free subset per prime (maximize count)
    best_per_prime = {}
    for p in primes:
        exps = prime_exponents[p]
        best = []
        for k in range(len(exps), -1, -1):
            found = False
            for combo in itertools.combinations(exps, k):
                if _is_subset_sum_free(list(combo)):
                    best = list(combo)
                    found = True
                    break
            if found:
                break
        best_per_prime[p] = best

    # Build initial set from prime powers
    candidate_items = []
    used_values = set()
    for p in primes:
        for e in best_per_prime[p]:
            v = p ** e
            if 2 <= v <= 40 and v not in used_values:
                candidate_items.append(v)
                used_values.add(v)

    # Verify initial set
    if not _check_distinct_subset_products(candidate_items):
        # Fallback: just all primes
        candidate_items = primes[:]
        used_values = set(primes)

    # Now try to add composite numbers (2..40) not already in the set
    # A composite c = p1^a1 * p2^a2 * ... is safe to add if it doesn't
    # create a collision. We check by verifying the full set after addition.
    # To be efficient, we only consider composites whose prime factors are
    # all in primes and whose value is in [2,40].
    
    # Generate all numbers in [2,40] not yet used
    all_numbers = list(range(2, 41))
    remaining = [x for x in all_numbers if x not in used_values]
    
    # Sort by "simplicity" - prefer numbers with fewer prime factors first
    # to reduce collision risk, then by value
    def num_prime_factors(n):
        count = 0
        for p in primes:
            while n % p == 0:
                count += 1
                n //= p
        return count
    
    remaining.sort(key=lambda x: (num_prime_factors(x), x))
    
    # Greedily add composites that preserve the property
    for v in remaining:
        if time_left() < 5.0:
            break
        new_set = candidate_items + [v]
        if _check_distinct_subset_products(new_set):
            candidate_items = new_set
            used_values.add(v)

    # Final verification
    if not _check_distinct_subset_products(candidate_items):
        candidate_items = primes[:]

    # Ensure Python ints and list type
    res = [int(x) for x in candidate_items]
    assert len(res) == len(set(res)), "duplicates"
    assert all(isinstance(x, int) and not isinstance(x, bool) and 2 <= x <= 40 for x in res)

    return list(res)
