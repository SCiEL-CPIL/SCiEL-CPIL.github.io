import time
from math import gcd

# EVOLVE-BLOCK-START

def _is_valid_set(nums):
    """Check if all 2^k subset products are distinct."""
    products = {1}
    for n in nums:
        new_products = {x * n for x in products}
        if new_products & products:
            return False
        products |= new_products
    return True

def _can_add(current_products, candidate):
    """Check if adding candidate keeps all subset products distinct."""
    new_products = {x * candidate for x in current_products}
    return len(new_products & current_products) == 0

def _build_base_set():
    """
    Build the maximum-size set of distinct integers in [2, 40] such that
    all subset products are distinct.
    
    Strategy: Greedy selection prioritizing numbers with few prime factors
    (which are "independent") and then filling with composites that don't
    create collisions.
    
    The key insight is that if we select numbers whose prime factorization
    vectors are linearly independent over the subset-sum sense, we get
    distinct products. Prime powers with distinct subset-sum exponents
    work well.
    """
    # Start with prime powers that have good subset-sum properties
    # For prime 2: powers 2, 4, 16 (exponents 1, 2, 4 - binary weights)
    # For prime 3: powers 3, 9 (exponents 1, 2)
    # For prime 5: powers 5, 25 (exponents 1, 2)
    # Remaining primes: 7, 11, 13, 17, 19, 23, 29, 31, 37
    
    base = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Verify the base set is valid
    if not _is_valid_set(base):
        # Fallback: just use primes
        base = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    return base

def construct():
    """
    Returns a list of distinct integers in the range 2..40 such that
    all 2^k subset products are distinct. Maximizes k.
    
    The optimal solution uses prime powers with carefully chosen exponents
    to achieve 16 elements.
    """
    t0 = time.time()
    deadline = t0 + 55.0  # 55 second budget (5s margin before 60s kill)
    
    # Start with the known optimal base set
    result = _build_base_set()
    current_products = {1}
    for n in result:
        current_products |= {x * n for x in current_products}
    
    # Try to add more numbers from [2, 40] that aren't already in the set
    candidates = [x for x in range(2, 41) if x not in result]
    
    # Sort candidates by number of distinct prime factors (fewer = more likely to fit)
    def count_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                count += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    candidates.sort(key=count_prime_factors)
    
    for c in candidates:
        if time.time() > deadline:
            break
        if _can_add(current_products, c):
            result.append(c)
            current_products |= {x * c for x in current_products}
    
    # Ensure all elements are Python int
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
