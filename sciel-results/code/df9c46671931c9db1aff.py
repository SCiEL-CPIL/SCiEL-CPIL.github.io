import time
from itertools import combinations

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
    """Check if adding candidate to the set preserves distinct subset products."""
    new_products = {x * candidate for x in current_products}
    return not (new_products & current_products)

def construct():
    """
    Returns a list of distinct integers in the range 2..40 such that
    all 2^k subset products are distinct. Maximizes k.
    
    Uses a greedy + local search approach:
    1. Start with a known valid set of primes.
    2. Try to add more numbers by checking if they maintain the property.
    3. Use prime powers with binary-weighted exponents to pack more numbers.
    """
    deadline = time.time() + 80  # Safety margin before 90s limit
    
    # All candidate numbers 2..40
    candidates = list(range(2, 41))
    
    # Strategy: Use prime powers with carefully chosen exponents
    # For each prime p, we can include p^e1, p^e2, ... as long as
    # the exponent vectors remain linearly independent.
    # 
    # Key insight: If we use powers with exponents that are powers of 2
    # (1, 2, 4, 8, ...), then subset sums of exponents are all distinct,
    # ensuring distinct subset products.
    #
    # Let's build a set using this approach:
    # - For prime 2: use 2^1, 2^2, 2^4 = 2, 4, 16 (exponents 1,2,4)
    #   Could also try 2^8=256 > 40, so max exponent is 5 (2^5=32)
    #   With exponents 1,2,4: subset sums are {0,1,2,3,4,5,6,7} - all distinct
    # - For prime 3: use 3^1, 3^2 = 3, 9 (exponents 1,2)
    #   3^4=81 > 40, so max exponent 3 (3^3=27)
    #   With exponents 1,2: subset sums {0,1,2,3} - all distinct
    # - For prime 5: use 5^1 = 5 (5^2=25, 5^3=125>40)
    #   With exponents 1,2: 5, 25 - subset sums {0,1,2,3} - all distinct
    # - For primes 7,11,13,17,19,23,29,31,37: just the prime itself
    #
    # Total: 3 + 2 + 2 + 9 = 16 numbers
    # Let's verify this works.
    
    # Build candidate set
    base_set = []
    
    # Prime 2: exponents 1, 2, 4 -> 2, 4, 16
    base_set.extend([2, 4, 16])
    
    # Prime 3: exponents 1, 2 -> 3, 9
    base_set.extend([3, 9])
    
    # Prime 5: exponents 1, 2 -> 5, 25
    base_set.extend([5, 25])
    
    # Primes 7, 11, 13, 17, 19, 23, 29, 31, 37
    base_set.extend([7, 11, 13, 17, 19, 23, 29, 31, 37])
    
    # Verify base_set is valid
    if not _is_valid_set(base_set):
        # Fallback to just primes
        base_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    best_set = list(base_set)
    best_products = {1}
    for n in best_set:
        best_products = {x * n for x in best_products}
    
    # Now try to add more numbers from remaining candidates
    remaining = [x for x in candidates if x not in best_set]
    
    # Try to add each remaining number if it doesn't conflict
    for cand in remaining:
        if time.time() > deadline:
            break
        if _can_add(best_products, cand):
            best_set.append(cand)
            best_products = {x * cand for x in best_products}
    
    # Try local search: swap one element for another to potentially add more
    # This is a structural improvement attempt
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_size = len(best_set)
        for i in range(current_size):
            for cand in remaining:
                if time.time() > deadline:
                    break
                # Try replacing best_set[i] with cand
                trial_set = best_set[:i] + [cand] + best_set[i+1:]
                if _is_valid_set(trial_set):
                    # Check if we can then add more
                    trial_products = {1}
                    for n in trial_set:
                        trial_products = {x * n for x in trial_products}
                    # Try to add any remaining
                    for extra in remaining:
                        if extra not in trial_set and _can_add(trial_products, extra):
                            trial_set.append(extra)
                            trial_products = {x * extra for x in trial_products}
                    if len(trial_set) > current_size:
                        best_set = trial_set
                        best_products = trial_products
                        remaining = [x for x in candidates if x not in best_set]
                        improved = True
                        break
            if improved:
                break
    
    # Final verification
    if not _is_valid_set(best_set):
        # Fallback to known valid set
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Ensure all are Python ints
    result = [int(x) for x in best_set]
    
    return result

# EVOLVE-BLOCK-END
