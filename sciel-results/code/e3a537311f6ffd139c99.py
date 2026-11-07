"""Rewritten program: greedy + exact subset-product collision check to maximize distinct integers in 2..40 with all 2^k subset products distinct."""
import time
import random
from math import isqrt

_P0 = time.time()
_DEADLINE = _P0 + 80.0  # safety margin under 90s wall-clock limit


def _is_prime(a):
    if a < 2:
        return False
    if a < 4:
        return True
    if a % 2 == 0 or a % 3 == 0:
        return False
    i = 5
    while i * i <= a:
        if a % i == 0 or a % (i + 2) == 0:
            return False
        i += 6
    return True


def _subset_products_set(nums):
    """Return the set of all subset products of nums (including empty product 1)."""
    prods = {1}
    for x in nums:
        add = set()
        for p in prods:
            q = p * x
            add.add(q)
        prods = prods | add
    return prods


def _can_add(nums, prods, x):
    """True iff adding x keeps all subset products distinct."""
    # For every p in prods, p*x must not already be in prods
    # (p*x values are mutually distinct since x != 1 and p's are distinct)
    for p in prods:
        if p * x in prods:
            return False
    return True


def _try_greedy_order(order):
    """Given a candidate ordering, greedily add numbers if they don't cause collisions."""
    result = []
    prods = {1}
    for x in order:
        if _can_add(result, prods, x):
            result.append(x)
            # Update prods
            new_prods = set()
            for p in prods:
                new_prods.add(p * x)
            prods = prods | new_prods
    return result


def construct():
    """Find the maximum number of distinct integers in 2..40 such that all 2^k subset products are distinct."""
    candidates = list(range(2, 41))
    
    # Start with a baseline: primes (always work since prime factorizations are unique)
    primes = [a for a in range(2, 41) if _is_prime(a)]
    best_set = _try_greedy_order(primes)
    best_len = len(best_set)
    
    # Try different orders to potentially get more than just primes
    # The key insight: we can include composites if they don't create duplicate subset products
    # This happens when the prime factorization exponents across all selected numbers allow
    # unique encoding of each subset as a vector of exponents.
    
    # Strategy: try random orders multiple times
    random.seed(42)  # deterministic for reproducibility
    
    num_attempts = 0
    while time.time() < _DEADLINE - 5.0:
        num_attempts += 1
        
        # Generate a random order with some structure
        # Mix of approaches:
        # 1. Pure random
        # 2. Primes first, then random composites
        # 3. Random with bias toward numbers with few prime factors
        # 4. Sorted by various criteria
        
        mode = num_attempts % 5
        
        if mode == 0:
            # Pure random
            order = candidates[:]
            random.shuffle(order)
        elif mode == 1:
            # Primes first, then random composites
            composites = [a for a in range(2, 41) if not _is_prime(a)]
            random.shuffle(composites)
            order = primes + composites
        elif mode == 2:
            # Random, but sort by number of prime factors (fewer first)
            def num_prime_factors(n):
                count = 0
                d = 2
                nn = n
                while d * d <= nn:
                    while nn % d == 0:
                        count += 1
                        nn //= d
                    d += 1
                if nn > 1:
                    count += 1
                return count
            
            order = candidates[:]
            order.sort(key=lambda x: (num_prime_factors(x), random.random()))
            random.shuffle(order)  # break ties randomly
        elif mode == 3:
            # Sorted by value descending
            order = sorted(candidates, reverse=True)
            # Add some random perturbation
            if random.random() < 0.5:
                i = random.randint(0, len(order) - 2)
                order[i], order[i + 1] = order[i + 1], order[i]
        else:
            # Sorted by value ascending
            order = sorted(candidates)
            if random.random() < 0.5:
                i = random.randint(0, len(order) - 2)
                order[i], order[i + 1] = order[i + 1], order[i]
        
        res = _try_greedy_order(order)
        
        if len(res) > best_len:
            best_len = len(res)
            best_set = res
            if best_len == 18:
                break
    
    # Final verification
    if best_set:
        prods = _subset_products_set(best_set)
        if len(prods) != 2 ** len(best_set):
            # Fallback to primes which are guaranteed valid
            best_set = primes
    
    return list(best_set)


if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} integers: {res}")
    if res:
        prods = _subset_products_set(res)
        if len(prods) == 2 ** len(res):
            print(f"Valid. Total products: {len(prods)}")
        else:
            print("ERROR: Invalid subset products found!")

