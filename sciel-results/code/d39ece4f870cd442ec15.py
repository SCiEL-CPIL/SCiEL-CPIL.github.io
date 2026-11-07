import time
import sys
from itertools import combinations

sys.setrecursionlimit(10000)

_START_TIME = time.time()
_DEADLINE = _START_TIME + 55.0


def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


def _prime_factors(n):
    """Return list of prime factors with multiplicity."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def _subset_products_distinct(nums):
    """Check if all 2^k subset products of nums are distinct."""
    if not nums:
        return True
    products = {1}
    for x in nums:
        new_prods = set()
        for p in products:
            np = p * x
            if np in products:
                return False
            new_prods.add(np)
        products |= new_prods
    return True


def _count_conflicts(nums):
    """Count number of pairs (i,j) where adding nums[i] to a set containing nums[j] causes conflict."""
    # This is a simplified heuristic: count how many pairs have a product collision
    # For a more accurate count, we'd need to check all subset products
    pass


def _get_order_key(x):
    """Heuristic ordering: primes first, then prime powers, then by complexity."""
    if _is_prime(x):
        return (0, x)
    
    # Check if prime power
    for p in range(2, 41):
        if _is_prime(p):
            for power in range(2, 7):
                if x == p ** power:
                    return (1, power, x)
    
    # Count total prime factors with multiplicity
    factors = _prime_factors(x)
    return (2, len(factors), x)


def _greedy_construct(candidates):
    """Greedy construction with heuristic ordering."""
    ordered = sorted(candidates, key=_get_order_key)
    products = {1}
    selected = []
    
    for x in ordered:
        if time.time() > _DEADLINE:
            break
        
        # Check if we can add x
        can_add = True
        new_prods = set()
        for p in products:
            np = p * x
            if np in products:
                can_add = False
                break
            new_prods.add(np)
        
        if can_add:
            products |= new_prods
            selected.append(x)
    
    return selected


def _local_search(nums, max_iter=1000):
    """Improve the set by trying to swap elements."""
    if not nums:
        return nums
    
    current = list(nums)
    best = len(current)
    candidates = list(range(2, 41))
    
    for _ in range(max_iter):
        if time.time() > _DEADLINE:
            break
        
        improved = False
        # Try adding each candidate
        for x in candidates:
            if x not in current:
                test = current + [x]
                if _subset_products_distinct(test):
                    current = test
                    improved = True
                    best = max(best, len(current))
                    break
        
        if improved:
            continue
        
        # Try swapping
        for i in range(len(current)):
            for x in candidates:
                if x in current:
                    continue
                test = current[:i] + [x] + current[i+1:]
                if _subset_products_distinct(test):
                    current = test
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return current


def construct():
    """
    Find the largest set of distinct integers in 2..40 such that all
    2^k subset products (including empty product = 1) are distinct.
    """
    candidates = list(range(2, 41))
    
    # Start with greedy
    best = _greedy_construct(candidates)
    
    # Try local search
    improved = _local_search(best, max_iter=2000)
    if len(improved) > len(best):
        best = improved
    
    # Try multiple random restarts with different orderings
    import random
    for _ in range(50):
        if time.time() > _DEADLINE:
            break
        
        # Random perturbation of ordering
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        # Mix in heuristic ordering
        ordered_heuristic = sorted(candidates, key=_get_order_key)
        # Create a mixed ordering
        mixed = []
        remaining = list(ordered_heuristic)
        for _ in range(len(candidates)):
            if random.random() < 0.5 and remaining:
                # Take from heuristic
                idx = remaining.pop(random.randint(0, len(remaining)-1))
                mixed.append(idx)
            elif shuffled:
                idx = shuffled.pop(0)
                if idx in remaining:
                    remaining.remove(idx)
                mixed.append(idx)
        
        result = _greedy_construct(mixed)
        if len(result) > len(best):
            best = result
            # Local search on the new best
            improved = _local_search(best, max_iter=500)
            if len(improved) > len(best):
                best = improved
    
    # Final verification
    if not _subset_products_distinct(best):
        # Fallback to just primes
        best = [x for x in range(2, 41) if _is_prime(x)]
    
    # Ensure all elements are Python int
    return [int(x) for x in best]
