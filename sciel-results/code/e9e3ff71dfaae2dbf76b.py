"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

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

def _subset_products_set(lst):
    """Return the set of all subset products of lst."""
    k = len(lst)
    prods = {1}
    for a in lst:
        new_prods = set()
        for p in prods:
            np = p * a
            new_prods.add(np)
        prods |= new_prods
    return prods

def _can_add(lst, candidate, current_prods):
    """Check if candidate can be added to lst without creating duplicate products."""
    k = len(lst)
    new_prods = set()
    for p in current_prods:
        np = p * candidate
        new_prods.add(np)
    # Check for overlap with existing products
    for np in new_prods:
        if np in current_prods:
            return False
    return True

def _try_add(lst, candidate, current_prods):
    """Try to add candidate to lst, returning (success, new_lst, new_prods)."""
    if not _can_add(lst, candidate, current_prods):
        return False, lst, current_prods
    new_lst = lst + [candidate]
    new_prods = set(current_prods)
    for p in current_prods:
        new_prods.add(p * candidate)
    return True, new_lst, new_prods

def _is_valid_set(lst):
    """Check if all subset products of lst are distinct."""
    prods = _subset_products_set(lst)
    return len(prods) == (1 << len(lst))

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s margin for 60s limit

    # Strategy: Start with all primes in [2, 40], then try to add composites.
    primes = [n for n in range(2, 41) if _is_prime(n)]
    composites = [n for n in range(2, 41) if not _is_prime(n)]
    
    # Sort composites by "interestingness" - prefer those with fewer prime factors
    # and smaller values first, as they're more likely to fit
    def composite_score(n):
        # Count distinct prime factors
        factors = set()
        m = n
        d = 2
        while d * d <= m:
            while m % d == 0:
                factors.add(d)
                m //= d
            d += 1
        if m > 1:
            factors.add(m)
        return (len(factors), n)
    
    composites.sort(key=composite_score)
    
    # Start with all primes
    selected = primes[:]
    current_prods = _subset_products_set(selected)
    
    # Try to add composites
    for c in composites:
        if time.time() > deadline:
            break
        success, new_lst, new_prods = _try_add(selected, c, current_prods)
        if success:
            selected = new_lst
            current_prods = new_prods
    
    # If we have fewer than 16, try a different approach: greedy from scratch
    if len(selected) < 16:
        selected = []
        current_prods = {1}
        # Try all numbers in [2, 40], preferring smaller and prime ones
        candidates = list(range(2, 41))
        candidates.sort(key=lambda x: (0 if _is_prime(x) else 1, x))
        
        for c in candidates:
            if time.time() > deadline:
                break
            success, new_lst, new_prods = _try_add(selected, c, current_prods)
            if success:
                selected = new_lst
                current_prods = new_prods
    
    # Final verification
    if not _is_valid_set(selected):
        # Fallback to just primes
        selected = primes
    
    # Ensure we return a list of Python ints
    return [int(x) for x in selected]
# EVOLVE-BLOCK-END
