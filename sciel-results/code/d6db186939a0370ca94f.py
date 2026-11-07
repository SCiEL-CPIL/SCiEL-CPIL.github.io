import random
import time

def _prime_factors(n):
    """Return tuple of prime exponents for n."""
    factors = []
    d = 2
    while d * d <= n:
        count = 0
        while n % d == 0:
            count += 1
            n //= d
        if count:
            factors.append((d, count))
        d += 1
    if n > 1:
        factors.append((n, 1))
    return tuple(factors)

def _subset_products_set(nums):
    """Return the set of all 2^k subset products."""
    products = {1}
    for num in nums:
        new_products = set()
        for p in products:
            new_products.add(p * num)
        products.update(new_products)
    return products

def _check_collision_incremental(current_products, new_num):
    """Check if adding new_num creates a collision. Returns (valid, new_products_set)."""
    # For each existing product p, p * new_num must not already be in current_products
    # and must not collide with any other p' * new_num
    new_products = set()
    for p in current_products:
        np = p * new_num
        if np in current_products:
            return False, None
        if np in new_products:
            return False, None
        new_products.add(np)
    return True, current_products | new_products

def _verify_subset_products(nums):
    """Verify all 2^k subset products are distinct."""
    k = len(nums)
    if k > 20:
        return False
    products = _subset_products_set(nums)
    return len(products) == (1 << k)

def construct():
    """Return a list of up to 18 distinct integers in 2..40 with distinct subset products."""
    
    # Known optimal candidate (size 16)
    best_set = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    if not _verify_subset_products(best_set):
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
    best_size = len(best_set)
    best_result = best_set[:]
    
    # Pool of all valid numbers
    all_nums = list(range(2, 41))
    
    # Precompute prime factorization vectors for all candidates
    prime_vecs = {n: _prime_factors(n) for n in all_nums}
    
    # 1. Try to extend the known best set by 1 using incremental check
    current_products = _subset_products_set(best_set)
    current_set = set(best_set)
    for num in all_nums:
        if num not in current_set:
            valid, new_products = _check_collision_incremental(current_products, num)
            if valid:
                return list(current_set) + [num]
    
    # 2. Randomized greedy growth with incremental product set maintenance
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline, leaving 10s safety margin
    
    def random_set(size):
        return random.sample(all_nums, size)
    
    # Strategy: Multiple random restarts with greedy growth
    max_restarts = 2000
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Start with a random set of size 14-16
        start_size = random.choice([14, 15, 16])
        current = set(random_set(start_size))
        
        # Verify the starting set is valid
        if not _verify_subset_products(list(current)):
            continue
        
        current_products = _subset_products_set(list(current))
        
        # Greedy growth: repeatedly try to add elements
        changed = True
        while changed and time.time() < deadline:
            changed = False
            pool = [n for n in all_nums if n not in current]
            random.shuffle(pool)
            
            for num in pool:
                valid, new_products = _check_collision_incremental(current_products, num)
                if valid:
                    current.add(num)
                    current_products = new_products
                    changed = True
                    if len(current) > best_size:
                        best_size = len(current)
                        best_result = list(current)
                    break  # Restart pool check as set changed
        
        if best_size >= 18:
            break
        if best_size >= 17:
            # Do a few more restarts to see if we can get 18
            if restart > 50:
                break
    
    # 3. If we have 17, try local search to find 18
    if best_size == 17:
        current = set(best_result)
        current_products = _subset_products_set(list(current))
        
        for _ in range(5000):
            if time.time() > deadline:
                break
            
            # Try to swap: remove one, add another
            to_remove = random.choice(list(current))
            current.remove(to_remove)
            
            # Recompute products after removal (expensive but necessary for correctness)
            current_products = _subset_products_set(list(current))
            
            pool = [n for n in all_nums if n not in current]
            random.shuffle(pool)
            
            added = False
            for num in pool:
                valid, new_products = _check_collision_incremental(current_products, num)
                if valid:
                    current.add(num)
                    current_products = new_products
                    added = True
                    if len(current) > best_size:
                        best_size = len(current)
                        best_result = list(current)
                    break
            
            if not added:
                current.add(to_remove)
                current_products = _subset_products_set(list(current))
            
            if best_size >= 18:
                break
    
    # Ensure types and validity
    result = []
    seen = set()
    for x in best_result:
        if isinstance(x, int) and not isinstance(x, bool) and 2 <= x <= 40 and x not in seen:
            seen.add(x)
            result.append(x)
    
    # Final verification
    if not _verify_subset_products(result):
        # Fallback to the known 16-element set
        result = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    return result
