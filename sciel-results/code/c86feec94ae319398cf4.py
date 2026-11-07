import time
from itertools import combinations

_P0 = time.time()
_DEADLINE = _P0 + 55.0  # Safety margin

def construct():
    if time.time() > _DEADLINE:
        return [2]

    # Primes in 2..40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Non-square-free composites in 2..40
    non_sqfree_composites = []
    for c in range(4, 41):
        if c in primes:
            continue
        is_sqfree = True
        d = 2
        n = c
        while d * d <= n:
            if n % (d * d) == 0:
                is_sqfree = False
                break
            d += 1
        if not is_sqfree:
            non_sqfree_composites.append(c)
            
    # Precompute subset products of primes
    prime_prods = {1}
    for x in primes:
        new_prods = set(prime_prods)
        for p in prime_prods:
            new_prods.add(p * x)
        prime_prods = new_prods
        
    best_result = list(primes)
    best_count = len(primes)
    
    # Try to add m composites, m from 6 down to 1
    for m in range(6, 0, -1):
        if time.time() > _DEADLINE:
            break
        for combo in combinations(non_sqfree_composites, m):
            if time.time() > _DEADLINE:
                break
            
            # Generate subset products of the combo
            comp_prods = {1}
            for x in combo:
                new_prods = set(comp_prods)
                for p in comp_prods:
                    new_prods.add(p * x)
                comp_prods = new_prods
            
            # Check for collisions
            # Total products should be 2^(12 + m)
            expected_size = 1 << (12 + m)
            
            # Efficient check:
            # If there is a collision, len(total) < expected_size
            # We can build the set and check size
            total_prods = set()
            # Optimization: if we find a duplicate, we can stop?
            # Not easily in a set comprehension.
            # But we can check size incrementally?
            
            # Let's just build it.
            # To speed up, we can use a list and then set, or just set.
            # Using a list might be faster for small sizes?
            # 4096 * 64 = 262144 elements.
            
            # We can abort early if we detect a collision?
            # Collision means p1*c1 == p2*c2.
            # This is hard to detect on the fly without storing.
            
            # Let's just build the set.
            # To make it faster, we can iterate over the smaller set first?
            # Both are fixed sizes for a given m.
            
            # We can try to break early if time is tight.
            valid = True
            # We will build the set in chunks to allow time checking
            pp_list = list(prime_prods)
            cp_list = list(comp_prods)
            
            # If m is large, this is slow.
            # But we only do this for a few combos before deadline.
            
            # Let's use a simple loop
            for p in pp_list:
                for c in cp_list:
                    total_prods.add(p * c)
                    
            if len(total_prods) == expected_size:
                return list(primes) + list(combo)
                
    return best_result
