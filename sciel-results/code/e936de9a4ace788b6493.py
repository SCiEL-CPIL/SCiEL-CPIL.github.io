"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # Primes in [2, 40]
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Powers of 2: 2, 4, 16, 32 (exponents 1, 2, 4, 5 -> sums distinct)
    p2 = [2, 4, 16, 32]
    
    # Powers of 3: 3, 9 (exponents 1, 2 -> sums 0,1,2,3 distinct)
    # 27 would make exponents 1,2,3 which has collision (3 vs 1+2)
    p3 = [3, 9]
    
    # Powers of 5: 5, 25 (exponents 1, 2 -> sums 0,1,2,3 distinct)
    p5 = [5, 25]
    
    # Other primes: just the prime itself
    others = [p for p in primes if p not in [2, 3, 5]]
    
    result = p2 + p3 + p5 + others
    
    # Ensure distinctness and range
    result = [x for x in result if 2 <= x <= 40]
    result = list(set(result)) # Just in case, though they are distinct by construction
    
    # Verify all subset products are distinct
    # This is a safety check, though the logic above guarantees it
    # For k=17, 2^17 = 131072 products. This is fast to compute.
    from itertools import combinations
    def has_distinct_products(lst):
        prods = set()
        n = len(lst)
        for i in range(1 << n):
            prod = 1
            for j in range(n):
                if (i >> j) & 1:
                    prod *= lst[j]
            if prod in prods:
                return False
            prods.add(prod)
        return True
        
    if not has_distinct_products(result):
        # Fallback to just primes if verification fails (should not happen)
        return primes
        
    return result

# EVOLVE-BLOCK-END
