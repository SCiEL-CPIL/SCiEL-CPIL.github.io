"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    from math import isqrt
    from itertools import combinations

    # Factor each number 2..40 into prime-exponent tuple.
    def factor(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    vecs = {}
    for n in range(2, 41):
        f = factor(n)
        vecs[n] = tuple(f.get(p, 0) for p in primes)

    def has_distinct_subset_products(selected):
        k = len(selected)
        if k > 20:
            return False
        products = set()
        for mask in range(1 << k):
            prod = 1
            for i in range(k):
                if mask >> i & 1:
                    prod *= selected[i]
            if prod in products:
                return False
            products.add(prod)
        return True

    # Strategy:
    # 1. Identify prime powers.
    # 2. Greedily add prime powers that maintain distinctness.
    # 3. Greedily add other numbers that maintain distinctness.
    
    # Generate list of numbers 2..40
    candidates = list(range(2, 41))
    
    # Heuristic ordering:
    # Prefer numbers with fewer prime factors (prime powers)
    # Then prefer smaller numbers?
    
    def heuristic_score(n):
        # Count prime factors with multiplicity
        cnt = 0
        temp = n
        for p in primes:
            while temp % p == 0:
                cnt += 1
                temp //= p
        # Return negative count so that fewer factors come first
        return cnt

    candidates_sorted = sorted(candidates, key=heuristic_score)
    
    selected = []
    for n in candidates_sorted:
        trial = selected + [n]
        if has_distinct_subset_products(trial):
            selected = trial
            
    # Try to improve with local search / random restarts if needed?
    # Given the small size, one good greedy pass might suffice.
    # But to be safe, let's try a few different orderings.
    
    best_set = selected
    best_size = len(selected)
    
    # Try a few random permutations of the remaining candidates or the whole set?
    # Actually, let's just try sorting by value, and by prime factor count descending.
    
    for order_key in [lambda x: x, lambda x: -heuristic_score(x), lambda x: x * heuristic_score(x)]:
        c_sorted = sorted(candidates, key=order_key)
        sel = []
        for n in c_sorted:
            trial = sel + [n]
            if has_distinct_subset_products(trial):
                sel = trial
        if len(sel) > best_size:
            best_size = len(sel)
            best_set = sel

    return list(best_set)

# EVOLVE-BLOCK-END
