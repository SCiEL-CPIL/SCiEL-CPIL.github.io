import sys
from math import isqrt, prod
from itertools import combinations, product

sys.setrecursionlimit(10000)

# Powerful numbers: n = x^2 * y^3 where y is squarefree
# We generate powerful numbers up to some limit to search for good N

def generate_powerful(limit):
    """Generate all powerful numbers up to limit."""
    pows = set()
    # n = a^2 * b^3, where b is squarefree
    max_b = int(round(limit ** (1/3))) + 2
    for b in range(1, max_b + 1):
        # b must be squarefree
        if is_squarefree(b):
            b3 = b ** 3
            if b3 > limit:
                break
            max_a = isqrt(limit // b3)
            for a in range(1, max_a + 1):
                val = a * a * b3
                if val <= limit:
                    pows.add(val)
    return sorted(pows)

def is_squarefree(n):
    """Check if n is squarefree."""
    if n == 1:
        return True
    for p in range(2, isqrt(n) + 1):
        if n % (p * p) == 0:
            return False
    return True

def factorize(n):
    """Exact factorization of n, returns dict prime -> exponent."""
    if n < 2:
        return {}
    factors = {}
    d = 2
    while d * d <= n:
        if n % d == 0:
            cnt = 0
            while n % d == 0:
                cnt += 1
                n //= d
            factors[d] = cnt
        d += 1 if d == 2 else 2
    if n > 1:
        factors[n] = 1
    return factors

def is_powerful(n):
    """Check if n is a powerful number (every prime factor appears with exponent >= 2)."""
    if n < 1:
        return False
    if n == 1:
        return True
    f = factorize(n)
    for exp in f.values():
        if exp < 2:
            return False
    return True

def construct():
    # Strategy:
    # 1. Generate powerful numbers up to a reasonable limit
    # 2. For candidate N values, count representations as sum of two powerful numbers
    # 3. Focus on N that are highly composite or have special structure
    
    # Powerful numbers up to 10^6 are manageable
    POW_LIMIT = 10**6
    powerful = generate_powerful(POW_LIMIT)
    pow_set = set(powerful)
    
    # For larger powerful numbers, we can't generate all up to 10^18
    # But we can check if a specific number is powerful using factorization
    
    # Strategy: try N values that are sums of small powerful numbers
    # and verify all pairs
    
    # Better approach: pick N from a set of candidates and count pairs
    # Candidates: multiples of highly composite numbers, or specific structures
    
    # Let's try: N = k * H where H is highly composite
    # Actually, let's just try to find N with many pairs by sampling
    
    # Generate candidate N values
    # Try N in ranges where many powerful numbers sum to N
    
    # For each powerful a <= N/2, check if N-a is powerful
    # This is expensive for large N, so limit to N where we can afford it
    
    # Alternative: precompute powerful numbers up to 10^6
    # For N up to 2*10^6, we can check all pairs
    # For larger N, we need to be smarter
    
    # Let's try N values that are multiples of small powerful-friendly numbers
    
    # Key insight: if N is a multiple of many small powerful numbers' gcds
    # Actually, let's just search over candidate N values
    
    # Candidate N: try values around 10^6 to 10^8 where we can still check
    # For N up to ~2*POW_LIMIT, all pairs (a, b) have a, b <= POW_LIMIT
    # so we can use the precomputed set
    
    best_N = 0
    best_pairs = []
    best_count = 0
    
    # Search N from 2 to 2*POW_LIMIT
    # This is 2*10^6 values, and for each we iterate over powerful numbers up to N/2
    # That's too slow: 2*10^6 * 10^4 = 2*10^10
    
    # Better: for each powerful a, for each powerful b, compute N = a + b
    # and count how many times each N appears
    # But that's O(len(powerful)^2) which is ~10^8, also too slow
    
    # Compromise: only consider powerful numbers up to a smaller limit
    # and use a dictionary to count sums
    
    SMALL_POW_LIMIT = 10**5
    small_powerful = generate_powerful(SMALL_POW_LIMIT)
    
    from collections import defaultdict
    sum_counts = defaultdict(list)
    
    n_pow = len(small_powerful)
    for i in range(n_pow):
        a = small_powerful[i]
        for j in range(i, n_pow):
            b = small_powerful[j]
            N = a + b
            if N > 2 * SMALL_POW_LIMIT:
                break
            sum_counts[N].append((a, b))
    
    # Find N with most pairs
    best_N = 0
    best_count = 0
    for N, pairs in sum_counts.items():
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_N = N
    
    # Now, also check for larger N by sampling
    # For N > 2*SMALL_POW_LIMIT, some pairs may involve larger powerful numbers
    # We can augment by checking additional powerful numbers
    
    # Let's also try specific candidate N values known to have many representations
    # E.g., N that are multiples of 4, 8, etc.
    
    # Also try: N = 2^k * 3^m * 5^p ... highly composite
    # But the key is sum of two powerful numbers
    
    # Let's extend the search: try N values and check pairs where
    # one component is small (from our list) and the other is checked via factorization
    
    extended_best_N = best_N
    extended_best_pairs = sum_counts.get(best_N, [])
    extended_best_count = best_count
    
    # Try some larger N candidates
    # For N in a few ranges, check if N - a is powerful for a in small_powerful
    for N_candidate in [10**6, 2*10**6, 5*10**6, 10**7, 2*10**7, 5*10**7, 10**8]:
        if N_candidate > 10**18:
            continue
        pairs = []
        for a in small_powerful:
            if a > N_candidate // 2:
                break
            b = N_candidate - a
            if b < 1:
                continue
            if b in pow_set:
                if a <= b:
                    pairs.append((a, b))
            else:
                # Check if b is powerful via factorization
                if is_powerful(b):
                    if a <= b:
                        pairs.append((a, b))
        
        if len(pairs) > extended_best_count:
            extended_best_count = len(pairs)
            extended_best_N = N_candidate
            extended_best_pairs = pairs
    
    N = extended_best_N
    pairs = extended_best_pairs
    
    # Convert to list of [a, b]
    result_pairs = [[a, b] for a, b in pairs]
    
    return {'N': N, 'pairs': result_pairs}
