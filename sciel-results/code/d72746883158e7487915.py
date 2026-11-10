"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    n = 14
    m = 35
    maximum = 10000
    
    # Strategy: Use powers of 2 with each exponent appearing twice.
    # This creates sum collisions for same-exponent pairs and
    # product uniqueness based on exponent sums.
    # Exponents 0-6, each used twice: 2^0, 2^0, 2^1, 2^1, ..., 2^6, 2^6
    # Values: 1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 32, 32, 64, 64
    # All <= 10000, all positive, all distinct? No - they're not distinct!
    # We need 14 DISTINCT positive integers.
    
    # Revised: Use distinct powers of 2: 2^0 through 2^13
    # But 2^13 = 8192, 2^14 = 16384 > 10000
    # So max exponent is 13. We have 14 distinct powers: 1,2,4,8,...,8192
    # Sums: 2^a + 2^b = 2^min(a,b)*(1+2^|a-b|) - all distinct for a!=b
    # Products: 2^(a+b) - determined by a+b, so collisions when a+b equal
    
    # Actually, let's think differently. We want to MINIMIZE distinct sums AND products.
    # With distinct powers of 2:
    #   Sums: each pair (a,b) with a<b gives 2^a + 2^b. These are all distinct 
    #         (binary representation has exactly two 1-bits at positions a and b)
    #         So 35 distinct sums.
    #   Products: 2^(a+b). Collisions when a+b is the same.
    #         Pairs with sum s: count depends on s.
    #         Exponents 0..13, pair sums range from 1 to 25.
    #         Number of distinct products = number of distinct pair sums = 25
    #         (sums 1,2,...,25 all achievable)
    #   Raw value = max(35, 25) = 35
    
    # Alternative: Use values that are NOT all powers of 2 to get sum collisions.
    # Try: values where many pairs sum to the same value.
    # E.g., symmetric pairs around a center: (1,100),(2,99),(3,98),...
    # But we need 14 distinct values <= 10000.
    
    # Let's try: pairs that sum to the same value.
    # If we have k pairs summing to S, that's k distinct sums (all = S) plus
    # cross-pair sums.
    
    # Better approach: Use arithmetic progression to get sum collisions.
    # Values: a, a+d, a+2d, ..., a+13d
    # Sums: (a+id) + (a+jd) = 2a + (i+j)d
    # Number of distinct sums = number of distinct i+j for i<j in 0..13
    # i+j ranges from 1 to 25, all achievable -> 25 distinct sums
    # Products: (a+id)(a+jd) = a^2 + a(i+j)d + ij*d^2
    # These are generally distinct unless carefully chosen.
    
    # Let's try a concrete approach: search over small structured sets.
    
    def raw_value(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))
    
    def all_edges():
        return [list(e) for e in combinations(range(n), 2)]
    
    edges = all_edges()
    
    # Candidate 1: Powers of 2
    vals1 = [2**i for i in range(14)]
    score1 = raw_value(vals1, edges)
    
    # Candidate 2: Arithmetic progression
    vals2 = [1 + i*100 for i in range(14)]  # 1, 101, 201, ..., 1301
    score2 = raw_value(vals2, edges)
    
    # Candidate 3: Try to minimize by using values with many additive relations
    # Use values: 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610 (Fibonacci-like)
    vals3 = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
    score3 = raw_value(vals3, edges)
    
    # Candidate 4: Symmetric around 5000: 100,200,...,700, 800,...,1400, 4900,4950,5050,5100
    # Let's try: small values that create many sum collisions
    vals4 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    score4 = raw_value(vals4, edges)
    
    # Candidate 5: Use values where products collide more
    # If values are all divisors of a highly composite number, products may collide
    vals5 = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]
    score5 = raw_value(vals5, edges)
    
    # Candidate 6: Geometric progression with ratio 3
    vals6 = [3**i for i in range(14)]
    # 3^13 = 1594323 > 10000, so this won't work
    # Use ratio 2: already tried (powers of 2)
    
    # Candidate 7: Mix - some small, some large to control sums
    vals7 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    score7 = raw_value(vals7, edges)
    
    candidates = [
        (vals1, score1),
        (vals2, score2),
        (vals3, score3),
        (vals4, score4),
        (vals5, score5),
        (vals7, score7),
    ]
    
    # Pick the best candidate
    best_vals, best_score = min(candidates, key=lambda x: x[1])
    
    # Verify validity
    assert len(best_vals) == n
    assert len(set(best_vals)) == n  # distinct
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in best_vals)
    assert all(1 <= v <= maximum for v in best_vals)
    assert len(edges) == m
    assert len(set(tuple(sorted(e)) for e in edges)) == m  # no duplicates
    
    return {'values': list(best_vals), 'edges': edges}
# EVOLVE-BLOCK-END

