"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for i in range(1, n + 1):
        if n % i == 0:
            divisors.append(i)
    divisors.sort(reverse=True)
    divisor_set = set(divisors)
    ndiv = len(divisors)
    
    # Try size 2
    for i in range(ndiv):
        for j in range(i + 1, ndiv):
            if divisors[i] + divisors[j] == target:
                return [divisors[i], divisors[j]]
    
    # Try size 3
    for i in range(ndiv):
        for j in range(i + 1, ndiv):
            rem = target - divisors[i] - divisors[j]
            if rem > 0 and rem in divisor_set and rem != divisors[i] and rem != divisors[j]:
                return [divisors[i], divisors[j], rem]
    
    # Try size 4: use pair-sum lookup
    # Precompute all pair sums
    pair_sums = {}
    for i in range(ndiv):
        for j in range(i + 1, ndiv):
            s = divisors[i] + divisors[j]
            if s not in pair_sums:
                pair_sums[s] = []
            pair_sums[s].append((i, j))
    
    for i in range(ndiv):
        for j in range(i + 1, ndiv):
            if time.time() > deadline:
                break
            rem = target - divisors[i] - divisors[j]
            if rem <= 0:
                continue
            if rem in pair_sums:
                for (k, l) in pair_sums[rem]:
                    if k != i and k != j and l != i and l != j:
                        return [divisors[i], divisors[j], divisors[k], divisors[l]]
    
    # Try size 5: use pair-sum lookup for the last 3, or triple-sum
    # More efficient: iterate over pairs (i,j), then need 3 more summing to rem
    # Precompute triple sums? Too many. Instead, for each pair, use pair-sum for remaining 3 as (1+2)
    # For each pair (i,j), rem = target - di - dj. Need 3 distinct divisors (not i,j) summing to rem.
    # Check if any divisor k (not i,j) exists such that rem - dk is a pair sum of two divisors (not i,j,k)
    for i in range(ndiv):
        for j in range(i + 1, ndiv):
            if time.time() > deadline:
                break
            rem = target - divisors[i] - divisors[j]
            if rem <= 0:
                continue
            # Try each k for the third element
            for k in range(ndiv):
                if k == i or k == j:
                    continue
                rem2 = rem - divisors[k]
                if rem2 <= 0:
                    continue
                if rem2 in pair_sums:
                    for (l, m) in pair_sums[rem2]:
                        if l != i and l != j and l != k and m != i and m != j and m != k:
                            return [divisors[i], divisors[j], divisors[k], divisors[l], divisors[m]]
    
    # Fallback: greedy
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    assert remainder == 0
    return chosen
# EVOLVE-BLOCK-END
