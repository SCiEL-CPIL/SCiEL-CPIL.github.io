"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

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
    
    # Try size 3: for each pair, check if remainder is a valid third divisor
    for i in range(len(divisors)):
        for j in range(i + 1, len(divisors)):
            if time.time() > deadline:
                break
            rem = target - divisors[i] - divisors[j]
            if rem > 0 and rem in divisor_set and rem != divisors[i] and rem != divisors[j]:
                return [divisors[i], divisors[j], rem]
    
    # Try size 4: for each pair, check if target - pair_sum can be made from 2 other divisors
    # More efficient: iterate over pairs, then for each, check if remaining sum is achievable with 2 divisors
    # For each pair (a,b), need c+d = target-a-b where c,d are distinct divisors not in {a,b}
    for i in range(len(divisors)):
        for j in range(i + 1, len(divisors)):
            if time.time() > deadline:
                break
            rem = target - divisors[i] - divisors[j]
            if rem <= 0:
                continue
            # Find two distinct divisors c, d (not a, not b) with c + d = rem
            for k in range(len(divisors)):
                c = divisors[k]
                if c == divisors[i] or c == divisors[j]:
                    continue
                d = rem - c
                if d <= 0 or d == c:
                    continue
                if d in divisor_set and d != divisors[i] and d != divisors[j] and d != c:
                    return [divisors[i], divisors[j], c, d]
    
    # Fallback: greedy (known to produce 5)
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
