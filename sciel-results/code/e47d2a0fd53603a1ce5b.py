# EVOLVE-BLOCK-START
import time
from math import gcd
import bisect

MAX_N = 10_000_000

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # 1) Compute sigma(n) for all n in [1, MAX_N] via divisor-sum sieve
    sigma = [0] * (MAX_N + 1)
    for d in range(1, MAX_N + 1):
        for m in range(d, MAX_N + 1, d):
            sigma[m] += d

    # 2) Group numbers by their sigma value
    groups = {}
    for n in range(1, MAX_N + 1):
        s = sigma[n]
        if s in groups:
            groups[s].append(n)
        else:
            groups[s] = [n]

    # 3) For each group with >=2 elements, find pair with ratio closest to 3/2
    #    We want min |a/b - 3/2| = min |2a - 3b| / (2b) over a,b in group, a != b
    best_err_num = None  # numerator of |a/b - 3/2| = |2a - 3b|
    best_err_den = None  # denominator = 2*b
    best_pair = None

    # Sort groups by size descending to check large groups first (more likely to have good pairs)
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)

    for s, lst in sorted_groups:
        if len(lst) < 2:
            continue
        if time.time() > DEADLINE - 2.0:
            break
        lst.sort()  # ascending
        m = len(lst)
        
        # For each j, we want a = lst[i] such that lst[i]/lst[j] ~ 3/2
        # i.e., lst[i] ~ 3*lst[j]/2
        # Use binary search to find the closest candidate
        for j in range(m):
            b = lst[j]
            target = 3 * b // 2  # approximate target for a
            
            # Binary search for target in lst
            pos = bisect.bisect_left(lst, target)
            
            # Check candidates around pos
            for ii in range(max(0, pos - 2), min(m, pos + 3)):
                if ii == j:
                    continue
                a = lst[ii]
                # error = |a/b - 3/2| = |2a - 3b| / (2b)
                num = abs(2 * a - 3 * b)
                den = 2 * b
                if num == 0:
                    continue  # exact 0 is invalid
                if best_err_num is None or (num * best_err_den < best_err_num * den):
                    best_err_num = num
                    best_err_den = den
                    best_pair = (a, b)

    if best_pair is None:
        # Fallback: return something valid (shouldn't happen)
        return [11, 6]

    a, b = best_pair
    return [int(a), int(b)]
# EVOLVE-BLOCK-END
