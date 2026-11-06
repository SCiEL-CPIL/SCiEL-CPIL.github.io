import time
from fractions import Fraction
import bisect
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 90s limit

    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)
    unit_vals = [Fraction(1, d) for d in denominators]

    # ---------- Meet-in-the-middle exact search ----------
    # LCM of 2..48
    L = 1
    for d in denominators:
        L = lcm(L, d)
    weights = [L // d for d in denominators]

    mid = n // 2  # 23
    w1 = weights[:mid]
    w2 = weights[mid:]

    n1 = len(w1)
    n2 = len(w2)

    def gen_sums(wlist):
        sums = [(0, 0)]
        for i, w in enumerate(wlist):
            bit = 1 << i
            # Generate new sums by adding w to existing sums
            # Only keep sums < L (strictly below 1 when divided by L)
            new = []
            for s, m in sums:
                ns = s + w
                if ns < L:
                    new.append((ns, m | bit))
            sums.extend(new)
            if time.time() > deadline - 30.0:
                return None  # signal timeout
        return sums

    # Generate subset sums for both halves
    sums1 = gen_sums(w1)
    if sums1 is None:
        # Fallback: return a simple valid solution
        return [0]

    sums2 = gen_sums(w2)
    if sums2 is None:
        # Fallback: return a simple valid solution
        return [0]

    # Sort sums2 by value for binary search
    sums2.sort(key=lambda x: x[0])
    vals2 = [s for s, _ in sums2]

    best_sum = -1
    best_mask = 0

    # For each sum in half1, find the largest sum in half2 such that total < L
    for s1, m1 in sums1:
        if time.time() > deadline - 20.0:
            break
        # We need s1 + s2 < L, so s2 < L - s1
        # Find the largest s2 < L - s1
        target = L - s1 - 1  # s2 must be <= target
        idx = bisect.bisect_right(vals2, target)
        if idx > 0:
            s2, m2 = sums2[idx - 1]
            total = s1 + s2
            if total > best_sum:
                best_sum = total
                best_mask = m1 | (m2 << mid)

    if best_sum < 0:
        # Fallback: return a simple valid solution
        return [0]

    # Construct indices from best_mask
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(i)

    if not result:
        return [0]

    # Verify the result is strictly below 1
    total_frac = sum(unit_vals[i] for i in result)
    if total_frac >= 1:
        # This shouldn't happen, but fallback
        return [0]

    return result
