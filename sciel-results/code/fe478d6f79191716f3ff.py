import time
import math
from itertools import combinations

def construct():
    n = 5040
    target = 4979
    deadline = time.time() + 50.0

    # Generate all divisors of 5040
    divisors = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort()
    divisor_set = set(divisors)
    n_divs = len(divisors)
    
    # k=1
    if target in divisor_set:
        return [target]
    if time.time() > deadline: return []

    # k=2
    for d in divisors:
        comp = target - d
        if comp > d and comp in divisor_set:
            return [d, comp]
    if time.time() > deadline: return []

    # k=3
    for i in range(n_divs):
        a = divisors[i]
        if a >= target:
            break
        for j in range(i + 1, n_divs):
            b = divisors[j]
            if a + b >= target:
                break
            c = target - a - b
            if c > b and c in divisor_set:
                return [a, b, c]
    if time.time() > deadline: return []

    # k=4
    pair_map = {}
    for i in range(n_divs):
        a = divisors[i]
        if a >= target:
            continue
        for j in range(i + 1, n_divs):
            b = divisors[j]
            s = a + b
            if s >= target:
                break
            if s not in pair_map:
                pair_map[s] = []
            pair_map[s].append((i, j))
            
    for s1, pairs1 in pair_map.items():
        s2 = target - s1
        if s2 not in pair_map:
            continue
        for (i1, j1) in pairs1:
            for (i2, j2) in pair_map[s2]:
                if len({i1, j1, i2, j2}) == 4:
                    return [divisors[i1], divisors[j1], divisors[i2], divisors[j2]]
    if time.time() > deadline: return []

    # k >= 5: Meet in the Middle
    mid = n_divs // 2
    left = divisors[:mid]
    right = divisors[mid:]
    
    for k in range(5, n_divs + 1):
        if time.time() > deadline:
            return []
            
        # Try all splits of k into i from left and k-i from right
        min_i = max(0, k - len(right))
        max_i = min(k, len(left))
        
        for i in range(min_i, max_i + 1):
            k_right = k - i
            if k_right < 0 or k_right > len(right):
                continue
                
            # Generate subsets of size i from left
            left_sums = {}
            for combo in combinations(range(len(left)), i):
                s = sum(left[c] for c in combo)
                if s > target:
                    continue
                if s not in left_sums:
                    left_sums[s] = []
                left_sums[s].append(combo)
                
            # Generate subsets of size k_right from right
            right_sums = {}
            for combo in combinations(range(len(right)), k_right):
                s = sum(right[c] for c in combo)
                if s > target:
                    continue
                if s not in right_sums:
                    right_sums[s] = []
                right_sums[s].append(combo)
                
            # Check for matching sums
            for s_l, combos_l in left_sums.items():
                s_r = target - s_l
                if s_r in right_sums:
                    c_l = combos_l[0]
                    c_r = right_sums[s_r][0]
                    res = [left[idx] for idx in c_l] + [right[idx] for idx in c_r]
                    return res
                    
    return []
