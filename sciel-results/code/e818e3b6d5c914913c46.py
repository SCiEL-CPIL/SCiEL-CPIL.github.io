import time
import random
from math import gcd

def construct():
    start_time = time.time()
    limit = 512
    deadline = start_time + 85
    n = limit

    # Precompute compatibility as bitmasks
    compat_mask = [0] * (n + 1)
    for i in range(1, n + 1):
        mask = 0
        for j in range(1, n + 1):
            if i * j // gcd(i, j) <= n:
                mask |= (1 << j)
        compat_mask[i] = mask

    def can_add(S_mask, x):
        return (S_mask & ~compat_mask[x]) == 0

    def complete(S_mask):
        # Single forward pass: try to add each element in order
        # This is faster than the while-changed loop
        for x in range(1, n + 1):
            if (S_mask & (1 << x)) == 0 and can_add(S_mask, x):
                S_mask |= (1 << x)
        return S_mask

    def mask_to_set(S_mask):
        return [i for i in range(1, n + 1) if (S_mask & (1 << i))]

    def local_search(S_mask):
        S_mask = complete(S_mask)
        current_size = bin(S_mask).count('1')
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            S_list = mask_to_set(S_mask)
            n_list = len(S_list)
            
            # 1-for-1 swaps: remove e, try to add x, then complete
            for e in S_list:
                if time.time() > deadline - 15:
                    break
                S_minus = S_mask & ~(1 << e)
                for x in range(1, n + 1):
                    if time.time() > deadline - 15:
                        break
                    if x != e and (S_mask & (1 << x)) == 0 and can_add(S_minus, x):
                        S2 = complete(S_minus | (1 << x))
                        s2_size = bin(S2).count('1')
                        if s2_size > current_size:
                            S_mask = S2
                            current_size = s2_size
                            improved = True
                        break

            # 1-for-2 swaps: remove e, add up to 2 new elements
            for e in S_list:
                if time.time() > deadline - 15:
                    break
                S_minus = S_mask & ~(1 << e)
                S2 = S_minus
                added = 0
                for x in range(1, n + 1):
                    if time.time() > deadline - 15:
                        break
                    if (S2 & (1 << x)) == 0 and can_add(S2, x):
                        S2 |= (1 << x)
                        added += 1
                        if added >= 2:
                            break
                s2_size = bin(S2).count('1')
                if s2_size > current_size:
                    S_mask = complete(S2)
                    current_size = bin(S_mask).count('1')
                    improved = True

            # 2-for-3 swaps
            if time.time() < deadline - 25:
                for i in range(n_list):
                    if time.time() > deadline - 25:
                        break
                    for j in range(i + 1, n_list):
                        if time.time() > deadline - 25:
                            break
                        e1 = S_list[i]
                        e2 = S_list[j]
                        S_minus = S_mask & ~(1 << e1) & ~(1 << e2)
                        S2 = S_minus
                        added = 0
                        for x in range(1, n + 1):
                            if time.time() > deadline - 25:
                                break
                            if (S2 & (1 << x)) == 0 and can_add(S2, x):
                                S2 |= (1 << x)
                                added += 1
                                if added >= 3:
                                    break
                        s2_size = bin(S2).count('1')
                        if s2_size > current_size:
                            S_mask = complete(S2)
                            current_size = bin(S_mask).count('1')
                            improved = True
                            break
                    if improved:
                        break

        return S_mask

    best_mask = 0
    best_size = 0

    # Strategy 0: Try divisor sets of all numbers <= 512, find the one with most divisors
    # where all pairs have lcm <= 512
    best_div_set = None
    best_div_count = 0
    for m in range(1, n + 1):
        divisors = [d for d in range(1, m + 1) if m % d == 0]
        # All divisors of m have lcm dividing m, so lcm <= m <= 512
        # But we need lcm(a,b) <= 512, which is guaranteed since lcm(a,b) | m
        if len(divisors) > best_div_count:
            best_div_count = len(divisors)
            best_div_set = divisors
    
    if best_div_set:
        S = 0
        for d in best_div_set:
            S |= (1 << d)
        S = local_search(S)
        sz = bin(S).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = S

    # Also try specific highly composite numbers
    for m in [360, 420, 480, 504, 240, 336, 384, 432, 468, 512, 320, 360, 400, 420, 432, 448, 468, 480, 490, 504]:
        if m > n:
            continue
        divisors = [d for d in range(1, m + 1) if m % d == 0]
        S = 0
        for d in divisors:
            S |= (1 << d)
        S = local_search(S)
        sz = bin(S).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = S

    # Strategy 1: Greedy from small numbers
    S = 0
    for x in range(1, n + 1):
        if can_add(S, x):
            S |= (1 << x)
    S = local_search(S)
    sz = bin(S).count('1')
    if sz > best_size:
        best_size = sz
        best_mask = S

    # Strategy 2: Greedy by compatibility count (most compatible first)
    comp_counts = [bin(compat_mask[i]).count('1') for i in range(n + 1)]
    order = sorted(range(1, n + 1), key=lambda x: -comp_counts[x])
    S = 0
    for x in order:
        if time.time() > deadline - 30:
            break
        if can_add(S, x):
            S |= (1 << x)
    S = local_search(S)
    sz = bin(S).count('1')
    if sz > best_size:
        best_size = sz
        best_mask = S

    # Strategy 3: Random Restarts with better seeds
    rng = random.Random(42)
    for _ in range(300):
        if time.time() > deadline - 15:
            break
        # Mix: start from a random subset of the best known set + random additions
        if best_size > 0 and rng.random() < 0.5:
            S_list = mask_to_set(best_mask)
            # Keep 70-90% of the best set
            keep_frac = rng.uniform(0.7, 0.9)
            S = 0
            for e in S_list:
                if rng.random() < keep_frac:
                    S |= (1 << e)
        else:
            S = 0
            start_size = rng.randint(5, 15)
            for _ in range(start_size):
                x = rng.randint(1, n)
                S |= (1 << x)
        S = local_search(S)
        sz = bin(S).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = S

    # Strategy 4: Start from 1..k for various k, then complete and search
    for k in [10, 15, 20, 25, 30, 40, 50, 64, 100]:
        if time.time() > deadline - 15:
            break
        S = 0
        for x in range(1, k + 1):
            S |= (1 << x)
        S = local_search(S)
        sz = bin(S).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = S

    # Final polish: try to expand the best solution
    if best_size >= 20 and time.time() < deadline - 5:
        S = best_mask
        S = local_search(S)
        sz = bin(S).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = S

    return mask_to_set(best_mask)
