import time
from math import gcd

def construct():
    start_time = time.time()
    limit = 512
    deadline = start_time + 80  # safety margin from 90s kill
    n = limit

    # Precompute LCM compatibility matrix
    # compat[x][y] = True if lcm(x,y) <= limit
    compat = [[False] * (n + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a, n + 1):
            l = (a * b) // gcd(a, b)
            c = l <= limit
            compat[a][b] = c
            compat[b][a] = c

    def is_valid_add(S, x):
        for y in S:
            if not compat[x][y]:
                return False
        return True

    def complete(S):
        S = set(S)
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if time.time() > deadline - 5:
                    return S
                if x not in S and is_valid_add(S, x):
                    S.add(x)
                    changed = True
        return S

    def local_search(S):
        S = set(S)
        S = complete(S)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # 1-for-1 swaps
            for e in list(S):
                if time.time() > deadline - 10:
                    break
                S_minus = S - {e}
                for x in range(1, n + 1):
                    if time.time() > deadline - 10:
                        break
                    if x != e and x not in S and is_valid_add(S_minus, x):
                        S2 = complete(S_minus | {x})
                        if len(S2) > len(S):
                            S = S2
                            improved = True
                        break
            # 1-for-2 swaps
            for e in list(S):
                if time.time() > deadline - 10:
                    break
                S_minus = S - {e}
                added = 0
                S2 = set(S_minus)
                for x in range(1, n + 1):
                    if time.time() > deadline - 10:
                        break
                    if x not in S2 and is_valid_add(S2, x):
                        S2.add(x)
                        added += 1
                        if added >= 2:
                            break
                if len(S2) > len(S):
                    S = complete(S2)
                    improved = True
            # 2-for-2 swaps
            for e1 in list(S):
                if time.time() > deadline - 10:
                    break
                for e2 in list(S):
                    if time.time() > deadline - 10:
                        break
                    if e1 >= e2:
                        continue
                    S_minus = S - {e1, e2}
                    S2 = set(S_minus)
                    added = 0
                    for x in range(1, n + 1):
                        if time.time() > deadline - 10:
                            break
                        if x not in S2 and is_valid_add(S2, x):
                            S2.add(x)
                            added += 1
                            if added >= 2:
                                break
                    if len(S2) > len(S):
                        S = complete(S2)
                        improved = True
        return S

    # Seed strategies
    best = set()
    
    # Strategy 1: Divisors of highly composite numbers
    seeds = []
    for m in [240, 360, 420, 480, 504, 336, 288, 210, 180, 168, 144, 120, 96, 72, 60, 48, 36, 24, 12, 6, 4, 2]:
        divs = set()
        for i in range(1, int(m**0.5) + 1):
            if m % i == 0:
                divs.add(i)
                divs.add(m // i)
        seeds.append(divs)
        
    # Also add ranges
    seeds.append(set(range(1, 17)))
    seeds.append(set(range(1, 13)))
    
    for seed in seeds:
        if time.time() > deadline - 20:
            break
        S = local_search(seed)
        if len(S) > len(best):
            best = S

    # Random restarts
    import random
    rng = random.Random(42)
    for _ in range(100):
        if time.time() > deadline - 10:
            break
        # Random seed based on small numbers
        seed = {1}
        # Add random small numbers
        for x in range(2, 30):
            if rng.random() < 0.5:
                if is_valid_add(seed, x):
                    seed.add(x)
        S = local_search(seed)
        if len(S) > len(best):
            best = S

    return sorted(best)
