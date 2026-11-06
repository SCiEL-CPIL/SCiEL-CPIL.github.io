import time
import random
from math import gcd

def construct():
    start_time = time.time()
    limit = 512
    deadline = start_time + 75
    n = limit

    # Precompute compatibility sets
    comp = [set() for _ in range(n + 1)]
    for i in range(1, n + 1):
        ci = comp[i]
        for j in range(i, n + 1):
            if i * j // gcd(i, j) <= n:
                ci.add(j)
                comp[j].add(i)

    def is_valid_add(S, x):
        cx = comp[x]
        for y in S:
            if y not in cx:
                return False
        return True

    def complete(S):
        # greedily add all compatible elements
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
                        # swap e for x, then complete
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
        return S

    best = set(range(1, 23))

    # Strategy 1: increasing greedy
    S = complete(set(range(1, 23)))
    S = local_search(S)
    if len(S) > len(best):
        best = S

    # Strategy 2: decreasing greedy
    S = set(range(1, 23))
    for x in range(n, 22, -1):
        if time.time() > deadline - 20:
            break
        if is_valid_add(S, x):
            S.add(x)
    S = local_search(S)
    if len(S) > len(best):
        best = S

    # Strategy 3: random restarts with kicks
    rng = random.Random(12345)
    for it in range(500):
        if time.time() > deadline - 15:
            break
        # random seed
        seed = {1}
        order = list(range(2, n + 1))
        rng.shuffle(order)
        for x in order:
            if time.time() > deadline - 15:
                break
            if is_valid_add(seed, x):
                seed.add(x)
        S = local_search(seed)
        if len(S) > len(best):
            best = S
            # kick: remove a random element and re-optimize
            if time.time() < deadline - 15 and len(best) > 1:
                e = rng.choice(sorted(best))
                S2 = local_search(best - {e})
                if len(S2) > len(best):
                    best = S2

    return sorted(best)
