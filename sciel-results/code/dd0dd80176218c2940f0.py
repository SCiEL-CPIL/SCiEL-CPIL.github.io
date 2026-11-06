import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 90s limit, 5s margin
    
    N = 512
    limit = N

    # Precompute compatibility sets
    comp = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        ci = comp[i]
        for j in range(i, N + 1):
            g = gcd(i, j)
            lcm_val = (i // g) * j
            if lcm_val <= limit:
                ci.add(j)
                if i != j:
                    comp[j].add(i)

    def is_valid_add(S, x):
        cx = comp[x]
        for y in S:
            if y not in cx:
                return False
        return True

    def complete(S):
        S = set(S)
        # Optimization: Iterate in a fixed order but check time.
        # To make it faster, we can try to add elements in an order that is likely to succeed?
        # Or just standard.
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if time.time() > deadline - 2:
                    return S
                if x not in S and is_valid_add(S, x):
                    S.add(x)
                    changed = True
        return S

    def local_search(S):
        S = set(S)
        S = complete(S)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # 1-for-1 swaps
            for e in list(S):
                if time.time() > deadline - 5:
                    break
                S_minus = S - {e}
                for x in range(1, N + 1):
                    if time.time() > deadline - 5:
                        break
                    if x != e and x not in S and is_valid_add(S_minus, x):
                        S2 = complete(S_minus | {x})
                        if len(S2) > len(S):
                            S = S2
                            improved = True
                        break
            # 1-for-2 swaps
            for e in list(S):
                if time.time() > deadline - 5:
                    break
                S_minus = S - {e}
                added = 0
                S2 = set(S_minus)
                for x in range(1, N + 1):
                    if time.time() > deadline - 5:
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
            # Pick two elements to remove, try to add two new ones
            # This is expensive: O(|S|^2 * N)
            # We only do this if time permits and previous moves stalled or to find better peaks
            if time.time() < deadline - 10:
                elements = list(S)
                for i in range(len(elements)):
                    if time.time() > deadline - 10:
                        break
                    e1 = elements[i]
                    for j in range(i + 1, len(elements)):
                        if time.time() > deadline - 10:
                            break
                        e2 = elements[j]
                        S_minus = S - {e1, e2}
                        # Try to add two elements
                        # Greedily pick the first two valid candidates?
                        # Or try all pairs of candidates? Too slow.
                        # Just greedily add valid ones until 2 added or no more.
                        S2 = set(S_minus)
                        added = 0
                        # To make it deterministic and fast, iterate x
                        for x in range(1, N + 1):
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
                            
            # 2-for-1 kick (Remove 2, add 0, then complete)
            # This can escape local optima if removing 2 bad elements allows adding 3 good ones
            if time.time() < deadline - 15 and not improved:
                 elements = list(S)
                 # Try a few random pairs to remove
                 for _ in range(10):
                     if time.time() > deadline - 15:
                         break
                     if len(S) < 2: break
                     e1, e2 = random.sample(elements, 2)
                     S_minus = S - {e1, e2}
                     S2 = complete(S_minus)
                     if len(S2) > len(S):
                         S = S2
                         improved = True

        return S

    # Initial baseline
    best = set(range(1, 24))
    best = complete(best)
    best = local_search(best)
    best_size = len(best)

    # Strategy 1: increasing greedy
    S = set(range(1, 23))
    S = complete(S)
    S = local_search(S)
    if len(S) > best_size:
        best = S
        best_size = len(S)

    # Strategy 2: decreasing greedy
    S = set(range(1, 23))
    for x in range(N, 22, -1):
        if time.time() > deadline - 20:
            break
        if is_valid_add(S, x):
            S.add(x)
    S = local_search(S)
    if len(S) > best_size:
        best = S
        best_size = len(S)

    # Strategy 3: random restarts with kicks
    rng = random.Random(12345)
    for it in range(200):
        if time.time() > deadline - 10:
            break
        # random seed
        seed = {1}
        order = list(range(2, N + 1))
        rng.shuffle(order)
        for x in order:
            if time.time() > deadline - 10:
                break
            if is_valid_add(seed, x):
                seed.add(x)
        S = local_search(seed)
        if len(S) > best_size:
            best = S
            best_size = len(S)
            # kick: remove a random element and re-optimize
            if time.time() < deadline - 10 and len(best) > 1:
                e = rng.choice(sorted(best))
                S2 = local_search(best - {e})
                if len(S2) > best_size:
                    best = S2
                    best_size = len(S2)

    return sorted(best)
