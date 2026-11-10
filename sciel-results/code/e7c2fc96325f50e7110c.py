"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    def is_valid_addition(s, candidate):
        """Check if adding candidate to s keeps the set valid."""
        s_with = s | {candidate}
        for a in s:
            if a == candidate:
                continue
            for b in s:
                if b <= a:
                    continue
                if (a + b) in s_with and (a + candidate) in s_with and (b + candidate) in s_with:
                    return False
        return True

    base_A = set(range(65, 129))
    base_B = set(range(17, 33))
    candidates = list(range(33, 65))

    best_set = base_A | base_B
    best_size = len(best_set)

    # Try adding candidates in various orders
    strategies = [
        list(range(33, 65)),
        list(range(64, 32, -1)),
        list(range(33, 49)) + list(range(64, 48, -1)),
        list(range(64, 48, -1)) + list(range(33, 49)),
    ]

    rng = random.Random(123)
    for _ in range(40):
        c = candidates[:]
        rng.shuffle(c)
        strategies.append(c)

    for strat in strategies:
        if time.time() > deadline:
            break
        S = base_A | base_B
        for x in strat:
            if time.time() > deadline:
                break
            if is_valid_addition(S, x):
                S.add(x)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    # Try local improvement: try swapping one element out for two in
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            S = set(best_set)
            for x in sorted(S):
                if time.time() > deadline - 5:
                    break
                S_remove = S - {x}
                # Try to add two elements from gap
                for y in candidates:
                    if y in S:
                        continue
                    if not is_valid_addition(S_remove, y):
                        continue
                    S_temp = S_remove | {y}
                    for z in candidates:
                        if z == y or z in S_temp:
                            continue
                        if is_valid_addition(S_temp, z):
                            S_new = S_temp | {z}
                            if len(S_new) > len(best_set):
                                best_set = S_new
                                best_size = len(S_new)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break

    result = [int(x) for x in sorted(best_set)]
    return result
# EVOLVE-BLOCK-END
