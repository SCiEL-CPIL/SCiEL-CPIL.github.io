"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

N = 128

def construct():
    start_time = time.time()
    deadline = start_time + 55

    def can_add(x, S_set):
        s_list = list(S_set)
        n = len(s_list)
        for i in range(n):
            a = s_list[i]
            for j in range(i + 1, n):
                b = s_list[j]
                if a + b in S_set and a + x in S_set and b + x in S_set:
                    return False
        return True

    base = set(range(65, 129))
    best_S = set(base)
    best_size = len(base)

    rng = random.Random(42)

    for attempt in range(300):
        if time.time() > deadline:
            break

        S = set(base)
        cands = list(range(1, 65))
        rng.shuffle(cands)

        for x in cands:
            if time.time() > deadline:
                break
            if can_add(x, S):
                S.add(x)

        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)

    result = sorted(best_S)
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
