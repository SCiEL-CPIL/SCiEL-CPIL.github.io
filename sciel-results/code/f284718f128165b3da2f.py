"""Combined construction: structural seed + greedy addition + local search refinement."""
import time
import random

def construct():
    start = time.time()
    DEADLINE = 55.0  # seconds, safety margin before 60s kill
    TARGET = 200
    MASK = (1 << (TARGET + 1)) - 1

    # Bitset-based reachable computation: bit i set means sum i is reachable
    def reachable_bitset(S):
        r = 1  # bit 0
        for x in sorted(S):
            r |= (r << x) & MASK
        return r

    def is_valid_bitset(r):
        return not (r & (1 << TARGET))

    def can_add(r, x):
        """Check if adding x to a set with reachable bitset r keeps 200 unreachable."""
        new_r = r | ((r << x) & MASK)
        return not (new_r & (1 << TARGET))

    # Step 1: Start with structural seed {67, ..., 100}
    S = set(range(67, 101))  # 34 elements, provably valid
    r = reachable_bitset(S)

    # Step 2: Greedy addition from 66 down to 1
    for x in range(66, 0, -1):
        if time.time() - start > DEADLINE:
            break
        if can_add(r, x):
            S.add(x)
            r |= (r << x) & MASK

    best_set = set(S)
    best_size = len(S)

    # Step 3: Try greedy addition with multiple orderings
    remaining = [x for x in range(1, 101) if x not in S]
    for ordering in [remaining, sorted(remaining, reverse=True), sorted(remaining)]:
        if time.time() - start > DEADLINE:
            break
        S_try = set(S)
        r_try = reachable_bitset(S_try)
        for x in ordering:
            if x in S_try:
                continue
            if can_add(r_try, x):
                S_try.add(x)
                r_try |= (r_try << x) & MASK
        if len(S_try) > best_size:
            best_size = len(S_try)
            best_set = S_try

    # Step 4: Shuffled greedy passes
    random.seed(42)
    S_work = set(best_set)
    for _ in range(20):
        if time.time() - start > DEADLINE:
            break
        S_try = set(S_work)
        r_try = reachable_bitset(S_try)
        remaining = [x for x in range(1, 101) if x not in S_try]
        random.shuffle(remaining)
        for x in remaining:
            if time.time() - start > DEADLINE:
                break
            if can_add(r_try, x):
                S_try.add(x)
                r_try |= (r_try << x) & MASK
        if len(S_try) > len(S_work):
            S_work = S_try

    # Step 5: Local search with swaps - try removing one element and adding another
    S_work = set(S_work)
    r_work = reachable_bitset(S_work)
    improved = True
    while improved and time.time() - start < DEADLINE:
        improved = False
        S_list = sorted(S_work)
        for i, a in enumerate(S_list):
            if time.time() - start > DEADLINE:
                break
            # Try removing a
            # Recompute reachable without a
            S_no_a = S_work - {a}
            r_no_a = reachable_bitset(S_no_a)
            # Try adding each element not in S_work
            for b in range(1, 101):
                if time.time() - start > DEADLINE:
                    break
                if b in S_work:
                    continue
                if b == a:
                    continue
                if can_add(r_no_a, b):
                    S_new = S_no_a | {b}
                    if len(S_new) > len(S_work):
                        S_work = S_new
                        r_work = reachable_bitset(S_work)
                        improved = True
                        break
            if improved:
                break

    # Step 6: Final greedy pass
    S_final = set(S_work)
    r_final = reachable_bitset(S_final)
    for x in range(100, 0, -1):
        if time.time() - start > DEADLINE:
            break
        if x in S_final:
            continue
        if can_add(r_final, x):
            S_final.add(x)
            r_final |= (r_final << x) & MASK

    result = list(S_final)
    if not result:
        result = [100]

    # Final validation
    if not is_valid_bitset(reachable_bitset(result)):
        result = list(range(67, 101))

    return result
