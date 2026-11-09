"""Stronger sum-free subset search with deadline guard."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random

    t_start = time.time()
    DEADLINE = 110.0  # safety margin under the 150s kill

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def greedy_add_remove(S, rng, max_passes=400):
        S = set(S)
        for _ in range(max_passes):
            if time.time() > DEADLINE:
                break
            improved = False
            # Try adding every element not in S.
            for x in S0:
                if x in S:
                    continue
                # x can be added iff no pair in S sums to x and 2x not in S.
                ok = True
                for a in S:
                    if a == x:
                        continue
                    if (x - a) in S:
                        ok = False
                        break
                if ok and (2 * x) not in S:
                    S.add(x)
                    improved = True
            # Try removing elements to allow more additions (local search).
            for x in list(S):
                if time.time() > DEADLINE:
                    break
                cand = S - {x}
                gain = 0
                for y in S0:
                    if y in cand:
                        continue
                    ok = True
                    for a in cand:
                        if (y - a) in cand:
                            ok = False
                            break
                    if ok and (2 * y) not in cand:
                        gain += 1
                if gain > 1:  # removing x lets us add at least 2 more
                    S = cand
                    improved = True
            if not improved:
                break
        return S

    def sum_free(S):
        S = set(S)
        for a in S:
            for b in S:
                if a + b in S:
                    return False
        return True

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S0 = instance(size, seed)
        rng = Random(seed + 1)
        # Start from odd subset, then improve.
        odd = [x for x in S0 if x % 2]
        best = greedy_add_remove(odd, rng)
        # Also try starting from empty with pure greedy add.
        empty = greedy_add_remove([], rng)
        if len(empty) > len(best):
            best = empty
        results.append(sorted(best))
    return results
# EVOLVE-BLOCK-END
