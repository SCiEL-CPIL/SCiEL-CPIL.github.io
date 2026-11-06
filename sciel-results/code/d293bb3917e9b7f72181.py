"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 140  # 10s safety margin before 150s kill

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def solve_one(vals):
        n = len(vals)
        if n == 0:
            return []

        # Initial: odd numbers
        S = [x for x in vals if x % 2 != 0]
        S_set = set(S)
        excluded = [x for x in vals if x not in S_set]

        if time.time() > deadline:
            return S

        # Greedy add
        while time.time() < deadline:
            added = False
            new_excluded = []
            for x in excluded:
                ok = True
                if x + x in S_set:
                    ok = False
                else:
                    for a in S:
                        if a + x in S_set:
                            ok = False
                            break
                if ok:
                    S.append(x)
                    S_set.add(x)
                    added = True
                else:
                    new_excluded.append(x)
            excluded = new_excluded
            if not added:
                break

        # Local search swaps
        while time.time() < deadline:
            improved = False
            for i in range(len(S)):
                if time.time() > deadline:
                    break
                y = S[i]
                for j in range(len(excluded)):
                    if time.time() > deadline:
                        break
                    x = excluded[j]
                    if x == y:
                        continue
                    ns = S_set - {y}
                    if x in ns:
                        continue
                    ok = True
                    if x + x in ns:
                        ok = False
                    else:
                        for a in S:
                            if a == y:
                                continue
                            if a + x in ns:
                                ok = False
                                break
                    if ok:
                        S[i] = x
                        S_set.remove(y)
                        S_set.add(x)
                        excluded[j] = y
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

        # Final greedy add
        while time.time() < deadline:
            added = False
            new_excluded = []
            for x in excluded:
                ok = True
                if x + x in S_set:
                    ok = False
                else:
                    for a in S:
                        if a + x in S_set:
                            ok = False
                            break
                if ok:
                    S.append(x)
                    S_set.add(x)
                    added = True
                else:
                    new_excluded.append(x)
            excluded = new_excluded
            if not added:
                break

        return S

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        results.append(solve_one(vals))
        if time.time() > deadline:
            break

    while len(results) < 3:
        results.append([])

    return results
# EVOLVE-BLOCK-END
