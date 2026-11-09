import time
import random
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin under 90s kill

    N = 128
    full = list(range(1, N + 1))

    # Precompute LCM table
    lcm_tbl = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            v = a * b // gcd(a, b)
            lcm_tbl[a][b] = v
            lcm_tbl[b][a] = v

    # Precompute, for each x, the set of packed pairs (a,b) with a<b
    # such that (x,a,b) is a bad triple.
    def pack(a, b):
        if a > b:
            a, b = b, a
        return a * 129 + b

    bad_pairs = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                if lcm_tbl[a][b] == lcm_tbl[a][c] == lcm_tbl[b][c]:
                    p = pack(b, c)
                    bad_pairs[a].add(p)
                    p = pack(a, c)
                    bad_pairs[b].add(p)
                    p = pack(a, b)
                    bad_pairs[c].add(p)

    def can_add(x, S_set, S_pairs):
        """Check if x can be added to S (S_set is set, S_pairs is set of packed pairs)."""
        for (a, b) in S_pairs:
            if (a * 129 + b) in bad_pairs[x]:
                return False
        return True

    def greedy(order):
        S = set()
        S_pairs = set()
        for x in order:
            if can_add(x, S, S_pairs):
                # add x: new pairs with all existing
                for a in S:
                    S_pairs.add(pack(a, x))
                S.add(x)
        return S

    best = greedy(full)
    best_size = len(best)

    # Randomized greedy restarts
    rng = random.Random(12345)
    while time.time() < deadline:
        order = full[:]
        rng.shuffle(order)
        S = greedy(order)
        if len(S) > best_size:
            best = S
            best_size = len(S)

        # Local search: try to grow best via swaps (remove one, add one or two)
        if time.time() < deadline:
            S = set(best)
            S_pairs = set()
            for a in S:
                for b in S:
                    if a < b:
                        S_pairs.add(pack(a, b))
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try adding any element not in S
                for x in full:
                    if x not in S and can_add(x, S, S_pairs):
                        for a in S:
                            S_pairs.add(pack(a, x))
                        S.add(x)
                        improved = True
                        break
                if improved:
                    continue
                # Try swap: remove r, add x
                for r in list(S):
                    S2 = S - {r}
                    S2_pairs = set()
                    for a in S2:
                        for b in S2:
                            if a < b:
                                S2_pairs.add(pack(a, b))
                    for x in full:
                        if x not in S2 and can_add(x, S2, S2_pairs):
                            # accept
                            for a in S2:
                                S2_pairs.add(pack(a, x))
                            S2.add(x)
                            S = S2
                            S_pairs = S2_pairs
                            improved = True
                            break
                    if improved:
                        break
            if len(S) > best_size:
                best = S
                best_size = len(S)

    return list(best)
