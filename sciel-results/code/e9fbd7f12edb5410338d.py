import time
import math

_START = time.time()
_HARD_DEADLINE = _START + 80.0  # Safety margin for the 90s limit


def _prime_factors(num):
    if num <= 1:
        return frozenset()
    out = []
    x = num
    if x % 2 == 0:
        out.append(2)
        while x % 2 == 0:
            x //= 2
    f = 3
    while f * f <= x:
        if x % f == 0:
            out.append(f)
            while x % f == 0:
                x //= f
        f += 2
    if x > 1:
        out.append(x)
    return frozenset(out)


def _is_smooth(num, P):
    """Check if num is P-smooth (all prime factors in P)."""
    x = num
    for p in P:
        while x % p == 0:
            x //= p
    return x == 1


def construct():
    deadline = time.time() + 80.0
    n = 12
    MAXV = 10000

    # Precompute prime factors for all possible sums (max sum 20000)
    pf_cache = {s: _prime_factors(s) for s in range(2, 20001)}

    def count_distinct_primes(arr):
        union = set()
        for i in range(n - 1):
            ai = arr[i]
            for j in range(i + 1, n):
                union |= pf_cache[ai + arr[j]]
        return len(union)

    # ------------------------------------------------------------------
    # STRATEGY 1: Graph-clique approach
    # For a candidate prime set P, build a graph on {1..MAXV} where
    # edge(u,v) iff u+v is P-smooth. Find a clique of size n.
    # ------------------------------------------------------------------
    # Candidate prime sets to try (small prime sets first)
    prime_sets = [
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
        [2, 3, 5, 7, 11, 13, 17],
    ]

    # Build adjacency for each prime set (as a list of sets for fast lookup)
    # To save memory/time, we only build adjacency for numbers up to a cap
    # and use a capped range for the clique search
    def build_adjacency(P, cap=5000):
        """Build adjacency list: adj[u] = set of v > u such that u+v is P-smooth."""
        adj = [set() for _ in range(cap + 1)]
        for u in range(1, cap + 1):
            for v in range(u + 1, cap + 1):
                s = u + v
                if s > 2 * MAXV:
                    break
                if _is_smooth(s, P):
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def greedy_clique(adj, cap, target_size):
        """Greedy maximal clique heuristic: repeatedly pick vertex with max degree in remaining subgraph."""
        remaining = set(range(1, cap + 1))
        clique = []
        while remaining and len(clique) < target_size:
            # Pick vertex with max degree within remaining
            best_v = None
            best_deg = -1
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            if best_v is None:
                break
            clique.append(best_v)
            remaining.discard(best_v)
            # Restrict remaining to neighbors of best_v
            remaining &= adj[best_v]
        return clique

    # Try each prime set and find a clique
    best_from_clique = None
    best_clique_score = 10**9

    for P in prime_sets:
        if time.time() > deadline - 20.0:
            break
        # Use a reasonable cap; larger cap = more candidates but slower
        cap = min(3000, MAXV)
        adj = build_adjacency(P, cap)
        clique = greedy_clique(adj, cap, n)
        if len(clique) >= n:
            arr = sorted(clique[:n])
            sc = count_distinct_primes(arr)
            if sc < best_clique_score:
                best_clique_score = sc
                best_from_clique = list(arr)

    # ------------------------------------------------------------------
    # STRATEGY 2: Direct search with smooth pool (from current program)
    # ------------------------------------------------------------------
    # Build a pool of 7-smooth numbers up to MAXV
    pool = set()
    for a in range(0, 14):
        v2 = 2 ** a
        if v2 > MAXV:
            break
        for b in range(0, 9):
            v3 = v2 * (3 ** b)
            if v3 > MAXV:
                break
            for c in range(0, 7):
                v5 = v3 * (5 ** c)
                if v5 > MAXV:
                    break
                for d in range(0, 6):
                    v7 = v5 * (7 ** d)
                    if v7 > MAXV:
                        break
                    if v7 >= 1:
                        pool.add(v7)

    for i in range(1, 101):
        pool.add(i)

    pool = sorted(list(pool))

    # Initial seeds
    seeds = []
    seeds.append(pool[:12])
    seeds.append([30 * i for i in range(1, 13)])
    seeds.append([30 * i + 1 for i in range(1, 13)])
    seeds.append(list(range(1, 13)))

    seq = []
    for a in range(5):
        for b in range(5):
            val = (2 ** a) * (3 ** b)
            if 1 <= val <= MAXV:
                seq.append(val)
    seq = sorted(set(seq))
    if len(seq) >= 12:
        seeds.append(seq[:12])

    # Add clique result as a seed if we found one
    if best_from_clique is not None:
        seeds.append(best_from_clique)

    best = None
    best_score = 10**9

    for s in seeds:
        if len(set(s)) != 12:
            continue
        if any(x > MAXV for x in s):
            continue
        if any(x <= 0 for x in s):
            continue
        sc = count_distinct_primes(s)
        if sc < best_score:
            best_score = sc
            best = list(s)

    if best is None:
        return list(range(1, 13))

    current = list(best)
    improved = True
    iters = 0
    max_iters = 500

    # Local search with incremental-style full recount
    while improved and iters < max_iters and time.time() < deadline - 3.0:
        improved = False
        iters += 1
        cur_score = count_distinct_primes(current)

        for idx in range(n):
            if time.time() > deadline - 3.0:
                return list(best)

            old_val = current[idx]
            existing = set(current)

            best_cand = None
            best_cand_score = cur_score

            for cand in pool:
                if cand == old_val or cand in existing:
                    continue
                if cand > MAXV:
                    break

                current[idx] = cand
                sc = count_distinct_primes(current)

                if sc < best_cand_score:
                    best_cand_score = sc
                    best_cand = cand
                else:
                    current[idx] = old_val

            if best_cand is not None:
                current[idx] = best_cand
                cur_score = best_cand_score
                improved = True
                if cur_score < best_score:
                    best_score = cur_score
                    best = list(current)
                if best_score <= 6:
                    return list(best)
                break

    return list(best)
