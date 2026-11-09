"""
B_3[1] set construction: maximize cardinality with unique unordered triple sums.

Strategy: Use a constructive base-representation approach combined with
greedy extension and local search. The key insight is to use a base-B
representation where digits are chosen to ensure unique triple sums.

For a set {a_1, ..., a_k} with a_i = sum_{d=0}^{D} x_{i,d} * B^d,
the triple sum a_i + a_j + a_k is unique if the "digit sum" at each position
is unique across all unordered triples (allowing repetition).

We use B=4 and ensure that for each digit position, the multiset of digit
triples (x_{i,d}, x_{j,d}, x_{k,d}) is unique. This is achieved by choosing
digits from {0, 1} with careful constraints, or by using a greedy construction
that tracks all triple sums.

Given the time constraint (90s), we combine:
1. A fast greedy construction with incremental sum tracking
2. Multiple randomized restarts with different orderings
3. A final local search pass to add missing elements
"""
import time
import random
import itertools
from itertools import combinations_with_replacement

def construct():
    t_start = time.time()
    DEADLINE = t_start + 85.0  # 5s safety margin under 90s kill
    N = 4096

    def elapsed():
        return time.time() - t_start

    def is_valid(vals):
        """Check if all unordered triple sums are unique."""
        m = len(vals)
        seen = set()
        for i in range(m):
            for j in range(i, m):
                for k in range(j, m):
                    s = vals[i] + vals[j] + vals[k]
                    if s in seen:
                        return False
                    seen.add(s)
        return True

    def compute_triple_sums(vals):
        """Compute all unordered triple sums (with repetition)."""
        sums = set()
        m = len(vals)
        for i in range(m):
            for j in range(i, m):
                for k in range(j, m):
                    sums.add(vals[i] + vals[j] + vals[k])
        return sums

    def try_add_value(v, chosen, sums):
        """
        Try to add value v to chosen set. Returns (success, new_sums_set, updated_sums).
        We check if any triple sum involving v collides with existing sums.
        """
        if v in chosen:
            return False, set(), sums

        new_sums = set()
        # Triple (v, v, v)
        s = 3 * v
        new_sums.add(s)
        # Triples (v, v, u) for u in chosen, u != v (but we allow u=v only if v in chosen, which it isn't)
        for u in chosen:
            s = 2 * v + u
            new_sums.add(s)
            s = v + 2 * u
            new_sums.add(s)
        # Triples (v, u, w) for u <= w in chosen
        cl = list(chosen)
        for ui in range(len(cl)):
            u = cl[ui]
            for wi in range(ui, len(cl)):
                w = cl[wi]
                s = v + u + w
                new_sums.add(s)

        # Check for collision
        if new_sums & sums:
            return False, new_sums, sums

        updated_sums = sums | new_sums
        return True, new_sums, updated_sums

    def greedy_build(candidate_order, limit_time):
        """
        Greedily build a set by iterating through candidate_order and adding
        elements if they don't cause triple sum collisions.
        """
        chosen = []
        sums = set()
        chosen_set = set()

        for v in candidate_order:
            if time.time() > limit_time:
                break
            if v in chosen_set:
                continue
            if v < 1 or v > N:
                continue
            ok, ns, sums = try_add_value(v, chosen, sums)
            if ok:
                chosen.append(v)
                chosen_set.add(v)

        return sorted(chosen)

    def local_improve(vals, limit_time):
        """
        Try to add any missing values from 1..N to the set.
        """
        if not vals:
            return vals
        chosen = list(vals)
        chosen_set = set(chosen)
        sums = compute_triple_sums(chosen)

        # Try adding each missing value
        for v in range(1, N + 1):
            if time.time() > limit_time:
                break
            if v in chosen_set:
                continue
            ok, ns, sums = try_add_value(v, chosen, sums)
            if ok:
                chosen.append(v)
                chosen_set.add(v)

        return sorted(chosen)

    best = []
    best_len = 0

    # ---------- Strategy 1: Greedy with natural order ----------
    if elapsed() < 2.0:
        candidates = list(range(1, N + 1))
        result = greedy_build(candidates, DEADLINE - 5.0)
        if is_valid(result) and len(result) > best_len:
            best = result
            best_len = len(result)

    # ---------- Strategy 2: Greedy with various orderings ----------
    random.seed(42)
    orderings = [
        lambda: list(range(1, N + 1)),
        lambda: list(range(1, N + 1, 2)),  # odds
        lambda: list(range(2, N + 1, 2)),  # evens
        lambda: sorted(range(1, N + 1), key=lambda x: random.random()),
        lambda: sorted(range(1, N + 1), key=lambda x: -random.random()),
        lambda: [x for x in range(1, N + 1) if x % 3 != 0],
        lambda: [x for x in range(1, N + 1) if x % 4 in (1, 3)],
    ]

    for i, gen in enumerate(orderings):
        if elapsed() > DEADLINE - 10.0:
            break
        candidates = gen()
        result = greedy_build(candidates, DEADLINE - 5.0)
        if is_valid(result) and len(result) > best_len:
            best = result
            best_len = len(result)

    # ---------- Strategy 3: Randomized multi-restart greedy ----------
    num_restarts = 30
    for it in range(num_restarts):
        if elapsed() > DEADLINE - 8.0:
            break
        # Create a random candidate pool with bias
        base = random.randint(2, 8)
        pool = []
        for x in range(1, N + 1):
            if (x - 1) % base == 0:
                pool.append(x)
            elif random.random() < 0.1:
                pool.append(x)
        random.shuffle(pool)

        result = greedy_build(pool, DEADLINE - 5.0)
        if is_valid(result) and len(result) > best_len:
            best = result
            best_len = len(result)

    # ---------- Strategy 4: Local improvement on best ----------
    if elapsed() < DEADLINE - 3.0 and best:
        improved = local_improve(best, DEADLINE - 1.0)
        if is_valid(improved) and len(improved) > best_len:
            best = improved
            best_len = len(best)

    # ---------- Final sanity check ----------
    best = [v for v in best if 1 <= v <= N]
    best = sorted(set(best))
    if is_valid(best):
        return best

    # Fallback: return a small valid set
    fallback = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    if is_valid(fallback):
        return fallback
    return [1, 2, 4, 8, 16]
