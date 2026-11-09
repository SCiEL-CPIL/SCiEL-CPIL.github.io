"""Rewritten program: search for a small set of target sums and products,
then construct 14 distinct values and 35 edges whose edge sums/products
fall within those target sets."""
import random
import time
from itertools import combinations

N = 14
M = 35
MAXV = 10000
ALL_EDGES = list(combinations(range(N), 2))


def _score(values, edges):
    """Raw value = max(#distinct sums, #distinct products)."""
    sums = set()
    products = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        products.add(values[i] * values[j])
    return max(len(sums), len(products))


def _try_construct(target_sums, target_products, max_iters=400):
    """Try to build 14 distinct values and 35 edges with sums in target_sums
    and products in target_products. Returns (values, edges) or None."""
    ts = sorted(target_sums)
    tp = sorted(target_products)
    ts_set = set(target_sums)
    tp_set = set(target_products)

    # Precompute: for each pair of target sums, candidate (a, b) values
    # such that a+b in ts_set, a*b in tp_set, a != b, 1 <= a,b <= MAXV
    # We'll pick values greedily.

    best = None
    best_score = M + 1  # worst case all distinct

    for it in range(max_iters):
        # Randomly pick 14 values from a pool
        # Generate candidate values: pick random a, then b such that a+b in ts, a*b in tp
        vals = []
        used = set()
        ok = True
        for _ in range(N):
            # Try random target sum
            for _attempt in range(30):
                s = random.choice(ts)
                # Pick a random a < s
                if s <= 2:
                    continue
                a = random.randint(1, s - 1)
                b = s - a
                if a == b:
                    continue
                if a in used or b in used:
                    continue
                if a > MAXV or b > MAXV:
                    continue
                if a * b not in tp_set:
                    continue
                # Also check a*b is in target products
                vals.append(a)
                vals.append(b)
                used.add(a)
                used.add(b)
                break
            else:
                ok = False
                break
        if not ok or len(vals) != N:
            continue

        # Now we have N values. Pick 35 edges where sum and product are in targets.
        valid_edges = []
        for i, j in ALL_EDGES:
            s = vals[i] + vals[j]
            p = vals[i] * vals[j]
            if s in ts_set and p in tp_set:
                valid_edges.append((i, j))

        if len(valid_edges) < M:
            continue

        # Pick M random edges from valid_edges
        edges = [list(e) for e in random.sample(valid_edges, M)]
        sc = _score(vals, edges)
        if sc < best_score:
            best_score = sc
            best = (list(vals), edges)
            if best_score <= 2:
                break

    if best is not None:
        return best
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill

    # Strategy: start with small target sets and grow if needed.
    # Target: #distinct sums <= K, #distinct products <= K, minimize K.

    best_result = None
    best_score = M + 1

    # Initial baseline: random
    random.seed(42)
    for trial in range(5):
        vals = random.sample(range(1, MAXV + 1), N)
        edges = [list(e) for e in random.sample(ALL_EDGES, M)]
        sc = _score(vals, edges)
        if sc < best_score:
            best_score = sc
            best_result = (vals, edges)

    # Now try structured construction with small target sets.
    # Try K from 3 upward.
    for K in range(3, 12):
        if time.time() > deadline:
            break
        # Try several random target sets of size K
        for attempt in range(30):
            if time.time() > deadline:
                break
            # Target sums: pick K values that are "nice" - e.g., multiples
            # or random in a range
            ts_base = random.randint(10, 500)
            ts_step = random.randint(1, 20)
            target_sums = [ts_base + ts_step * i for i in range(K)]
            # Ensure all positive and distinct
            if len(set(target_sums)) != K or any(s < 2 for s in target_sums):
                continue

            # Target products: pick K values
            # Products should be consistent with sums. For a+b=s, a*b=p.
            # a and b are roots of x^2 - sx + p = 0, discriminant = s^2 - 4p >= 0
            # So p <= s^2/4 for some s in target_sums.
            max_p = max(s * s // 4 for s in target_sums)
            if max_p < 1:
                continue
            target_products = random.sample(range(1, min(max_p, MAXV * MAXV) + 1), K)

            result = _try_construct(target_sums, target_products, max_iters=200)
            if result is not None:
                vals, edges = result
                sc = _score(vals, edges)
                if sc < best_score:
                    best_score = sc
                    best_result = (vals, edges)
                    if best_score <= 2:
                        break
        if best_score <= 2:
            break

    # Local search refinement on best_result
    if best_result is not None and time.time() < deadline:
        vals, edges = best_result
        sc = best_score
        # Try swapping values or edges to reduce distinct count
        for _iter in range(5000):
            if time.time() > deadline:
                break
            if sc <= 2:
                break
            # Try changing one value
            idx = random.randint(0, N - 1)
            old_val = vals[idx]
            new_val = random.randint(1, MAXV)
            if new_val == old_val:
                continue
            if new_val in set(vals) - {old_val}:
                continue
            vals[idx] = new_val
            new_sc = _score(vals, edges)
            if new_sc < sc:
                sc = new_sc
            else:
                vals[idx] = old_val

            # Try changing one edge
            eidx = random.randint(0, M - 1)
            old_edge = edges[eidx]
            new_edge = list(random.choice(ALL_EDGES))
            if new_edge == old_edge:
                continue
            # Check no duplicate
            if new_edge in edges:
                continue
            edges[eidx] = new_edge
            new_sc = _score(vals, edges)
            if new_sc < sc:
                sc = new_sc
            else:
                edges[eidx] = old_edge

        best_result = (vals, edges)
        best_score = sc

    if best_result is None:
        # Fallback
        vals = list(range(1, N + 1))
        edges = [list(e) for e in ALL_EDGES[:M]]
        best_result = (vals, edges)

    vals, edges = best_result
    # Ensure types are correct
    vals = [int(v) for v in vals]
    edges = [[int(i), int(j)] for i, j in edges]
    return {'values': vals, 'edges': edges}
