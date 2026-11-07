"""Find maximum valid edge set: no 3-edge path s-u-v-t with deg(s)==deg(t)."""
import time

# EVOLVE-BLOCK-START
N = 15
ALL_EDGES = [(i, j) for i in range(N) for j in range(i + 1, N)]
EDGE_IDX = {(i, j): k for k, (i, j) in enumerate(ALL_EDGES)}
M = len(ALL_EDGES)

def _degrees_from_mask(mask):
    deg = [0] * N
    for k, (u, v) in enumerate(ALL_EDGES):
        if mask >> k & 1:
            deg[u] += 1
            deg[v] += 1
    return deg

def _is_valid_mask(mask):
    deg = _degrees_from_mask(mask)
    for k, (u, v) in enumerate(ALL_EDGES):
        if not (mask >> k & 1):
            continue
        # s in N(u) \ {v}, t in N(v) \ {u}, s != t
        for s in range(N):
            if s == u or s == v:
                continue
            if not (mask >> EDGE_IDX[min(s, u), max(s, u)] & 1):
                continue
            for t in range(N):
                if t == u or t == v or t == s:
                    continue
                if not (mask >> EDGE_IDX[min(t, v), max(t, v)] & 1):
                    continue
                if deg[s] == deg[t]:
                    return False
    return True

def construct():
    deadline = time.time() + 80.0
    best_mask = 0
    best_count = 0

    # Start from a good greedy seed: complete bipartite-ish / random greedy
    import random
    rng = random.Random(12345)

    def greedy(seed_edges):
        mask = 0
        for k in seed_edges:
            trial = mask | (1 << k)
            if _is_valid_mask(trial):
                mask = trial
        return mask

    # Greedy over shuffled edges
    order = list(range(M))
    rng.shuffle(order)
    m = greedy(order)
    if m.bit_count() > best_count:
        best_mask, best_count = m, m.bit_count()

    # Local improvement: try adding each missing edge
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for k in range(M):
            if best_mask >> k & 1:
                continue
            trial = best_mask | (1 << k)
            if _is_valid_mask(trial):
                best_mask = trial
                best_count += 1
                improved = True

    # Branch and bound to try to beat best_count
    # We build a conflict structure: two edges conflict if they cannot coexist.
    # Simpler: recursive search adding edges, pruning when upper bound <= best.
    # Upper bound: remaining candidate edges + current count.
    candidates = list(range(M))

    def rec(idx, mask, count):
        nonlocal best_mask, best_count
        if time.time() > deadline:
            return
        if count + (len(candidates) - idx) <= best_count:
            return
        if idx == len(candidates):
            if count > best_count:
                best_mask, best_count = mask, count
            return
        k = candidates[idx]
        # skip
        rec(idx + 1, mask, count)
        # take
        trial = mask | (1 << k)
        if _is_valid_mask(trial):
            rec(idx + 1, trial, count + 1)

    # Only run B&B if there is headroom; start from best_mask to reduce space
    if time.time() < deadline - 2 and best_count < M:
        # candidates = edges not in best_mask, ordered by heuristic (try high-degree-impact first)
        rem = [k for k in range(M) if not (best_mask >> k & 1)]
        # order: edges whose endpoints currently have low degree first (easier to add)
        deg = _degrees_from_mask(best_mask)
        rem.sort(key=lambda k: deg[ALL_EDGES[k][0]] + deg[ALL_EDGES[k][1]])
        base = best_mask
        basec = best_count
        # recursive add-only search
        def rec_add(idx, mask, count):
            nonlocal best_mask, best_count
            if time.time() > deadline:
                return
            if count + (len(rem) - idx) <= best_count:
                return
            if idx == len(rem):
                if count > best_count:
                    best_mask, best_count = mask, count
                return
            k = rem[idx]
            # skip
            rec_add(idx + 1, mask, count)
            # take
            trial = mask | (1 << k)
            if _is_valid_mask(trial):
                rec_add(idx + 1, trial, count + 1)
        rec_add(0, base, basec)

    # Final polish: try to add any remaining edge
    for k in range(M):
        if best_mask >> k & 1:
            continue
        trial = best_mask | (1 << k)
        if _is_valid_mask(trial):
            best_mask = trial
            best_count += 1

    edges = [[u, v] for k, (u, v) in enumerate(ALL_EDGES) if best_mask >> k & 1]
    return edges
# EVOLVE-BLOCK-END
