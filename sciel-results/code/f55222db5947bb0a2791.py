"""Rewritten program: guaranteed valid fallback plus a time-budgeted search for a larger C5-free graph with alpha < 9."""
import random
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # The evaluator kills after ~150s wall-clock; keep a hard safety margin.
    deadline = start_time + 118.0

    def remaining():
        return deadline - time.time()

    # ---- Guaranteed valid fallback: 8 disjoint K4s -> n=32, alpha=8, no C5 ----
    def k4_blocks(nblocks):
        n = 4 * nblocks
        edges = []
        for b in range(nblocks):
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([4 * b + i, 4 * b + j])
        return {'n': n, 'edges': edges}

    fallback = k4_blocks(8)  # n = 32

    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def has_c5(adj, n, tlimit):
        # C5 exists iff some vertex is the start of a 4-vertex path whose last vertex
        # is adjacent to the first (with distinct vertices). Bound the search by time.
        t0 = time.time()
        for s in range(n):
            sbit = 1 << s
            stack = [(s, sbit, 0)]
            while stack:
                v, used, depth = stack.pop()
                if depth == 3:
                    if adj[v] & sbit:
                        return True
                else:
                    cand = adj[v] & ~used
                    while cand:
                        lb = cand & (-cand)
                        w = lb.bit_length() - 1
                        stack.append((w, used | lb, depth + 1))
                        cand ^= lb
            if time.time() - t0 > tlimit:
                return None  # unknown
        return False

    def alpha_is_at_least_9(adj, n, tlimit):
        # alpha(G) >= 9  <=>  complement has a clique of size 9.
        # Branch-and-bound with greedy upper bound on remaining vertices.
        comp = [((1 << n) - 1) ^ (adj[i] | (1 << i)) for i in range(n)]
        order = list(range(n))
        order.sort(key=lambda i: -bin(comp[i]).count('1'))
        t0 = time.time()

        def rec(cand, k):
            if k >= 9:
                return True
            if time.time() - t0 > tlimit:
                return None
            # upper bound: |cand| + k
            if bin(cand).count('1') + k < 9:
                return False
            # greedy upper bound on clique size within cand
            cb = cand
            gc = 0
            while cb:
                lb = cb & (-cb)
                gc += 1
                cb &= (cb - 1)
                cb &= comp[lb.bit_length() - 1]
            if gc + k < 9:
                return False
            lb = cand & (-cand)
            v = lb.bit_length() - 1
            r = rec(cand & comp[v], k + 1)
            if r is True:
                return True
            if r is None:
                return None
            return rec(cand ^ lb, k)

        return rec((1 << n) - 1, 0)

    def is_valid(n, edges, tlimit):
        adj = build_adj(n, edges)
        c5 = has_c5(adj, n, tlimit)
        if c5 is True:
            return False
        if c5 is None:
            return False  # inconclusive -> treat as invalid to be safe
        a9 = alpha_is_at_least_9(adj, n, tlimit)
        if a9 is True:
            return False
        if a9 is None:
            return False  # inconclusive -> treat as invalid to be safe
        return True

    best = fallback  # always valid

    # ---- Search: add a 33rd vertex to 8xK4 with a neighbor in every block ----
    if remaining() > 12.0:
        base_n = 32
        base_edges = fallback['edges']
        base_adj = build_adj(base_n, base_edges)
        base_valid = True  # known valid
        n = 33
        new = 32
        # Each block b occupies vertices 4b..4b+3.
        # Choose one vertex per block to be adjacent to the new vertex.
        best_edges = None
        t0 = time.time()
        attempts = 0
        while time.time() - t0 < min(remaining() - 6.0, 90.0):
            choices = [4 * b + random.randint(0, 3) for b in range(8)]
            new_edges = base_edges + [[new, c] for c in choices]
            # quick C5 check
            if has_c5(base_adj, 32, 1.0) is not False:
                pass  # base is known C5-free; skip
            ok = is_valid(n, new_edges, min(remaining() - 8.0, 6.0))
            if ok:
                best = {'n': n, 'edges': new_edges}
                break
            attempts += 1
            if attempts % 200 == 0:
                # periodic safety
                if time.time() > deadline - 5.0:
                    break

    # ---- Search: try to grow to n=34,35 via random C5-free graphs ----
    # (only if we still have a comfortable time budget)
    for target in (34, 35):
        if remaining() < 25.0:
            break
        # Build a random graph with edge probability tuned so it is likely C5-free
        # and dense enough that alpha < 9. Then verify exactly.
        t0 = time.time()
        limit = min(remaining() - 15.0, 20.0)
        found = None
        for _ in range(2000):
            if time.time() - t0 > limit:
                break
            # Random graph, then remove edges to break C5s is complex; instead use
            # a "blow-up of K4 with random cross edges" heuristic:
            # Partition into 8 blocks of ~target/8 vertices, make each block a clique,
            # and add random cross edges, then verify.
            n = target
            block_size = n // 8
            edges = []
            # clique within each block
            for b in range(8):
                lo = b * block_size
                hi = lo + block_size
                for i in range(lo, hi):
                    for j in range(i + 1, hi):
                        edges.append([i, j])
            # random cross edges
            for b in range(8):
                for c in range(b + 1, 8):
                    lo1 = b * block_size
                    hi1 = lo1 + block_size
                    lo2 = c * block_size
                    hi2 = lo2 + block_size
                    for i in range(lo1, hi1):
                        for j in range(lo2, hi2):
                            if random.random() < 0.5:
                                edges.append([i, j])
            ok = is_valid(n, edges, min(remaining() - 20.0, 5.0))
            if ok:
                found = {'n': n, 'edges': edges}
                break
        if found is not None:
            best = found
            break

    # Final safety: ensure we return before the hard deadline.
    if time.time() > deadline - 2.0:
        return fallback

    return best
# EVOLVE-BLOCK-END
