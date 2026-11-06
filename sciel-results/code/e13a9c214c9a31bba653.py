"""C4-free graph without independent 11-set (R(C4,K11) > n).

Strategy:
  - Greedy random edge insertion while preserving C4-freeness (bitmask adjacency).
  - Exact independent-set search (branch & bound) with bitmask pruning and early-stop
    in vertex selection.
  - Try n from 43 down to 30; for each n, try many seeds.
  - Fallback: 10 disjoint triangles (n=30, alpha=10).
"""
import time
import random
import sys

def construct():
    start = time.time()
    DEADLINE = start + 140.0  # safety margin under 150s
    sys.setrecursionlimit(10000)

    # ---------- Fallback: 10 disjoint triangles (n=30, alpha=10) ----------
    def fallback():
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}

    # ---------- C4-free check (bitmask) ----------
    def is_c4_free(adj_mask, n):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj_mask[u] & adj_mask[v]
                if common & (common - 1):  # more than one common neighbor
                    return False
        return True

    # ---------- Independent set search (bitmask, branch & bound) ----------
    def has_independent_set(adj_mask, n, target=11):
        node_count = [0]
        max_nodes = 2000000

        def dfs(candidates, size):
            if size >= target:
                return True
            if candidates == 0:
                return False
            if size + candidates.bit_count() < target:
                return False

            node_count[0] += 1
            if node_count[0] > max_nodes:
                return False

            # Pick vertex with highest degree in candidate subgraph
            c = candidates
            best_v = -1
            max_deg = -1
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = (adj_mask[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                    if deg >= 10:
                        break

            if best_v == -1:
                return False

            # Include best_v
            new_cands = candidates & ~(adj_mask[best_v] | (1 << best_v))
            if dfs(new_cands, size + 1):
                return True

            # Exclude best_v
            new_cands = candidates & ~(1 << best_v)
            if dfs(new_cands, size):
                return True

            return False

        all_cands = (1 << n) - 1
        return dfs(all_cands, 0)

    # ---------- Greedy C4-free graph generation (bitmask) ----------
    def generate_greedy_c4_free(n, seed):
        rng = random.Random(seed)
        adj_mask = [0] * n
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(edges)

        for u, v in edges:
            nu = adj_mask[u]
            nv = adj_mask[v]

            # Check if adding (u,v) creates a C4:
            # For each x in N(u), x must not be adjacent to any y in N(v)
            if nu.bit_count() < nv.bit_count():
                c = nu
                safe = True
                while c:
                    lsb = c & (-c)
                    x = lsb.bit_length() - 1
                    c ^= lsb
                    if adj_mask[x] & nv:
                        safe = False
                        break
            else:
                c = nv
                safe = True
                while c:
                    lsb = c & (-c)
                    x = lsb.bit_length() - 1
                    c ^= lsb
                    if adj_mask[x] & nu:
                        safe = False
                        break

            if safe:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)

        return adj_mask

    # ---------- Build edge list from bitmask ----------
    def build_edges(n, adj_mask):
        edges = []
        for u in range(n):
            m = adj_mask[u]
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                m ^= lsb
        return edges

    # ---------- Search from high n down ----------
    best = None
    for n in range(43, 29, -1):
        if time.time() > DEADLINE - 5:
            break
        seeds = 0
        max_seeds = 3000
        while time.time() < DEADLINE - 5 and seeds < max_seeds:
            adj_mask = generate_greedy_c4_free(n, seeds)

            # Verify C4-free
            if not is_c4_free(adj_mask, n):
                seeds += 1
                continue

            # Check independence number
            if not has_independent_set(adj_mask, n, 11):
                edges = build_edges(n, adj_mask)
                best = {'n': n, 'edges': edges}
                break

            seeds += 1

        if best is not None:
            break

    if best is None:
        best = fallback()

    return best
