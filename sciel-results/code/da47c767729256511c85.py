import time
import numpy as np

N = 16
DEADLINE = 130.0  # wall-clock safety margin below the 150s kill


def _q3_vertices():
    return [0, 1, 2, 3, 4, 5, 6, 7]


def _q3_edges():
    # 12 edges of the 3-cube Q3 on vertices 0..7
    return [[0, 1], [0, 2], [0, 4],
            [1, 3], [1, 5],
            [2, 3], [2, 6],
            [3, 7],
            [4, 5], [4, 6],
            [5, 7],
            [6, 7]]


_Q3_V = _q3_vertices()
_Q3_E = _q3_edges()


def _count_q3(A, n):
    """Exact count of subgraphs isomorphic to Q3 in the graph with 0/1 adjacency A."""
    m = 12
    count = 0
    for v0 in range(n):
        nbs0 = [x for x in range(n) if A[v0, x]]
        if len(nbs0) < 3:
            continue
        for i0 in range(len(nbs0)):
            v1 = nbs0[i0]
            nbs1 = [x for x in range(n) if A[v1, x]]
            if len(nbs1) < 2:
                continue
            for i1 in range(len(nbs1)):
                v2 = nbs1[i1]
                if v2 == v0:
                    continue
                nbs2 = [x for x in range(n) if A[v2, x]]
                if len(nbs2) < 2:
                    continue
                for i2 in range(len(nbs2)):
                    v3 = nbs2[i2]
                    if v3 in (v0, v1, v2):
                        continue
                    nbs3 = [x for x in range(n) if A[v3, x]]
                    if len(nbs3) < 1:
                        continue
                    for i3 in range(len(nbs3)):
                        v4 = nbs3[i3]
                        if v4 in (v0, v1, v2, v3):
                            continue
                        if not A[v4, v1]:
                            continue
                        nbs4 = [x for x in range(n) if A[v4, x]]
                        if len(nbs4) < 2:
                            continue
                        for i4 in range(len(nbs4)):
                            v5 = nbs4[i4]
                            if v5 in (v0, v1, v2, v3, v4):
                                continue
                            if not A[v5, v2]:
                                continue
                            if not A[v5, v3]:
                                continue
                            nbs5 = [x for x in range(n) if A[v5, x]]
                            if len(nbs5) < 2:
                                continue
                            for i5 in range(len(nbs5)):
                                v6 = nbs5[i5]
                                if v6 in (v0, v1, v2, v3, v4, v5):
                                    continue
                                if not A[v6, v2]:
                                    continue
                                if not A[v6, v4]:
                                    continue
                                nbs6 = [x for x in range(n) if A[v6, x]]
                                if len(nbs6) < 1:
                                    continue
                                for i6 in range(len(nbs6)):
                                    v7 = nbs6[i6]
                                    if v7 in (v0, v1, v2, v3, v4, v5, v6):
                                        continue
                                    if not A[v7, v3]:
                                        continue
                                    if not A[v7, v5]:
                                        continue
                                    if not A[v7, v6]:
                                        continue
                                    count += 1
    return count


def _edges_from_A(A, n):
    return [[i, j] for i in range(n) for j in range(i + 1, n) if A[i, j]]


def _c4_counts(A, n):
    """Return an array c4[i,j] = number of 4-cycles through edge (i,j)."""
    A2 = A @ A
    c4 = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                c4[i, j] = A2[i, j] - 1
    return c4


def construct():
    t_start = time.time()
    n = N
    # Start from the complete graph K16
    A = np.ones((n, n), dtype=np.int64)
    np.fill_diagonal(A, 0)

    best_A = A.copy()
    best_edges = _edges_from_A(best_A, n)
    best_count = len(best_edges)

    # Precompute Q3 data for exact verification
    q3v = _Q3_V
    q3e = _Q3_E

    # Greedy edge removal driven by 4-cycle reduction
    # We maintain the full 4-cycle count matrix and update it after each removal.
    c4 = _c4_counts(A, n)

    # Candidate edge list (all edges present)
    present = set()
    for i in range(n):
        for j in range(i + 1, n):
            present.add((i, j))

    # We remove edges one at a time while Q3-free is not guaranteed;
    # but we want to maximize edges, so we only remove when it helps.
    # Strategy: iteratively find the edge whose removal reduces total 4-cycles most,
    # remove it, and repeat until no Q3 remains or time runs out.
    # However removing an edge that is not in any Q3 is wasteful; so we first
    # check if the current graph is Q3-free; if so we are done.

    # Because K16 has many Q3s, we remove edges greedily.
    # To keep it fast, we recompute Q3 count periodically and stop early if free.

    # Initial exact check
    if _count_q3(A, n) == 0:
        return best_edges

    # Greedy loop
    while time.time() - t_start < DEADLINE - 5.0:
        # Find the edge with maximum c4 count
        max_c = -1
        max_edge = None
        for (i, j) in present:
            c = c4[i, j]
            if c > max_c:
                max_c = c
                max_edge = (i, j)
        if max_edge is None or max_c <= 0:
            # No edge lies on a 4-cycle; if still Q3 present we need another rule.
            # Fall back: remove the edge with highest degree sum.
            best_deg = -1
            for (i, j) in present:
                d = A[i].sum() + A[j].sum()
                if d > best_deg:
                    best_deg = d
                    max_edge = (i, j)
            if max_edge is None:
                break
        i, j = max_edge
        # Remove edge (i,j)
        A[i, j] = 0
        A[j, i] = 0
        present.discard((i, j))

        # Update c4 counts: removing edge (i,j) affects c4 for all edges.
        # A 4-cycle containing (i,j) is removed; for each other edge (a,b),
        # c4[a,b] decreases by the number of 4-cycles that use both (i,j) and (a,b).
        # That equals the number of common neighbors pairs... Simpler: recompute
        # c4 for edges incident to i or j, and also for edges sharing a 4-cycle.
        # For correctness and simplicity given n=16, recompute c4 fully but
        # only occasionally; here recompute every removal is O(n^3) ~ 4096 ops,
        # which is fine.
        c4 = _c4_counts(A, n)

        # Periodic exact Q3 check (every few removals) to detect when free
        # We track removals count
        # Use a counter via len(present)
        removed = (n * (n - 1) // 2) - len(present)
        if removed % 3 == 0:
            if _count_q3(A, n) == 0:
                # Q3-free achieved
                cand_edges = _edges_from_A(A, n)
                if len(cand_edges) > best_count:
                    best_count = len(cand_edges)
                    best_edges = cand_edges
                return best_edges

    # Time nearly up: ensure we return a valid (Q3-free) graph.
    # If current A is Q3-free, use it; otherwise fall back to best known valid.
    if _count_q3(A, n) == 0:
        cand_edges = _edges_from_A(A, n)
        if len(cand_edges) > best_count:
            best_count = len(cand_edges)
            best_edges = cand_edges

    # If we never found a Q3-free graph (shouldn't happen since K3,13 is a
    # fallback), fall back to the guaranteed seed construction.
    if _count_q3(np.array([[1 if (i < 3 and j >= 3) or (j < 3 and i >= 3) else 0 for j in range(n)] for i in range(n)]), n) == 0:
        pass  # fallback valid

    # Guarantee validity: if best_edges graph still contains Q3, use K3,13.
    # Reconstruct A from best_edges for a final check.
    A_best = np.zeros((n, n), dtype=np.int64)
    for (i, j) in best_edges:
        A_best[i, j] = 1
        A_best[j, i] = 1
    if _count_q3(A_best, n) != 0:
        # Use the guaranteed K3,13
        best_edges = [[a, b] for a in range(3) for b in range(3, n)]

    return best_edges
