"""
4-colouring of K_n with no monochromatic C6 (R_4(C6) > n).

Combines the vectorized incremental C6-violation counting from the inspiration
with a girth-aware greedy initialisation and a multi-restart simulated-annealing
local search.  The key structural change relative to the current program is that
validation is no longer a full DFS over every colour class; instead we precompute
every 6-cycle once per n, maintain a running count of monochromatic cycles, and
update that count in O(K) per edge flip (K = number of cycles containing the
edge).  This lets us run millions of search steps within the time budget.
"""

import time
import math
import random
import itertools
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edge_index(i, j, n):
    """Lexicographic edge index for pair (i,j), i < j, in K_n."""
    if i > j:
        i, j = j, i
    # number of edges before vertex i: i*(i-1)/2
    # within vertex i's block, edge to j is at offset (j - i - 1)
    return i * (2 * n - i - 3) // 2 + (j - i - 1)


def _build_c6_cycle_data(n):
    """
    Precompute all C6 cycles in K_n as edge-index arrays.

    Returns
    -------
    cycles : np.ndarray, shape (N, 6), dtype int32
        Each row is the 6 edge-indices of a (non-induced) 6-cycle.
        We enumerate each undirected cycle once by fixing the smallest
        vertex and requiring the second vertex < the last vertex.
    edge_to_other_edges : list of np.ndarray
        edge_to_other_edges[e] has shape (K_e, 5) – for each cycle that
        contains edge e, the other 5 edge-indices.
    """
    verts = list(range(n))
    cycles_list = []

    for combo in itertools.combinations(verts, 6):
        v0 = combo[0]
        others = combo[1:]
        # Permute the 5 remaining vertices; fix ordering to avoid duplicates:
        # cycle is v0 -> p0 -> p1 -> p2 -> p3 -> p4 -> v0
        # We require p0 < p4 to break the reversal symmetry.
        for perm in itertools.permutations(others):
            if perm[0] > perm[-1]:
                continue
            path = [v0] + list(perm)
            edges = []
            for k in range(6):
                u = path[k]
                v = path[(k + 1) % 6]
                edges.append(_edge_index(u, v, n))
            cycles_list.append(edges)

    m = n * (n - 1) // 2
    if not cycles_list:
        return np.array([], dtype=np.int32).reshape(0, 6), [np.array([], dtype=np.int32).reshape(0, 5) for _ in range(m)]

    cycles = np.array(cycles_list, dtype=np.int32)
    N = len(cycles)

    # Map edge -> list of cycle indices
    edge_to_cycles = [[] for _ in range(m)]
    for ci in range(N):
        for e in cycles[ci]:
            edge_to_cycles[e].append(ci)

    edge_to_other_edges = [None] * m
    for e in range(m):
        cids = edge_to_cycles[e]
        if not cids:
            edge_to_other_edges[e] = np.array([], dtype=np.int32).reshape(0, 5)
            continue
        other = []
        for cid in cids:
            ce = cycles[cid]
            mask = ce != e
            other.append(ce[mask])
        edge_to_other_edges[e] = np.array(other, dtype=np.int32)

    return cycles, edge_to_other_edges


def _count_mono_cycles(colors, cycles):
    """Count monochromatic 6-cycles.  colors: np.uint8 array, cycles: (N,6) int32."""
    if len(cycles) == 0:
        return 0
    cc = colors[cycles]  # (N, 6)
    first = cc[:, 0]
    return int(np.sum(np.all(cc == first[:, None], axis=1)))


def _greedy_init(n, seed):
    """
    Greedy edge-insertion: process edges in lexicographic order; for each edge
    pick the colour that minimises the number of *new* monochromatic C6s it
    would create (considering only edges already coloured).  Ties broken
    randomly.  This is a heuristic; it does not guarantee zero violations but
    gives a much better starting point than uniform random.
    """
    rng = np.random.default_rng(seed)
    m = n * (n - 1) // 2
    colors = np.full(m, -1, dtype=np.int8)

    # We need to check, for a candidate colour c on edge e, how many C6s would
    # become monochromatic.  A C6 becomes mono if all 6 edges are c.  When we
    # set edge e to c, a C6 containing e becomes mono iff the other 5 edges
    # are already all c.
    #
    # To do this efficiently during greedy, we maintain for each edge e and
    # colour c the count of C6s containing e where the other 5 edges are all c.
    # That's expensive to maintain incrementally.  Instead, for n <= 20 the
    # number of C6s is manageable (~hundreds of thousands at most), so we can
    # just check directly but vectorise.

    cycles, edge_to_other = _build_c6_cycle_data(n)
    N = len(cycles)

    if N == 0:
        return np.zeros(m, dtype=np.uint8)

    # Precompute: for each edge e, list of (cycle_idx) – already have edge_to_other
    # For greedy we'll just iterate edges and for each candidate colour check
    # the relevant cycles.  For n up to ~17, m <= 136, and per edge the number
    # of containing cycles is moderate.

    # Build edge -> cycle-indices map
    edge_to_cyc_ids = [[] for _ in range(m)]
    for ci in range(N):
        for e in cycles[ci]:
            edge_to_cyc_ids[e].append(ci)

    for e in range(m):
        cids = edge_to_cyc_ids[e]
        best_c = 0
        best_cnt = float('inf')
        for c in range(4):
            cnt = 0
            for cid in cids:
                other = cycles[cid]
                # check if all other 5 edges are already colour c
                ok = True
                for oe in other:
                    if oe == e:
                        continue
                    if colors[oe] != c:
                        ok = False
                        break
                if ok:
                    cnt += 1
            if cnt < best_cnt:
                best_cnt = cnt
                best_c = c
            elif cnt == best_cnt:
                if rng.random() < 0.5:
                    best_c = c
        colors[e] = best_c

    return colors.astype(np.uint8)


# ---------------------------------------------------------------------------
# Main search for a given n
# ---------------------------------------------------------------------------

def _search_n(n, time_budget, seed_offset=0):
    """
    Try to find a valid 4-colouring of K_n (no mono C6) within time_budget
    seconds.  Returns {'n': n, 'colors': [...]} or None.
    """
    if time.time() > time_budget:
        return None

    m = n * (n - 1) // 2
    rng = np.random.default_rng(42 + seed_offset)
    random.seed(42 + seed_offset)

    cycles, edge_to_other = _build_c6_cycle_data(n)
    N = len(cycles)

    if N == 0:
        return {'n': n, 'colors': np.zeros(m, dtype=np.uint8).tolist()}

    # ---- initialisation: greedy ----
    colors = _greedy_init(n, seed=42 + seed_offset)
    bad = _count_mono_cycles(colors, cycles)

    if bad == 0:
        return {'n': n, 'colors': colors.tolist()}

    best_colors = colors.copy()
    best_bad = bad

    # ---- simulated annealing with incremental delta ----
    T_start = 2.0
    T_end = 0.01
    t0 = time.time()

    while True:
        elapsed = time.time() - t0
        if elapsed >= time_budget:
            break
        if time.time() > time_budget + 2:  # hard stop
            break

        # pick a random edge
        e = rng.integers(0, m)
        c_old = int(colors[e])
        # pick a different colour
        c_new = int(rng.integers(0, 4))
        while c_new == c_old:
            c_new = int(rng.integers(0, 4))

        # incremental delta
        other = edge_to_other[e]
        if len(other) == 0:
            delta = 0
        else:
            oc = colors[other]  # (K, 5)
            delta = int(np.sum(np.all(oc == c_new, axis=1))) - int(np.sum(np.all(oc == c_old, axis=1)))

        # acceptance
        if delta <= 0:
            accept = True
        else:
            progress = min(1.0, elapsed / time_budget)
            T = T_start * (T_end / T_start) ** progress
            if delta / T > 50:
                accept = False
            else:
                accept = random.random() < math.exp(-delta / T)

        if accept:
            colors[e] = c_new
            bad += delta
            if bad < best_bad:
                best_bad = bad
                best_colors = colors.copy()
                if best_bad == 0:
                    return {'n': n, 'colors': colors.tolist()}

    if best_bad == 0:
        return {'n': n, 'colors': best_colors.tolist()}
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def construct():
    start = time.time()
    deadline = start + 115.0  # safety margin vs 120 s limit

    # Fallback: trivially valid small colouring
    best_result = {'n': 5, 'colors': [0] * 10}

    # Targets from high to low; allocate time budgets
    targets = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]
    # Rough time allocation (seconds)
    alloc = {17: 35, 16: 25, 15: 20, 14: 15, 13: 12, 12: 10,
             11: 8, 10: 6, 9: 4, 8: 3, 7: 2, 6: 1}

    for n in targets:
        if time.time() > deadline - 5:
            break
        tb = alloc.get(n, 5)
        # Multiple restarts within the budget
        num_restarts = max(1, min(5, tb // 5))
        per_restart = tb / num_restarts

        found = False
        for r in range(num_restarts):
            if time.time() > deadline - 3:
                break
            result = _search_n(n, per_restart, seed_offset=r * 1000)
            if result is not None:
                best_result = result
                found = True
                break

        if not found:
            # If we can't find for this n, try the next lower one
            continue

        # If we found a valid colouring for n, try to extend by one vertex
        # (only if we have time)
        if time.time() < deadline - 10:
            cur_n = best_result['n']
            cur_colors = np.array(best_result['colors'], dtype=np.uint8)
            for new_n in range(cur_n + 1, 21):
                if time.time() > deadline - 3:
                    break
                # Try to extend: add new_n - cur_n vertices
                # For simplicity, try random extension with a short SA
                ext_budget = min(8.0, deadline - 3 - time.time())
                if ext_budget < 1:
                    break
                ext_result = _search_n(new_n, ext_budget, seed_offset=99999)
                if ext_result is not None:
                    best_result = ext_result
                    cur_n = new_n
                    cur_colors = np.array(ext_result['colors'], dtype=np.uint8)
                else:
                    break

    return best_result
