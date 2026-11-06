"""Model-and-optimize approach for the 16-vertex, 65-edge triangle degree-sum problem.

Combines:
  - Systematic full-neighborhood local search (current program)
  - Bitmask/NumPy fast scoring (inspiration 2)
  - Spectral proxy as a secondary tie-breaker / perturbation guide
  - Final high-precision polish pass
"""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_DEADLINE = 150.0
_DEADLINE_GUARD = 128.0  # safety margin

def _build_mask(n, edge_set):
    """Build a boolean numpy adjacency matrix from an edge set."""
    mask = np.zeros((n, n), dtype=np.uint8)
    for u, v in edge_set:
        mask[u, v] = 1
        mask[v, u] = 1
    return mask

def _score_from_mask(mask, n=16):
    """Compute max triangle degree-sum from a boolean adjacency matrix.
    Uses numpy for speed: for each edge (i,j) with i<j, find common neighbors k>j."""
    deg = mask.sum(axis=1)
    best = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if mask[i, j]:
                # Common neighbors k > j: connected to both i and j
                common = mask[i, j+1:] & mask[j, j+1:]
                if np.any(common):
                    ks = np.where(common)[0] + j + 1
                    # Vectorized degree sum
                    sums = deg[i] + deg[j] + deg[ks]
                    s = int(sums.max())
                    if s > best:
                        best = s
    return best

def _score_from_edge_set(n, edge_set):
    """Score from edge set via mask (fast path)."""
    mask = _build_mask(n, edge_set)
    return _score_from_mask(mask, n)

def _spectral_proxy(n, edge_set):
    """Largest eigenvalue of adjacency matrix as a smooth proxy.
    Not the objective, but useful for tie-breaking and perturbation guidance."""
    mask = _build_mask(n, edge_set).astype(np.float64)
    try:
        eigvals = np.linalg.eigvalsh(mask)
        return float(eigvals[-1])
    except Exception:
        return 0.0

def _local_search(edge_set, n, max_time=None):
    """Systematic full-neighborhood local search with bitmask scoring.
    Tries all single-edge swaps (remove one edge, add one non-edge) and accepts
    the best improving swap. Uses numpy bitmask for fast score evaluation."""
    if max_time is None:
        max_time = _DEADLINE_GUARD - time.time()
    start = time.time()

    current_score = _score_from_mask(_build_mask(n, edge_set), n)
    best_score = current_score
    best_edge_set = set(edge_set)

    # Precompute all possible edges
    all_possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_edge_set = set(all_possible_edges)

    while True:
        if time.time() - start > max_time or time.time() > _DEADLINE_GUARD:
            break

        current_edges = list(edge_set)
        non_edges = [e for e in all_possible_edges if e not in edge_set]

        best_swap_score = current_score
        best_swap = None

        # Try all swaps: remove e_rm, add e_add
        for e_rm in current_edges:
            u, v = e_rm
            edge_set.discard(e_rm)

            for e_add in non_edges:
                a, b = e_add
                edge_set.add(e_add)

                # Fast score via bitmask
                mask = _build_mask(n, edge_set)
                score = _score_from_mask(mask, n)

                edge_set.discard(e_add)

                if score < best_swap_score:
                    best_swap_score = score
                    best_swap = (e_rm, e_add)

            edge_set.add(e_rm)

        if best_swap:
            e_rm, e_add = best_swap
            edge_set.discard(e_rm)
            edge_set.add(e_add)
            current_score = best_swap_score
            if current_score < best_score:
                best_score = current_score
                best_edge_set = set(edge_set)
        else:
            # Local minimum reached
            break

    return best_score, best_edge_set

def _perturb(edge_set, n, k):
    """Perturb by flipping k random edges (maintaining exactly n_edges)."""
    es = set(edge_set)
    n_edges = len(es)
    all_possible = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for _ in range(k):
        existing = list(es)
        if not existing:
            break
        e_rm = random.choice(existing)
        es.discard(e_rm)
        # Find a non-edge to add
        for _try in range(200):
            a = random.randint(0, n - 1)
            b = random.randint(0, n - 1)
            if a == b:
                continue
            if a > b:
                a, b = b, a
            if (a, b) in es:
                continue
            es.add((a, b))
            break
    return es

def _polish(edge_set, n):
    """High-precision final polish: iterate local search with very tight time budget,
    using spectral proxy to guide which swaps to try first."""
    start = time.time()
    budget = min(15.0, _DEADLINE_GUARD - time.time() - 2.0)
    if budget <= 0:
        return _score_from_mask(_build_mask(n, edge_set), n), edge_set

    current_score = _score_from_mask(_build_mask(n, edge_set), n)
    best_score = current_score
    best_es = set(edge_set)
    es = set(edge_set)

    all_possible = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Sort non-edges by spectral proxy improvement potential (heuristic)
    while time.time() - start < budget:
        current_edges = list(es)
        non_edges = [e for e in all_possible if e not in es]

        # Compute spectral proxy for current graph
        spec_cur = _spectral_proxy(n, es)

        best_swap_score = current_score
        best_swap = None
        best_swap_spec = spec_cur

        for e_rm in current_edges:
            u, v = e_rm
            es.discard(e_rm)

            for e_add in non_edges:
                a, b = e_add
                es.add(e_add)

                score = _score_from_mask(_build_mask(n, es), n)

                es.discard(e_add)

                if score < best_swap_score:
                    best_swap_score = score
                    best_swap = (e_rm, e_add)
                    best_swap_spec = _spectral_proxy(n, es | {e_add})

            es.add(e_rm)

        if best_swap:
            e_rm, e_add = best_swap
            es.discard(e_rm)
            es.add(e_add)
            current_score = best_swap_score
            if current_score < best_score:
                best_score = current_score
                best_es = set(es)
        else:
            break

    return best_score, best_es

def construct():
    start_time = time.time()
    n = 16
    target_edges = 65

    # Start with K_{6,10} + 5 matching edges in B
    # Part A: 0-5 (size 6), Part B: 6-15 (size 10)
    edge_set = set()
    for u in range(6):
        for v in range(6, 16):
            edge_set.add((u, v))
    for a, b in [(6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]:
        edge_set.add((a, b))

    best_score = _score_from_mask(_build_mask(n, edge_set), n)
    best_edge_set = set(edge_set)

    random.seed(42)

    # Main search loop: multiple trials with perturbation
    trial = 0
    while time.time() < _DEADLINE_GUARD - 20.0:
        trial += 1

        if trial == 1:
            es_cur = set(edge_set)
        else:
            # Perturb the best found so far
            k = random.randint(5, 15)
            es_cur = _perturb(best_edge_set, n, k)

        # Run local search with remaining time budget
        remaining = _DEADLINE_GUARD - 20.0 - time.time()
        if remaining <= 0:
            break
        score, es_new = _local_search(es_cur, n, max_time=remaining)

        if score < best_score:
            best_score = score
            best_edge_set = es_new

        if best_score <= 23:
            break

    # Final high-precision polish
    if time.time() < _DEADLINE_GUARD - 5.0:
        polish_score, polish_es = _polish(best_edge_set, n)
        if polish_score < best_score:
            best_score = polish_score
            best_edge_set = polish_es

    # Final verification
    adj_final_mask = _build_mask(n, best_edge_set)
    final_score = _score_from_mask(adj_final_mask, n)
    if final_score != best_score:
        best_score = final_score

    # Ensure exactly 65 edges
    if len(best_edge_set) != target_edges:
        # Fallback: rebuild from K_{6,10}+5
        edge_set = set()
        for u in range(6):
            for v in range(6, 16):
                edge_set.add((u, v))
        for a, b in [(6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]:
            edge_set.add((a, b))
        best_edge_set = edge_set
        best_score = _score_from_mask(_build_mask(n, best_edge_set), n)

    edges = [[u, v] for u, v in sorted(best_edge_set)]

    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
