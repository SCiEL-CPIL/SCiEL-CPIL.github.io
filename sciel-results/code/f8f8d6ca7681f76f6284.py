"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def _build_graph():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return n, adj

def _match_length2(non_adj_pairs, B_set, n, adj):
    """Try to assign a distinct internal vertex w to each non-adjacent pair (u,v)
    such that u-w and w-v are edges and w is not a branch.
    Uses Kuhn's bipartite matching (pairs on left, internal vertices on right).
    Returns list of internal vertices (same order as non_adj_pairs) or None."""
    if not non_adj_pairs:
        return []
    # For each pair, list of valid internal vertices (not in B_set)
    pair_to_verts = []
    for (u, v) in non_adj_pairs:
        valid = [w for w in (adj[u] & adj[v]) if w not in B_set]
        if not valid:
            return None
        pair_to_verts.append(valid)

    # Kuhn's algorithm: match internal vertex -> pair index
    match_vertex = {}  # vertex -> pair_idx
    def try_kuhn(pi, seen):
        for w in pair_to_verts[pi]:
            if w in seen:
                continue
            seen.add(w)
            if w not in match_vertex or try_kuhn(match_vertex[w], seen):
                match_vertex[w] = pi
                return True
        return False

    # Order pairs by fewest options first (MRV heuristic) for better success
    order = sorted(range(len(non_adj_pairs)), key=lambda i: len(pair_to_verts[i]))
    for pi in order:
        seen = set()
        if not try_kuhn(pi, seen):
            return None

    # Build result in original order
    result = [None] * len(non_adj_pairs)
    for w, pi in match_vertex.items():
        result[pi] = w
    return result

def _try_subdivision(B, n, adj):
    """Try to find a subdivision for branch set B using length-2 (single-internal)
    paths for non-adjacent pairs and direct edges for adjacent pairs.
    Returns list of paths (one per unordered pair) or None."""
    B_set = set(B)
    k = len(B)
    non_adj_pairs = []
    adj_pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            u, v = B[i], B[j]
            if v in adj[u]:
                adj_pairs.append((u, v))
            else:
                non_adj_pairs.append((u, v))

    # Quick feasibility: need at least one distinct internal per non-adj pair
    # (matching will check this properly, but a cheap bound helps)
    if len(non_adj_pairs) > n - k:
        return None

    match = _match_length2(non_adj_pairs, B_set, n, adj)
    if match is None:
        return None

    paths = []
    for u, v in adj_pairs:
        paths.append([u, v])
    for idx, (u, v) in enumerate(non_adj_pairs):
        w = match[idx]
        paths.append([u, w, v])
    return paths

def _bron_kerbosch(n, adj):
    """Find maximum clique via Bron-Kerbosch with pivot."""
    best = []
    def bk(R, P_set, X):
        nonlocal best
        if not P_set and not X:
            if len(R) > len(best):
                best = list(R)
            return
        if not P_set:
            return
        pivot_candidates = P_set | X
        if pivot_candidates:
            pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P_set))
            candidates = P_set - adj[pivot]
        else:
            candidates = P_set
        for v in list(candidates):
            bk(R | {v}, P_set & adj[v], X & adj[v])
            P_set.discard(v)
            X.add(v)
    bk(set(), set(range(n)), set())
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    n, adj = _build_graph()
    degrees = [len(adj[i]) for i in range(n)]

    # Fallback valid result (2 adjacent branches)
    fallback = {'branches': [int(P['edges'][0][0]), int(P['edges'][0][1])],
                'paths': [[int(P['edges'][0][0]), int(P['edges'][0][1])]]}

    best_branches = None
    best_paths = None

    def update(B):
        nonlocal best_branches, best_paths
        paths = _try_subdivision(sorted(B), n, adj)
        if paths is not None:
            if best_branches is None or len(B) > len(best_branches):
                best_branches = sorted(B)
                best_paths = paths
            return True
        return False

    # Phase 1: start from max clique
    max_clique = _bron_kerbosch(n, adj)
    if len(max_clique) >= 2:
        update(max_clique)

    if best_branches is None:
        return fallback

    # Phase 2: greedy growth — add vertices one at a time
    current = set(best_branches)
    sorted_vertices = sorted(range(n), key=lambda x: -degrees[x])
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for extra in sorted_vertices:
            if time.time() > deadline:
                break
            if extra in current:
                continue
            candidate = current | {extra}
            if update(candidate):
                current = set(best_branches)
                improved = True
                break

    # Phase 3: random growth from current best
    random.seed(123)
    for _ in range(500):
        if time.time() > deadline:
            break
        current = set(best_branches)
        improved = False
        for extra in random.sample(range(n), min(n, 12)):
            if extra in current:
                continue
            candidate = current | {extra}
            if update(candidate):
                current = set(best_branches)
                improved = True
                break
        if not improved:
            break

    # Phase 4: random subsets of increasing size to find larger solutions
    current_k = len(best_branches)
    for size in range(current_k + 1, min(n + 1, current_k + 6)):
        if time.time() > deadline:
            break
        found = False
        for attempt in range(3000):
            if time.time() > deadline:
                break
            candidates = set(random.sample(range(n), size))
            if update(candidates):
                current_k = size
                found = True
                break
        if not found:
            continue

    # Phase 5: final growth push
    for _ in range(300):
        if time.time() > deadline:
            break
        current = set(best_branches)
        improved = False
        for extra in random.sample(range(n), min(n, 15)):
            if extra in current:
                continue
            candidate = current | {extra}
            if update(candidate):
                current = set(best_branches)
                improved = True
                break
        if not improved:
            break

    if best_branches is None:
        return fallback

    branches_out = [int(b) for b in best_branches]
    paths_out = [[int(x) for x in p] for p in best_paths]
    return {'branches': branches_out, 'paths': paths_out}
# EVOLVE-BLOCK-END
