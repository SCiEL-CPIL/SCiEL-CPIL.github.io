"""Rewritten program: structural change to a CSP / max-flow based disjoint-path solver
with smarter candidate generation (clique + near-clique + greedy extension)."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import deque
from scipy.sparse.csgraph import maximum_bipartite_matching

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 70

    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Precompute all-pairs shortest path lengths
    dist = [[-1] * n for _ in range(n)]
    for src in range(n):
        dist[src][src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[src][v] == -1:
                    dist[src][v] = dist[src][u] + 1
                    q.append(v)

    def find_max_clique():
        best = []
        def bronk(R, P_set, X):
            nonlocal best
            if not P_set and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            if not P_set:
                return
            pivot_candidates = list(P_set | X)
            if pivot_candidates:
                pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P_set))
                candidates = list(P_set - adj[pivot])
            else:
                candidates = list(P_set)
            for v in candidates:
                bronk(R | {v}, P_set & adj[v], X & adj[v])
                P_set = P_set - {v}
                X = X | {v}
        bronk(set(), set(range(n)), set())
        return best

    max_clique = find_max_clique()

    # ---- Core solver: find internally-disjoint paths for all pairs of branches ----
    # We use a max-flow formulation:
    # For a given set of branches, we need |B| choose 2 paths, each between a pair of branches,
    # with all internal vertices disjoint.
    #
    # We model this as a flow network:
    #   - Split each non-branch vertex v into v_in -> v_out with capacity 1.
    #   - Branch vertices have infinite capacity (they can be used by multiple paths as endpoints).
    #   - For each edge (u,v) in the graph, add directed edges u_out -> v_in and v_out -> u_in
    #     with capacity 1 (or infinity if either endpoint is a branch, to allow multiple paths
    #     to share edges incident to branches).
    #   - For each pair (u,v) of branches, we need 1 unit of flow from u to v.
    #   - Total flow needed = C(k,2) where k = |branches|.
    #
    # Actually, a simpler approach: since we need paths between specific pairs, we can use
    # a multi-commodity flow, but that's NP-hard. Instead, we'll use a greedy + backtracking
    # approach with a smarter ordering, combined with a matching-based feasibility check.
    #
    # Better approach: use the fact that for small k (up to ~8), we can enumerate path
    # assignments. But the key insight is: we can use a single max-flow if we treat each
    # pair's path as needing 1 unit, and the constraint is vertex-disjointness for interiors.
    #
    # Let's use a different structural approach:
    # 1. For each pair of branches (u,v), precompute a small set of candidate shortest paths
    #    (e.g., all shortest paths, or top-k shortest paths).
    # 2. Then solve a set-packing / matching problem: select one path per pair such that
    #    no internal vertex is used by more than one path.
    #
    # For step 2, we can use a bipartite matching or a simple backtracking since k is small.

    def get_shortest_paths(u, v, forbidden, max_paths=20):
        """Get up to max_paths shortest paths from u to v avoiding forbidden internal vertices."""
        if u == v:
            return []
        # BFS to find shortest distance
        target_dist = -1
        parent = {u: None}
        q = deque([u])
        visited = {u}
        while q:
            curr = q.popleft()
            if curr == v:
                target_dist = 0
                break
            for nb in adj[curr]:
                if nb not in visited and (nb == v or nb not in forbidden):
                    visited.add(nb)
                    parent[nb] = curr
                    q.append(nb)
        if v not in parent:
            return []

        # Reconstruct one shortest path
        def reconstruct(start, end):
            path = []
            node = end
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path

        # Find all shortest paths via DFS with depth limit
        d = 0
        node = v
        while node != u:
            d += 1
            node = parent[node]

        paths = []
        def dfs_paths(curr, target, path_so_far, depth):
            if len(paths) >= max_paths:
                return
            if curr == target:
                paths.append(list(path_so_far))
                return
            if depth >= d:
                return
            for nb in sorted(adj[curr]):
                if nb == target or nb not in forbidden:
                    if len(path_so_far) + 1 <= d + 1:
                        path_so_far.append(nb)
                        dfs_paths(nb, target, path_so_far, depth + 1)
                        path_so_far.pop()

        path_so_far = [u]
        dfs_paths(u, v, path_so_far, 0)

        if not paths:
            # Fall back to the BFS path
            p = reconstruct(u, v)
            paths = [p]
        return paths

    def try_subdivision(branches):
        """Try to find internally-disjoint paths for all pairs of branches.
        Returns dict of (u,v) -> path, or None if impossible."""
        k = len(branches)
        if k < 2:
            return None
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        m = len(pairs)

        # Check all pairs are connected
        for u, v in pairs:
            if dist[u][v] == -1:
                return None

        # Order pairs by difficulty (longest shortest path first, then fewer options)
        pair_info = []
        for u, v in pairs:
            d = dist[u][v]
            pair_info.append((d, u, v))
        pair_info.sort(key=lambda x: -x[0])

        # For each pair, get candidate paths (shortest paths avoiding other branches as internals)
        candidate_paths = {}
        for d, u, v in pair_info:
            forbidden = branch_set - {u, v}
            paths = get_shortest_paths(u, v, forbidden, max_paths=15)
            if not paths:
                return None
            candidate_paths[(u, v)] = paths

        # Backtracking: assign one path per pair, ensuring internal vertex disjointness
        paths_assigned = {}
        used_int = set()

        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == m:
                return True
            d, u, v = pair_info[idx]
            key = (u, v)
            for path in candidate_paths[key]:
                interiors = set(path[1:-1])
                if interiors & used_int:
                    continue
                paths_assigned[key] = path
                used_int.update(interiors)
                if backtrack(idx + 1):
                    return True
                del paths_assigned[key]
                used_int.difference_update(interiors)
            return False

        if backtrack(0):
            return paths_assigned
        return None

    # ---- Candidate generation and search ----
    best_result = None
    best_k = 0

    def evaluate(branches):
        nonlocal best_result, best_k
        if len(branches) <= best_k:
            return
        if time.time() > deadline:
            return
        result = try_subdivision(branches)
        if result is not None:
            best_result = (list(branches), result)
            best_k = len(branches)

    # Start with max clique
    if len(max_clique) >= 2:
        evaluate(sorted(max_clique))

    # Greedy extension: try to add vertices one at a time
    if best_result is not None:
        current = set(best_result[0])
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for extra in range(n):
                if extra in current:
                    continue
                candidate = sorted(current | {extra})
                result = try_subdivision(candidate)
                if result is not None:
                    best_result = (candidate, result)
                    best_k = len(candidate)
                    current = set(candidate)
                    improved = True
                    break

    # Try random extensions
    random.seed(42)
    for _ in range(500):
        if time.time() > deadline:
            break
        current = set(best_result[0])
        for extra in random.sample(range(n), min(n, 12)):
            if extra in current:
                continue
            candidate = sorted(current | {extra})
            result = try_subdivision(candidate)
            if result is not None:
                best_result = (candidate, result)
                best_k = len(candidate)
                break

    # Try random subsets of increasing size
    for size in range(best_k + 1, min(n + 1, best_k + 10)):
        if time.time() > deadline:
            break
        found = False
        for attempt in range(3000):
            if time.time() > deadline:
                break
            candidates = random.sample(range(n), size)
            result = try_subdivision(candidates)
            if result is not None:
                best_result = (sorted(candidates), result)
                best_k = size
                found = True
                break
        if not found:
            continue

    # Final greedy extension
    if best_result is not None:
        current = set(best_result[0])
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for extra in range(n):
                if extra in current:
                    continue
                candidate = sorted(current | {extra})
                result = try_subdivision(candidate)
                if result is not None:
                    best_result = (candidate, result)
                    best_k = len(candidate)
                    current = set(candidate)
                    improved = True
                    break

    # Safety: if we still have no result, use a minimal valid one
    if best_result is None:
        # Find any edge
        for u, v in edges:
            best_result = ([u, v], {(u, v): [u, v]})
            break
        if best_result is None:
            best_result = ([0, 1], {(0, 1): [0, 1]})

    branches, paths = best_result
    branches_out = [int(b) for b in branches]
    paths_out = []
    for (u, v), path in sorted(paths.items()):
        paths_out.append([int(x) for x in path])

    return {'branches': branches_out, 'paths': paths_out}
# EVOLVE-BLOCK-END
