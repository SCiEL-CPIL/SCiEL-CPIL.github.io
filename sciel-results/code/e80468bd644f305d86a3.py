"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
from itertools import combinations
from collections import deque
import random as _random

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    deadline = _time.time() + 55

    # Build adjacency list as sorted lists for deterministic iteration
    adj_list = [sorted(adj[i]) for i in range(n)]

    def find_path_bfs(src, dst, forbidden):
        """Find any path from src to dst avoiding forbidden vertices (except src/dst)."""
        if src == dst:
            return [src]
        visited = {src}
        parent = {src: None}
        q = deque([src])
        while q:
            u = q.popleft()
            if u == dst:
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                return list(reversed(path))
            for v in adj_list[u]:
                if v not in visited and (v == dst or v not in forbidden):
                    visited.add(v)
                    parent[v] = u
                    q.append(v)
        return None

    def check_feasibility(branches):
        """
        Check if a set of branches has valid internally-disjoint paths between every pair.
        Uses backtracking with multiple path-finding strategies.
        Returns dict {(u,v): path} or None.
        """
        bset = frozenset(branches)
        bsorted = sorted(branches)
        pairs = list(combinations(bsorted, 2))
        k = len(bsorted)
        if k < 2:
            return {}

        # Precompute: for each pair, find all simple paths up to a reasonable length
        # This is expensive, so we limit path length
        max_path_len = 6  # max number of edges in a path

        # Use backtracking: assign paths one by one
        # Order pairs by difficulty (fewest available paths first)
        
        def get_available_paths(u, v, used_interiors):
            """Find all simple paths from u to v with length <= max_path_len, avoiding used_interiors and other branches."""
            forbidden = bset - {u, v}
            forbidden = forbidden | used_interiors
            paths = []
            # DFS with depth limit
            def dfs(current, path, visited):
                if len(path) - 1 > max_path_len:
                    return
                if current == v:
                    if len(path) >= 2:
                        paths.append(list(path))
                    return
                for nxt in adj_list[current]:
                    if nxt not in visited and (nxt == v or nxt not in forbidden):
                        visited.add(nxt)
                        path.append(nxt)
                        dfs(nxt, path, visited)
                        path.pop()
                        visited.discard(nxt)
            dfs(u, [u], {u})
            return paths

        # Backtracking search
        best_result = None

        def backtrack(idx, assigned_paths, used_interiors):
            nonlocal best_result
            if _time.time() > deadline:
                return False
            if idx == len(pairs):
                # All pairs assigned
                result = {}
                for i, (u, v) in enumerate(pairs):
                    result[(u, v)] = assigned_paths[i]
                best_result = result
                return True
            u, v = pairs[idx]
            # Try to find a path for this pair
            # First try shortest path
            shortest = find_path_bfs(u, v, used_interiors | (bset - {u, v}))
            candidates = []
            if shortest is not None:
                candidates.append(shortest)
            # Also try a few more paths
            if len(candidates) < 3:
                extra = get_available_paths(u, v, used_interiors)
                for p in extra:
                    if p not in candidates:
                        candidates.append(p)
                    if len(candidates) >= 5:
                        break
            for path in candidates:
                interior = frozenset(path[1:-1])
                if interior & bset:
                    continue
                if interior & used_interiors:
                    continue
                new_used = used_interiors | interior
                if backtrack(idx + 1, assigned_paths + [path], new_used):
                    return True
            return False

        # Try multiple pair orderings
        rng = _random.Random(42)
        orderings = [
            list(pairs),  # natural order
            sorted(pairs, key=lambda p: (p[0], p[1])),
        ]
        # Sort by minimum path length (difficulty)
        def pair_difficulty(p):
            u, v = p
            path = find_path_bfs(u, v, bset - {u, v})
            if path is None:
                return float('inf')
            return len(path)
        orderings.append(sorted(pairs, key=pair_difficulty))
        orderings.append(sorted(pairs, key=lambda p: -pair_difficulty(p)))
        # Random orderings
        for _ in range(5):
            shuffled = list(pairs)
            rng.shuffle(shuffled)
            orderings.append(shuffled)

        for ordering in orderings:
            if _time.time() > deadline:
                break
            best_result = None
            # Re-index pairs according to ordering
            pairs_ordered = ordering
            if backtrack(0, [], frozenset()):
                if best_result is not None:
                    # Convert keys to match original pairs
                    return best_result

        return None

    def try_grow(initial_branches):
        """Try to grow a branch set greedily."""
        current = list(initial_branches)
        improved = True
        while improved and _time.time() < deadline:
            improved = False
            candidates = [v for v in range(n) if v not in current]
            # Score by connectivity to existing branches
            def cand_score(v):
                score = 0
                for u in current:
                    if v in adj[u]:
                        score += 2
                # Also consider degree
                score += len(adj[v]) * 0.1
                return score
            candidates.sort(key=cand_score, reverse=True)
            for candidate in candidates:
                new_set = current + [candidate]
                paths = check_feasibility(new_set)
                if paths is not None:
                    current = new_set
                    improved = True
                    break
        return current

    # Find maximum clique
    def bron_kerbosch(R, P_set, X):
        if not P_set and not X:
            return [tuple(sorted(R))]
        cliques = []
        if P_set or X:
            pivot_candidates = list(P_set | X)
            pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P_set))
        else:
            return []
        candidates = [v for v in P_set if v not in adj[pivot]]
        for v in candidates:
            new_R = R | {v}
            new_P = P_set & adj[v]
            new_X = X & adj[v]
            cliques.extend(bron_kerbosch(new_R, new_P, new_X))
            P_set.discard(v)
            X.add(v)
        return cliques

    max_clique = set()
    all_cliques = bron_kerbosch(set(), set(range(n)), set())
    for c in all_cliques:
        if len(c) > len(max_clique):
            max_clique = set(c)

    best_branches = None
    best_paths = None

    def update_best(b, p):
        nonlocal best_branches, best_paths
        if b is not None and p is not None:
            if best_branches is None or len(b) > len(best_branches):
                best_branches = list(b)
                best_paths = p
            elif len(b) == len(best_branches):
                total_new = sum(len(path[1:-1]) for path in p.values())
                total_old = sum(len(path[1:-1]) for path in best_paths.values())
                if total_new < total_old:
                    best_branches = list(b)
                    best_paths = p

    # Strategy 1: Start from maximum clique
    if len(max_clique) >= 2:
        grown = try_grow(list(max_clique))
        paths = check_feasibility(grown)
        update_best(grown, paths)

    # Strategy 2: Start from top-degree vertices
    deg = [len(adj[i]) for i in range(n)]
    top = sorted(range(n), key=lambda x: -deg[x])
    
    for start_size in [3, 4, 5]:
        if _time.time() > deadline:
            break
        grown = try_grow(list(top[:start_size]))
        paths = check_feasibility(grown)
        update_best(grown, paths)

    # Strategy 3: Random starts
    _random.seed(777)
    for _ in range(40):
        if _time.time() > deadline:
            break
        start_size = _random.randint(2, 5)
        start = _random.sample(range(n), start_size)
        grown = try_grow(start)
        paths = check_feasibility(grown)
        update_best(grown, paths)

    # Strategy 4: Try all pairs of high-degree vertices
    if _time.time() < deadline:
        high_deg = [v for v in range(n) if deg[v] >= 4]
        for i in range(len(high_deg)):
            if _time.time() > deadline:
                break
            for j in range(i + 1, len(high_deg)):
                if _time.time() > deadline:
                    break
                seed = [high_deg[i], high_deg[j]]
                grown = try_grow(seed)
                paths = check_feasibility(grown)
                update_best(grown, paths)

    # Fallback
    if best_branches is None or len(best_branches) < 2:
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = {(u, v): [u, v]}

    # Format output
    branches = [int(b) for b in sorted(best_branches)]
    paths_list = []
    for (u, v) in combinations(branches, 2):
        path = best_paths.get((u, v)) or best_paths.get((v, u))
        if path is None:
            path = find_path_bfs(u, v, set(best_branches) - {u, v})
            if path is None:
                path = [u, v]
        paths_list.append([int(x) for x in path])

    return {'branches': branches, 'paths': paths_list}
# EVOLVE-BLOCK-END
