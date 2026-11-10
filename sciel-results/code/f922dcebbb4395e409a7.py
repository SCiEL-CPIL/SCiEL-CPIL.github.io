"""Rewritten program: maximize branch vertices with clique heuristic + greedy expansion."""
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    n = int(P['n'])
    edge_set = set()
    adj = [set() for _ in range(n)]
    for e in P['edges']:
        u, v = int(e[0]), int(e[1])
        edge_set.add((u, v))
        edge_set.add((v, u))
        adj[u].add(v)
        adj[v].add(u)

    def are_adjacent(a, b):
        return (a, b) in edge_set

    def find_shortest_path(s, t, forbidden):
        # BFS avoiding forbidden vertices; returns path list or None
        if s in forbidden:
            return None
        if t in forbidden:
            return None
        if s == t:
            return [s]
        parent = {s: None}
        q = deque([s])
        while q:
            u = q.popleft()
            if u == t:
                break
            for w in adj[u]:
                if w in forbidden:
                    continue
                if w not in parent:
                    parent[w] = u
                    q.append(w)
        if t not in parent:
            return None
        path = []
        cur = t
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path

    def greedy_expand(initial_branches):
        # Try to add vertices one at a time, checking feasibility
        branches = list(initial_branches)
        # Sort candidates by some heuristic (degree to existing branches)
        while True:
            if time.time() > DEADLINE:
                break
            # Find the best candidate to add
            candidates = [v for v in range(n) if v not in branches]
            if not candidates:
                break
            best = None
            best_paths = None
            best_key = None
            for cand in candidates:
                trial = branches + [cand]
                m = len(trial)
                # Try to find disjoint paths for all pairs
                paths = {}
                used_interior = set()
                feasible = True
                # Order pairs: those involving cand first (most constrained)
                pair_list = list(combinations(range(m), 2))
                # Pairs involving cand first
                pair_list.sort(key=lambda p: 0 if (p[0] == m-1 or p[1] == m-1) else 1)
                for (i, j) in pair_list:
                    s, t = trial[i], trial[j]
                    forbidden = used_interior | set(trial)
                    p = find_shortest_path(s, t, forbidden)
                    if p is None:
                        feasible = False
                        break
                    # Mark interior as used
                    for v in p[1:-1]:
                        if v in used_interior:
                            feasible = False
                            break
                        used_interior.add(v)
                    if not feasible:
                        break
                    paths[(i, j)] = p
                if feasible:
                    # Score: prefer candidates that maintain feasibility
                    # Use path length as tiebreaker (shorter is better)
                    total_len = sum(len(paths[k]) for k in paths)
                    key = (total_len,)
                    if best_key is None or key < best_key:
                        best = cand
                        best_paths = paths
                        best_key = key
            if best is None:
                break
            branches.append(best)
        return branches, best_paths if best_paths is not None else None

    # Build a good initial clique using greedy coloring (Tomita-style)
    def max_clique_greedy():
        # Greedy: repeatedly add vertex with max degree to current clique
        # with backtracking for small graphs
        best_clique = []
        # Try all vertices as starting point
        vertices = list(range(n))
        # Sort by degree descending
        vertices.sort(key=lambda v: -len(adj[v]))
        
        def expand_clique(clique, candidates, deadline):
            nonlocal best_clique
            if time.time() > deadline:
                return
            if len(clique) > len(best_clique):
                best_clique = list(clique)
            if not candidates:
                return
            # Color the candidates for pruning
            # Simple greedy coloring
            color_map = {}
            uncolored = list(candidates)
            colors = 0
            while uncolored:
                colors += 1
                chosen = []
                remaining = []
                for v in uncolored:
                    if all(v not in adj[u] for u in chosen):
                        chosen.append(v)
                    else:
                        remaining.append(v)
                for v in chosen:
                    color_map[v] = colors
                uncolored = remaining
            # Branch on vertices in reverse color order
            for v in reversed(uncolored):
                if time.time() > deadline:
                    return
                # Pruning: if clique size + color of v <= best, skip
                if len(clique) + color_map.get(v, colors) <= len(best_clique):
                    break
                new_clique = clique + [v]
                new_candidates = [u for u in candidates if u in adj[v]]
                expand_clique(new_clique, new_candidates, deadline)
        
        # Try from each vertex
        for v in vertices:
            if time.time() > DEADLINE:
                break
            expand_clique([v], [u for u in vertices if u in adj[v] and u > v], DEADLINE)
        
        return best_clique

    # Get initial clique
    init_clique = max_clique_greedy()
    if len(init_clique) < 2:
        # Fallback: use an edge
        u, v = int(P['edges'][0][0]), int(P['edges'][0][1])
        init_clique = [u, v]

    # Expand from the clique
    expanded, paths = greedy_expand(init_clique)

    # Try multiple initial seeds for diversity
    best_branches = expanded
    best_paths = paths

    # Try greedy expansion from different starting points
    if time.time() < DEADLINE - 5:
        # Try starting from each high-degree vertex
        for start in range(n):
            if time.time() > DEADLINE:
                break
            init = [start]
            # Add neighbors greedily
            neighbors = sorted(adj[start], key=lambda x: -len(adj[x]))
            for nb in neighbors:
                if time.time() > DEADLINE:
                    break
                trial = init + [nb]
                # Quick check: just expand
                exp, _ = greedy_expand(trial)
                if len(exp) > len(best_branches):
                    best_branches = exp
                    best_paths = None

    # Final: compute paths for the best branch set
    if best_paths is None or len(best_paths) != len(list(combinations(range(len(best_branches)), 2))):
        m = len(best_branches)
        paths = {}
        used_interior = set()
        pair_list = list(combinations(range(m), 2))
        # Try multiple orderings if needed
        for ordering in [pair_list, list(reversed(pair_list)),
                         sorted(pair_list, key=lambda p: (best_branches[p[0]], best_branches[p[1]]))]:
            paths = {}
            used_interior = set()
            feasible = True
            for (i, j) in ordering:
                s, t = best_branches[i], best_branches[j]
                forbidden = used_interior | set(best_branches)
                p = find_shortest_path(s, t, forbidden)
                if p is None:
                    feasible = False
                    break
                for v in p[1:-1]:
                    if v in used_interior:
                        feasible = False
                        break
                    used_interior.add(v)
                if not feasible:
                    break
                paths[(i, j)] = p
            if feasible:
                break

    # Build result
    branches = [int(b) for b in best_branches]
    result_paths = []
    for (i, j) in combinations(range(len(branches)), 2):
        result_paths.append([int(x) for x in paths[(i, j)]])

    return {'branches': branches, 'paths': result_paths}
