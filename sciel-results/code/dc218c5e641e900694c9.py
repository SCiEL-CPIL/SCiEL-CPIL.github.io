"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
from itertools import combinations, permutations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    deadline = _time.time() + 55  # safety margin
    
    def find_shortest_path(src, dst, forbidden):
        """Find shortest path from src to dst avoiding forbidden vertices (except src/dst)."""
        if src == dst:
            return [src]
        from collections import deque
        visited = {src}
        parent = {src: None}
        q = deque([src])
        while q:
            u = q.popleft()
            if u == dst:
                # Reconstruct path
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                return list(reversed(path))
            for v in adj[u]:
                if v not in visited and (v == dst or v not in forbidden):
                    visited.add(v)
                    parent[v] = u
                    q.append(v)
        return None
    
    def check_branch_set(branches):
        """Check if a set of branches has valid disjoint paths. Returns paths or None."""
        bset = set(branches)
        pairs = list(combinations(branches, 2))
        # We need to assign disjoint interior vertices to each pair
        # Greedy: process pairs in some order, find shortest path avoiding used interiors
        used_interiors = set()
        paths = {}
        
        # Sort pairs by shortest path length (shorter first to leave more room)
        def pair_key(p):
            path = find_shortest_path(p[0], p[1], bset - {p[0], p[1]})
            if path is None:
                return (float('inf'), p)
            return (len(path), p)
        
        sorted_pairs = sorted(pairs, key=pair_key)
        
        for u, v in sorted_pairs:
            # Find path from u to v avoiding bset (except u,v) and used_interiors
            forbidden = (bset - {u, v}) | used_interiors
            path = find_shortest_path(u, v, forbidden)
            if path is None:
                return None
            # Check interior doesn't include any branch
            interior = set(path[1:-1])
            if interior & bset:
                return None
            if interior & used_interiors:
                return None
            used_interiors |= interior
            paths[(u, v)] = path
        
        return paths
    
    def try_grow(initial_branches):
        """Try to grow a branch set greedily."""
        current = list(initial_branches)
        improved = True
        while improved and _time.time() < deadline:
            improved = False
            for candidate in range(n):
                if candidate in current:
                    continue
                new_set = current + [candidate]
                paths = check_branch_set(new_set)
                if paths is not None:
                    current = new_set
                    improved = True
                    break
        return current
    
    # Find a good initial set: start with a clique or high-degree vertices
    # Proper maximum clique via Bron-Kerbosch with pivot (deterministic)
    def find_max_clique():
        best = []
        def bk(R, P, X):
            nonlocal best
            if not P and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            # pivot: choose u in P|X with max |P ∩ adj[u]|
            cand = list(P) + list(X)
            if not cand:
                return
            pivot = max(cand, key=lambda u: len(P & adj[u]))
            # iterate over P \ adj(pivot); if pivot not in P, iterate all of P
            for v in list(P):
                if v in adj[pivot]:
                    continue
                P.remove(v)
                bk(R | {v}, P & adj[v], X & adj[v])
                X.add(v)
        bk(set(), set(range(n)), set())
        return best

    def local_search(branches):
        """Local search: try adding a vertex; if stuck, try 1-swap (remove one, add another)."""
        cur = list(branches)
        improved = True
        while improved and _time.time() < deadline:
            improved = False
            # try add
            for cand in range(n):
                if cand in cur:
                    continue
                new_set = cur + [cand]
                paths = check_branch_set(new_set)
                if paths is not None:
                    cur = new_set
                    improved = True
                    break
            if improved:
                continue
            # try swap: remove one, add one (net +0 but may enable further growth)
            for rem in list(cur):
                if _time.time() > deadline:
                    break
                base = [x for x in cur if x != rem]
                for cand in range(n):
                    if cand in base:
                        continue
                    new_set = base + [cand]
                    paths = check_branch_set(new_set)
                    if paths is not None:
                        # only accept if it actually leads to growth later; accept to escape
                        cur = new_set
                        improved = True
                        break
                if improved:
                    break
        return cur

    # Try multiple starting points
    best_branches = None
    best_paths = None

    def consider(result):
        nonlocal best_branches, best_paths
        if result is None or len(result) < 2:
            return
        paths = check_branch_set(result)
        if paths is None:
            return
        if best_branches is None or len(result) > len(best_branches):
            best_branches = list(result)
            best_paths = paths

    # Start with a maximum clique
    clique = find_max_clique()
    if len(clique) >= 2:
        consider(local_search(try_grow(clique)))

    # Also try starting from top-degree vertices
    deg = [len(adj[i]) for i in range(n)]
    top = sorted(range(n), key=lambda x: -deg[x])[:6]
    consider(local_search(try_grow(list(top))))

    # Try random-ish starts
    import random
    random.seed(42)
    for _ in range(50):
        if _time.time() > deadline:
            break
        start = random.sample(range(n), min(3, n))
        consider(local_search(try_grow(start)))
    
    # Fallback: at least 2 branches
    if best_branches is None or len(best_branches) < 2:
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = {(u, v): [u, v]}

    # Final polish: try to grow the best set a bit more, and re-validate
    polished = local_search(list(best_branches))
    ppaths = check_branch_set(polished)
    if ppaths is not None and len(polished) >= len(best_branches):
        best_branches = polished
        best_paths = ppaths

    # Format output
    branches = [int(b) for b in best_branches]
    paths_list = []
    for (u, v) in combinations(branches, 2):
        path = best_paths.get((u, v)) or best_paths.get((v, u))
        if path is None:
            # Should not happen
            path = [u, v]
        paths_list.append([int(x) for x in path])

    return {'branches': branches, 'paths': paths_list}
# EVOLVE-BLOCK-END
