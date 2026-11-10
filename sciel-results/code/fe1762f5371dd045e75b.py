"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u,v), max(u,v)))
    
    deadline = _time.time() + 58  # safety margin (90s limit, 32s buffer)
    
    def find_path(src, dst, forbidden):
        """Find shortest path from src to dst avoiding forbidden vertices (except src/dst)."""
        if src == dst:
            return [src]
        if (min(src,dst), max(src,dst)) in edge_set:
            # Direct edge - check it's not forbidden
            return [src, dst]
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
            for v in adj[u]:
                if v not in visited and (v == dst or v not in forbidden):
                    visited.add(v)
                    parent[v] = u
                    q.append(v)
        return None
    
    def check_branch_set(branches):
        """Check if a set of branches has valid disjoint paths. Returns paths dict or None."""
        bset = set(branches)
        pairs = list(combinations(sorted(branches), 2))
        used_interiors = set()
        paths = {}
        
        # Sort pairs: prefer direct edges first (shortest paths)
        def pair_key(p):
            u, v = p
            if (min(u,v), max(u,v)) in edge_set:
                return (0, p)
            # Check if there's a path of length 2
            common = adj[u] & adj[v]
            common -= bset
            common -= used_interiors
            if common:
                return (1, p)
            return (2, p)
        
        sorted_pairs = sorted(pairs, key=pair_key)
        
        for u, v in sorted_pairs:
            forbidden = (bset - {u, v}) | used_interiors
            path = find_path(u, v, forbidden)
            if path is None:
                return None
            interior = set(path[1:-1])
            if interior & bset:
                return None
            if interior & used_interiors:
                return None
            used_interiors |= interior
            paths[frozenset((u,v))] = path
        
        return paths
    
    def find_max_clique():
        """Bron-Kerbosch with pivot to find maximum clique."""
        best = []
        def bk(R, P, X):
            nonlocal best
            if not P and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            if not P:
                return
            # pivot
            cand = list(P) + list(X)
            pivot = max(cand, key=lambda u: len(P & adj[u]))
            for v in list(P - adj[pivot]):
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        bk(set(), set(range(n)), set())
        return best
    
    def local_search(branches, max_iters=100):
        """Local search: try adding a vertex; if stuck, try swaps."""
        cur = list(branches)
        improved = True
        iters = 0
        while improved and iters < max_iters and _time.time() < deadline:
            improved = False
            iters += 1
            # Try adding a vertex
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
            # Try 1-swap: remove one, add another
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
                        cur = new_set
                        improved = True
                        break
                if improved:
                    break
        return cur
    
    def consider(result, best):
        """Update best if result is better."""
        if result is None or len(result) < 2:
            return best
        paths = check_branch_set(result)
        if paths is None:
            return best
        if best is None or len(result) > len(best[0]):
            return (result, paths)
        return best
    
    # Find maximum clique
    clique = find_max_clique()
    
    best = None
    
    # Start from maximum clique
    if len(clique) >= 2:
        grown = local_search(clique)
        best = consider(grown, best)
    
    # Try top-degree vertices
    deg = [len(adj[i]) for i in range(n)]
    top = sorted(range(n), key=lambda x: -deg[x])[:8]
    grown = local_search(list(top))
    best = consider(grown, best)
    
    # Try various random starts
    import random
    random.seed(123)
    for trial in range(200):
        if _time.time() > deadline:
            break
        # Random subset of size 3-6
        size = random.randint(3, min(6, n))
        start = random.sample(range(n), size)
        grown = local_search(start)
        best = consider(grown, best)
    
    # Try starting from pairs of high-degree vertices
    top5 = sorted(range(n), key=lambda x: -deg[x])[:5]
    for i in range(len(top5)):
        for j in range(i+1, len(top5)):
            if _time.time() > deadline:
                break
            start = [top5[i], top5[j]]
            grown = local_search(start)
            best = consider(grown, best)
    
    # If we still have time, try more systematic search
    if _time.time() < deadline - 10:
        # Try all pairs as starting points
        for i in range(n):
            for j in range(i+1, n):
                if _time.time() > deadline - 5:
                    break
                start = [i, j]
                grown = local_search(start, max_iters=20)
                best = consider(grown, best)
            if _time.time() > deadline - 5:
                break
    
    # Fallback
    if best is None:
        u, v = edges[0]
        best = ([u, v], {frozenset((u,v)): [u, v]})
    
    branches, paths = best
    branches = [int(b) for b in sorted(branches)]
    
    # Build final paths list
    paths_list = []
    for u, v in combinations(branches, 2):
        path = paths.get(frozenset((u,v)))
        if path is None:
            path = [u, v]
        paths_list.append([int(x) for x in path])
    
    return {'branches': branches, 'paths': paths_list}
# EVOLVE-BLOCK-END
