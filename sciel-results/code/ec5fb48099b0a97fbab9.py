# EVOLVE-BLOCK-START
import time as _time
import random
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
    
    deadline = _time.time() + 58
    
    def find_path(src, dst, forbidden):
        """Find shortest path from src to dst avoiding forbidden vertices (except src/dst)."""
        if src == dst:
            return [src]
        if (min(src,dst), max(src,dst)) in edge_set:
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
        
        # Sort pairs: prefer direct edges first, then shortest potential paths
        def pair_key(p):
            u, v = p
            if (min(u,v), max(u,v)) in edge_set:
                return (0, p)
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

    def random_local_search(initial_size, max_iters=50):
        """Local search starting from a random set of given size."""
        # Initialize with random high-degree vertices
        deg = [len(adj[i]) for i in range(n)]
        # Weighted sampling
        weights = [deg[i] + 1 for i in range(n)]
        total_w = sum(weights)
        
        def sample_one(current_set):
            # Pick a vertex not in current_set, weighted by degree
            candidates = [i for i in range(n) if i not in current_set]
            if not candidates:
                return None
            # Simple random pick from candidates, biased by degree
            # To keep it fast, just pick random
            return random.choice(candidates)

        # Start with a random set
        cur = set(random.sample(range(n), initial_size))
        best_cur = None
        best_paths = None
        
        # Check initial
        paths = check_branch_set(list(cur))
        if paths:
            best_cur = list(cur)
            best_paths = paths
            
        for _ in range(max_iters):
            if _time.time() > deadline:
                break
            improved = False
            
            # Try swap: remove one, add one
            rem = random.choice(list(cur))
            cur.discard(rem)
            
            # Try several candidates to add
            added = False
            for _ in range(10):
                cand = random.randint(0, n-1)
                if cand in cur:
                    continue
                cur.add(cand)
                paths = check_branch_set(list(cur))
                if paths:
                    if best_cur is None or len(cur) > len(best_cur):
                        best_cur = list(cur)
                        best_paths = paths
                    added = True
                    break
                else:
                    cur.discard(cand)
            
            if not added:
                # If we couldn't find a valid swap, maybe we lost a valid one?
                # Revert to previous best state if we are worse? 
                # Actually, if check fails, we just discard cand.
                # If we never added, cur is missing 'rem'. We should put it back?
                # No, we are exploring. If we can't find a valid replacement, 
                # we might want to keep the set size constant or shrink.
                # Let's just retry with a new random swap.
                pass
            
            # Occasionally try to grow if valid
            if len(cur) < n:
                cand = random.randint(0, n-1)
                if cand not in cur:
                    cur.add(cand)
                    paths = check_branch_set(list(cur))
                    if paths:
                        if best_cur is None or len(cur) > len(best_cur):
                            best_cur = list(cur)
                            best_paths = paths
                    else:
                        cur.discard(cand)

        return best_cur, best_paths

    best_branches = None
    best_paths = None
    
    # Try different initial sizes
    # We know clique is max 6-7. Let's try up to 10.
    for size in [10, 9, 8, 7, 6, 5]:
        if _time.time() > deadline - 5:
            break
        for trial in range(50):
            if _time.time() > deadline - 2:
                break
            b, p = random_local_search(size, max_iters=20)
            if b is not None:
                if best_branches is None or len(b) > len(best_branches):
                    best_branches = b
                    best_paths = p
                    # If we found a large one, we can stop early if it's very large
                    if len(b) >= 10:
                        break
    
    # Fallback to previous logic if random search failed
    if best_branches is None:
        # Simple clique based fallback
        def find_max_clique():
            best = []
            def bk(R, P, X):
                nonlocal best
                if not P and not X:
                    if len(R) > len(best):
                        best = list(R)
                    return
                if not P:
                    return
                cand = list(P) + list(X)
                pivot = max(cand, key=lambda u: len(P & adj[u]))
                for v in list(P - adj[pivot]):
                    bk(R | {v}, P & adj[v], X & adj[v])
                    P.remove(v)
                    X.add(v)
            bk(set(), set(range(n)), set())
            return best
        
        clique = find_max_clique()
        if len(clique) >= 2:
            # Local search from clique
            cur = list(clique)
            for _ in range(50):
                if _time.time() > deadline:
                    break
                improved = False
                for cand in range(n):
                    if cand in cur:
                        continue
                    new_set = cur + [cand]
                    paths = check_branch_set(new_set)
                    if paths is not None:
                        cur = new_set
                        improved = True
                        break
                if not improved:
                    break
            best_branches = cur
            best_paths = check_branch_set(cur)
            if best_paths is None:
                 # Should not happen if we started with valid clique
                 best_branches = clique[:2]
                 best_paths = check_branch_set(best_branches)

    if best_branches is None or len(best_branches) < 2:
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = {frozenset((u,v)): [u, v]}
        
    branches = [int(b) for b in sorted(best_branches)]
    paths_list = []
    for u, v in combinations(branches, 2):
        path = best_paths.get(frozenset((u,v)))
        if path is None:
            path = [u, v]
        paths_list.append([int(x) for x in path])
    
    return {'branches': branches, 'paths': paths_list}
# EVOLVE-BLOCK-END
