import random
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

_N = P['n']
_ADJ = [0] * _N
for _e in P['edges']:
    _ADJ[_e[0]] |= 1 << _e[1]
    _ADJ[_e[1]] |= 1 << _e[0]

def _bfs_path(src, dst, blocked):
    """BFS shortest path from src to dst avoiding blocked interior vertices."""
    if src == dst:
        return [src]
    if (1 << dst) & _ADJ[src]:
        return [src, dst]
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
            path.reverse()
            return path
        for nb in range(_N):
            if (1 << nb) & _ADJ[u] and nb not in parent:
                if nb != dst and nb in blocked:
                    continue
                parent[nb] = u
                q.append(nb)
    return None

def _route_paths_random(branches, max_attempts=50):
    """
    Try to route internally disjoint paths for all pairs in branches.
    Uses randomized ordering of pairs and backtracking.
    """
    pairs = [(i, j) for i in range(len(branches)) for j in range(i + 1, len(branches))]
    if not pairs:
        return []
    
    def solve(idx, used_interiors, paths):
        if idx == len(pairs):
            return paths
        
        i, j = pairs[idx]
        a, b = branches[i], branches[j]
        blocked = set(used_interiors)
        
        # 1. Try standard BFS
        path = _bfs_path(a, b, blocked)
        if path is not None:
            interiors = set(path[1:-1])
            if not (interiors & used_interiors):
                new_used = used_interiors | interiors
                new_paths = paths + [path]
                res = solve(idx + 1, new_used, new_paths)
                if res is not None:
                    return res
        
        # 2. Try alternative paths by blocking interior vertices
        if path is not None and len(path) > 2:
            for alt_block in list(set(path[1:-1])):
                blocked2 = set(used_interiors) | {alt_block}
                path2 = _bfs_path(a, b, blocked2)
                if path2 is not None:
                    interiors2 = set(path2[1:-1])
                    if not (interiors2 & used_interiors):
                        new_used2 = used_interiors | interiors2
                        new_paths2 = paths + [path2]
                        res2 = solve(idx + 1, new_used2, new_paths2)
                        if res2 is not None:
                            return res2
        
        return None

    for _ in range(max_attempts):
        random.shuffle(pairs)
        res = solve(0, set(), [])
        if res is not None:
            return res
    return None

def _random_maximal_clique():
    """Generate a random maximal clique using a greedy randomized approach."""
    vertices = list(range(_N))
    random.shuffle(vertices)
    clique = []
    available = set(vertices)
    
    for v in vertices:
        if v not in available:
            continue
        # Check if v is adjacent to all in clique
        if all((1 << c) & _ADJ[v] for c in clique):
            clique.append(v)
            # Update available: must be neighbor of v and in available
            new_available = set()
            for u in available:
                if (1 << u) & _ADJ[v]:
                    new_available.add(u)
            available = new_available
            available.discard(v)
            
    return clique

def construct():
    best_branches = None
    best_paths = None
    best_size = 0
    
    start_time = time.time()
    time_limit = 50.0 # Leave 10s buffer
    
    cliques = set()
    # Generate diverse cliques
    for i in range(1000):
        if time.time() - start_time > time_limit * 0.4:
            break
        c = _random_maximal_clique()
        cliques.add(tuple(sorted(c)))
        
    def clique_score(c):
        size = len(c)
        ext_conn = 0
        c_set = set(c)
        for v in c:
            for nb in range(_N):
                if (1 << nb) & _ADJ[v] and nb not in c_set:
                    ext_conn += 1
        return (size, ext_conn)
    
    sorted_cliques = sorted(cliques, key=clique_score, reverse=True)
    
    for c_tuple in sorted_cliques:
        if time.time() - start_time > time_limit:
            break
            
        branches = list(c_tuple)
        if len(branches) <= best_size:
            continue
            
        paths = _route_paths_random(branches, max_attempts=20)
        if paths is not None:
            best_branches = branches
            best_paths = paths
            best_size = len(branches)
            
    if best_branches is None:
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
            
    return {
        'branches': [int(b) for b in best_branches],
        'paths': [[int(x) for x in p] for p in best_paths]
    }
