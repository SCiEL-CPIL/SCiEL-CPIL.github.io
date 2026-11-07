# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

N = P['n']
DEADLINE = 85.0  # safety margin before 90s kill

def _build_adj():
    adj = [set() for _ in range(N)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bfs_path(src, dst, adj, blocked):
    """Find shortest path from src to dst avoiding blocked vertices (except src/dst)."""
    if src == dst:
        return [src]
    if src in blocked or dst in blocked:
        return None
    parent = {src: None}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v in parent:
                continue
            if v != dst and v in blocked:
                continue
            parent[v] = u
            if v == dst:
                # reconstruct
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            q.append(v)
    return None

def _try_subdivision(branches, adj, rng):
    """Try to find vertex-disjoint paths for all pairs in branches.
    Returns list of paths or None."""
    n = len(branches)
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    # Randomize pair order
    order = list(range(len(pairs)))
    rng.shuffle(order)
    blocked = set()
    paths = [None] * len(pairs)
    for idx in order:
        bi, bj = pairs[idx]
        src, dst = branches[bi], branches[bj]
        # Temporarily unblock src and dst for this BFS
        path = _bfs_path(src, dst, adj, blocked)
        if path is None:
            return None
        # Block interior vertices
        for v in path[1:-1]:
            if v in branches:
                return None  # interior hits a branch
            if v in blocked:
                return None
            blocked.add(v)
        paths[idx] = path
    return paths

def _make_result(branches, paths):
    """Convert to the required return format."""
    path_list = []
    for p in paths:
        path_list.append([int(v) for v in p])
    return {'branches': [int(v) for v in branches], 'paths': path_list}

def _score(branches):
    return len(branches)

def construct():
    start = time.time()
    rng = random.Random(42)
    adj = _build_adj()
    
    # Degree ordering for bias
    deg = [len(adj[i]) for i in range(N)]
    vertices_by_deg = sorted(range(N), key=lambda v: -deg[v])
    
    best_branches = None
    best_paths = None
    best_k = 0
    
    def update_best(br, pa):
        nonlocal best_branches, best_paths, best_k
        if len(br) > best_k:
            best_k = len(br)
            best_branches = br
            best_paths = pa
    
    # Phase 1: Try all triples (fast, exhaustive)
    from itertools import combinations
    for tri in combinations(range(N), 3):
        if time.time() > DEADLINE:
            break
        pa = _try_subdivision(list(tri), adj, rng)
        if pa is not None:
            update_best(list(tri), pa)
    
    # Phase 2: Randomized search for larger sets
    # Candidate pool: top-degree vertices
    pool = vertices_by_deg[:16]
    
    for attempt in range(20000):
        if time.time() > DEADLINE:
            break
        
        # Pick a random subset size
        k = rng.randint(3, min(8, N))
        if k > len(pool):
            k = len(pool)
        
        # Sample k vertices from pool (with some randomness)
        if k <= len(pool):
            branches = rng.sample(pool, k)
        else:
            branches = list(range(N))
            rng.shuffle(branches)
            branches = branches[:k]
        
        pa = _try_subdivision(branches, adj, rng)
        if pa is not None:
            update_best(branches, pa)
    
    # Phase 3: Hill-climbing - try to extend or swap
    if best_branches is not None:
        current = list(best_branches)
        current_paths = best_paths
        for step in range(5000):
            if time.time() > DEADLINE:
                break
            improved = False
            
            # Try adding a new branch
            for v in vertices_by_deg:
                if v in current:
                    continue
                new_br = current + [v]
                pa = _try_subdivision(new_br, adj, rng)
                if pa is not None:
                    current = new_br
                    current_paths = pa
                    update_best(current, current_paths)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping a branch
            if len(current) >= 3:
                for i, old_v in enumerate(current):
                    for new_v in vertices_by_deg:
                        if new_v in current:
                            continue
                        new_br = current[:i] + [new_v] + current[i+1:]
                        pa = _try_subdivision(new_br, adj, rng)
                        if pa is not None:
                            current = new_br
                            current_paths = pa
                            update_best(current, current_paths)
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                # Try removing a branch to allow re-adding
                if len(current) > 3:
                    for i in range(len(current)):
                        new_br = current[:i] + current[i+1:]
                        pa = _try_subdivision(new_br, adj, rng)
                        if pa is not None:
                            current = new_br
                            current_paths = pa
                            break
    
    # Phase 4: Final exhaustive check on best found set size
    if best_k >= 3:
        # Try all subsets of size best_k from top vertices
        top = vertices_by_deg[:best_k + 3]
        for subset in combinations(top, best_k):
            if time.time() > DEADLINE:
                break
            pa = _try_subdivision(list(subset), adj, rng)
            if pa is not None:
                update_best(list(subset), pa)
    
    # Fallback
    if best_branches is None:
        u, v = P['edges'][0]
        best_branches = [u, v]
        best_paths = [[u, v]]
    
    return _make_result(best_branches, best_paths)

# EVOLVE-BLOCK-END
