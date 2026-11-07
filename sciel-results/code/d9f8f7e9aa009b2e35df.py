"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

# Build adjacency structure
N = P['n']
ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)
DEADLINE = time.time() + 80  # safety margin before 90s kill

def _bfs_path(src, dst, forbidden):
    """Find a simple path from src to dst avoiding vertices in 'forbidden'."""
    if src in forbidden or dst in forbidden:
        return None
    if src == dst:
        return [src]
    parent = {src: None}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in ADJ[u]:
            if v in forbidden:
                continue
            if v not in parent:
                parent[v] = u
                if v == dst:
                    # reconstruct
                    path = [v]
                    while path[-1] != src:
                        path.append(parent[path[-1]])
                    path.reverse()
                    return path
                q.append(v)
    return None

def _validate(branches, paths):
    """Check all constraints; return True if valid."""
    if len(branches) < 2:
        return False
    bset = set(branches)
    if len(bset) != len(branches):
        return False
    # Check each path
    for (i, j), path in zip(combinations(branches, 2), paths):
        if len(path) < 2:
            return False
        if path[0] != i or path[-1] != j:
            return False
        # interior vertices must not be branches
        interior = set(path[1:-1])
        if interior & bset:
            return False
        # consecutive adjacency
        for a, b in zip(path, path[1:]):
            if b not in ADJ[a]:
                return False
    # interior vertices pairwise disjoint
    all_interiors = []
    for path in paths:
        all_interiors.extend(path[1:-1])
    if len(set(all_interiors)) != len(all_interiors):
        return False
    return True

def _find_clique_subdivision(branches, max_time=5.0):
    """Given a set of branch vertices, try to find vertex-disjoint paths for all pairs.
    Returns list of paths or None."""
    t0 = time.time()
    pairs = list(combinations(branches, 2))
    n_pairs = len(pairs)
    
    # Sort pairs by difficulty: shortest path length first
    def pair_dist(p):
        # BFS distance ignoring branches as interior
        src, dst = p
        forbidden = set(branches) - {src, dst}
        # BFS
        parent = {src: None}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in ADJ[u]:
                if v in forbidden:
                    continue
                if v not in parent:
                    parent[v] = u
                    if v == dst:
                        d = 0
                        cur = v
                        while cur != src:
                            d += 1
                            cur = parent[cur]
                        return d
                    q.append(v)
        return 10**9
    
    pairs_sorted = sorted(pairs, key=pair_dist)
    
    # Greedy: assign paths one by one, tracking used interior vertices
    used_interior = set()
    result_paths = {}
    
    for p in pairs_sorted:
        if time.time() - t0 > max_time:
            return None
        src, dst = p
        forbidden = used_interior | (set(branches) - {src, dst})
        path = _bfs_path(src, dst, forbidden)
        if path is None:
            return None
        interior = set(path[1:-1])
        used_interior |= interior
        result_paths[p] = path
    
    # Build in original pair order
    paths = [result_paths[p] for p in pairs]
    return paths

def _try_add_branch(current_branches, candidate, max_time=3.0):
    """Try to add 'candidate' to current_branches and find valid paths."""
    new_branches = list(current_branches) + [candidate]
    t0 = time.time()
    paths = _find_clique_subdivision(new_branches, max_time)
    return paths

def construct():
    t_start = time.time()
    
    # Phase 1: Find a good initial clique (set of vertices where every pair has a path)
    # Start with high-degree vertices
    degrees = [len(ADJ[i]) for i in range(N)]
    sorted_verts = sorted(range(N), key=lambda x: -degrees[x])
    
    # Greedy: start with top vertex, try to add others
    best_branches = [sorted_verts[0]]
    for v in sorted_verts[1:]:
        if time.time() - t_start > 70:
            break
        candidate_branches = best_branches + [v]
        paths = _find_clique_subdivision(candidate_branches, max_time=4.0)
        if paths is not None:
            best_branches = candidate_branches
    
    # Phase 2: Local search - try swapping or adding
    # Try to add any remaining vertex
    for v in range(N):
        if v in best_branches:
            continue
        if time.time() - t_start > 72:
            break
        candidate = best_branches + [v]
        paths = _find_clique_subdivision(candidate, max_time=3.0)
        if paths is not None:
            best_branches = candidate
            # After adding, try again for more
            for v2 in range(N):
                if v2 in best_branches:
                    continue
                if time.time() - t_start > 74:
                    break
                candidate2 = best_branches + [v2]
                paths2 = _find_clique_subdivision(candidate2, max_time=2.5)
                if paths2 is not None:
                    best_branches = candidate2
    
    # Phase 3: Try random restarts with different orderings
    best_result = None
    best_count = len(best_branches)
    
    if time.time() - t_start < 75:
        # Try a few random orderings
        random.seed(42)
        for attempt in range(10):
            if time.time() - t_start > 75:
                break
            random_order = sorted_verts[:]
            random.shuffle(random_order)
            cur = [random_order[0]]
            for v in random_order[1:]:
                if time.time() - t_start > 75:
                    break
                candidate = cur + [v]
                paths = _find_clique_subdivision(candidate, max_time=2.0)
                if paths is not None:
                    cur = candidate
            if len(cur) > best_count:
                best_count = len(cur)
                best_branches = cur
    
    # Final: get paths for best_branches
    if time.time() - t_start > 76:
        # Emergency: just return a simple 2-branch result
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    paths = _find_clique_subdivision(best_branches, max_time=10.0)
    if paths is None:
        # Fallback: try smaller set
        for size in range(len(best_branches) - 1, 1, -1):
            if time.time() - t_start > 78:
                break
            subset = best_branches[:size]
            paths = _find_clique_subdivision(subset, max_time=5.0)
            if paths is not None:
                best_branches = subset
                break
        if paths is None:
            u, v = P['edges'][0]
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    # Validate
    if not _validate(best_branches, paths):
        # Fallback
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    return {'branches': [int(b) for b in best_branches], 'paths': [list(int(x) for x in p) for p in paths]}
# EVOLVE-BLOCK-END
