"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
from collections import defaultdict, deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

_ADJ = None
_DEADLINE = None
_TIME_MARGIN = 0.5  # seconds before hard limit to stop

def _time_left():
    return _DEADLINE - _time.time() - _TIME_MARGIN

def _get_adj():
    global _ADJ
    if _ADJ is None:
        _ADJ = defaultdict(set)
        for u, v in P['edges']:
            _ADJ[u].add(v)
            _ADJ[v].add(u)
    return _ADJ

def _bipartite_max_matching(left_nodes, right_nodes, adj_map):
    """
    Hopcroft-Karp for max bipartite matching.
    left_nodes: list of left-side node ids
    right_nodes: list of right-side node ids
    adj_map: dict mapping left_node -> set of right_nodes it connects to
    Returns (match_count, matchL) where matchL[left_node] = right_node or None
    """
    from collections import defaultdict as _dd
    matchL = {u: None for u in left_nodes}
    matchR = {v: None for v in right_nodes}
    dist = {}
    
    def bfs():
        q = deque()
        for u in left_nodes:
            if matchL[u] is None:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = float('inf')
        found = False
        while q:
            u = q.popleft()
            for v in adj_map.get(u, ()):
                mu = matchR.get(v)
                if mu is None:
                    found = True
                elif dist[mu] == float('inf'):
                    dist[mu] = dist[u] + 1
                    q.append(mu)
        return found
    
    def dfs(u):
        for v in adj_map.get(u, ()):
            mu = matchR.get(v)
            if mu is None or (dist.get(mu, float('inf')) == dist[u] + 1 and dfs(mu)):
                matchL[u] = v
                matchR[v] = u
                return True
        dist[u] = float('inf')
        return False
    
    matching = 0
    while bfs():
        for u in left_nodes:
            if matchL[u] is None:
                if dfs(u):
                    matching += 1
    return matching, matchL

def _try_assign_steiner(branches, steiner_pool):
    """
    Try to assign a distinct Steiner vertex to each edge of the complete graph on branches.
    A Steiner s can serve edge (u,v) if u-s and s-v are both edges.
    Returns dict edge_key -> steiner, or None if impossible.
    """
    adj = _get_adj()
    branch_set = set(branches)
    edges = list(combinations(branches, 2))
    if not edges:
        return {}
    
    # Build bipartite graph: left = edges, right = steiner_pool
    # Left node id: index into edges list
    # Right node id: steiner vertex
    adj_map = {}
    for i, (u, v) in enumerate(edges):
        adj_map[i] = set()
        for s in steiner_pool:
            if s not in branch_set and u in adj[s] and v in adj[s]:
                adj_map[i].add(s)
    
    left_nodes = list(range(len(edges)))
    right_nodes = list(steiner_pool)
    
    if not right_nodes:
        return None
    
    matching, matchL = _bipartite_max_matching(left_nodes, right_nodes, adj_map)
    
    if matching < len(edges):
        return None
    
    # Convert back
    assignment = {}
    for i, (u, v) in enumerate(edges):
        s = matchL[i]
        if s is None:
            return None
        key = (min(u, v), max(u, v))
        assignment[key] = s
    
    return assignment

def _find_path_bfs(start, end, blocked, adj, max_depth=6):
    """Find a simple path from start to end avoiding blocked vertices (interior)."""
    if start == end:
        return [start]
    if start not in adj or end not in adj:
        return None
    # BFS
    parent = {start: None}
    q = deque([start])
    while q:
        u = q.popleft()
        if u == end:
            # Reconstruct
            path = []
            cur = u
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path
        for v in adj[u]:
            if v not in parent and v not in blocked and v != start:
                parent[v] = u
                q.append(v)
    return None

def _solve_with_long_paths(branches, adj, n):
    """
    Try to find disjoint interior paths for each pair of branches.
    Uses greedy BFS with backtracking-lite: try each pair, find shortest path avoiding used interiors.
    Returns dict edge_key -> path, or None.
    """
    branch_set = set(branches)
    edges = sorted(combinations(branches, 2))
    used_interiors = set()
    paths = {}
    
    for u, v in edges:
        if _time_left() <= 0:
            return None
        # Try BFS for shortest path avoiding used_interiors and other branches
        path = _find_path_bfs(u, v, used_interiors | (branch_set - {u, v}), adj)
        if path is None:
            return None
        # Check interior is valid
        interior = set(path[1:-1])
        if interior & used_interiors:
            return None
        if interior & branch_set:
            return None
        used_interiors |= interior
        paths[(u, v)] = path
    
    return paths

def _check_and_build(branches, adj, n):
    """
    Given a branch set, try to find valid paths.
    First try length-3 (single Steiner per edge) via matching.
    If that fails, try BFS with longer paths.
    Returns (branches_list, paths_dict) or None.
    """
    if len(branches) < 2:
        return None
    
    branch_set = set(branches)
    non_branches = [v for v in range(n) if v not in branch_set]
    
    # Try 1: Steiner matching for length-3 paths
    assignment = _try_assign_steiner(branches, non_branches)
    if assignment is not None:
        # Verify and build
        paths = {}
        for (u, v), s in assignment.items():
            paths[(u, v)] = [u, s, v]
        return list(branches), paths
    
    # Try 2: BFS with longer paths
    paths = _solve_with_long_paths(branches, adj, n)
    if paths is not None:
        return list(branches), paths
    
    return None

def _greedy_clique(adj, n, max_size=None):
    """
    Greedy: build a large set of vertices that are "compatible" — 
    for each pair, there exists at least one common neighbor not in the set.
    This is a relaxation of the clique condition for length-3 paths.
    """
    if max_size is None:
        max_size = n
    # Sort by degree descending
    verts = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
    selected = []
    selected_set = set()
    
    for v in verts:
        if len(selected) >= max_size:
            break
        # Check compatibility with all selected
        compatible = True
        for u in selected:
            # Need a common neighbor of u and v not in selected_set
            common = adj[u] & adj[v]
            common -= selected_set
            if not common:
                compatible = False
                break
        if compatible:
            selected.append(v)
            selected_set.add(v)
    
    return selected

def _try_grow(branches, adj, n):
    """
    Try to grow the branch set by adding one vertex at a time.
    Returns a larger valid branch set or the original.
    """
    current = list(branches)
    current_set = set(current)
    improved = True
    while improved and _time_left() > 0:
        improved = False
        # Try adding each non-branch vertex
        candidates = [v for v in range(n) if v not in current_set]
        # Sort by degree
        candidates.sort(key=lambda v: len(adj[v]), reverse=True)
        
        for v in candidates:
            if _time_left() <= 0:
                break
            new_set = current + [v]
            result = _check_and_build(new_set, adj, n)
            if result is not None:
                current = result[0]
                current_set = set(current)
                improved = True
                break
    
    return current

def construct():
    global _DEADLINE
    _DEADLINE = _time.time() + 89.0  # 1s safety margin
    
    adj = _get_adj()
    n = P['n']
    
    best_result = None
    best_count = 0
    
    # Strategy: Start with greedy compatible set, then try to grow
    # Also try different starting points
    
    # Phase 1: Greedy compatible set
    greedy = _greedy_clique(adj, n)
    if len(greedy) >= 2:
        result = _check_and_build(greedy, adj, n)
        if result is not None:
            branches, paths = result
            best_result = (branches, paths)
            best_count = len(branches)
    
    # Phase 2: Try to grow from best
    if best_result is not None and _time_left() > 0:
        grown = _try_grow(best_result[0], adj, n)
        if len(grown) > best_count:
            result = _check_and_build(grown, adj, n)
            if result is not None:
                branches, paths = result
                best_result = (branches, paths)
                best_count = len(branches)
    
    # Phase 3: Try random restarts with different seeds
    import random
    rng = random.Random(42)
    for _ in range(50):
        if _time_left() <= 1.0:
            break
        # Random subset of high-degree vertices
        high_deg = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
        k = min(n, best_count + 1)
        subset = rng.sample(high_deg, min(k, len(high_deg)))
        if len(subset) < 2:
            continue
        result = _check_and_build(subset, adj, n)
        if result is not None:
            branches, paths = result
            if len(branches) > best_count:
                best_count = len(branches)
                best_result = (branches, paths)
                # Try to grow
                if _time_left() > 0:
                    grown = _try_grow(branches, adj, n)
                    if len(grown) > best_count:
                        result2 = _check_and_build(grown, adj, n)
                        if result2 is not None:
                            branches, paths = result2
                            best_count = len(branches)
                            best_result = (branches, paths)
    
    # Phase 4: Systematic search for larger sets
    if _time_left() > 0:
        # Try all subsets of size best_count+1 from high-degree vertices
        high_deg = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
        top = high_deg[:min(15, n)]
        target_size = best_count + 1
        if target_size <= len(top):
            for combo in combinations(top, target_size):
                if _time_left() <= 0.5:
                    break
                result = _check_and_build(list(combo), adj, n)
                if result is not None:
                    branches, paths = result
                    if len(branches) > best_count:
                        best_count = len(branches)
                        best_result = (branches, paths)
    
    # Fallback: always return at least 2 branches
    if best_result is None:
        u, v = P['edges'][0]
        best_result = ([u, v], {(min(u,v), max(u,v)): [u, v]})
    
    branches, paths = best_result
    
    # Build output
    out_paths = []
    for u, v in combinations(branches, 2):
        key = (min(u, v), max(u, v))
        path = paths.get(key)
        if path is None:
            # Should not happen, but fallback
            path = [u, v]
        out_paths.append(list(path))
    
    return {
        'branches': list(branches),
        'paths': out_paths
    }
# EVOLVE-BLOCK-END
