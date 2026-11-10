"""Rewritten program: find max clique in graph square, then extract disjoint paths."""
# EVOLVE-BLOCK-START
import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

_DEADLINE = time.time() + 80.0

def _build_graph():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bfs_distances(adj, src):
    """BFS from src, returns distance array."""
    n = len(adj)
    dist = [-1] * n
    dist[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def _build_square(adj):
    """Build the square of the graph: edge if distance <= 2."""
    n = len(adj)
    sq = [set() for _ in range(n)]
    for u in range(n):
        dist = _bfs_distances(adj, u)
        for v in range(n):
            if v != u and dist[v] <= 2:
                sq[u].add(v)
    return sq

def _max_clique_bk(sq):
    """Bron–Kerbosch with pivot to find a maximum clique in sq."""
    n = len(sq)
    best = [0]
    best_clique = [set()]

    def bk(R, P_set, X_set):
        if time.time() > _DEADLINE:
            return
        if not P_set and not X_set:
            if len(R) > len(best_clique[0]):
                best_clique[0] = set(R)
                best[0] = len(R)
            return
        # Pivot: choose vertex with max degree in P
        if P_set:
            pivot = max(P_set, key=lambda v: len(P_set & sq[v]))
        else:
            pivot = -1
        candidates = P_set - sq[pivot] if pivot >= 0 else set(P_set)
        for v in list(candidates):
            if time.time() > _DEADLINE:
                return
            new_R = R | {v}
            new_P = P_set & sq[v]
            new_X = X_set & sq[v]
            bk(new_R, new_P, new_X)
            P_set.remove(v)
            X_set.add(v)

    bk(set(), set(range(n)), set())
    return best_clique[0]

def _find_path(adj, src, dst, blocked):
    """BFS to find a simple path from src to dst avoiding blocked vertices (except endpoints)."""
    n = len(adj)
    parent = [-1] * n
    parent[src] = -2
    q = deque([src])
    while q:
        u = q.popleft()
        if u == dst:
            # Reconstruct path
            path = []
            cur = dst
            while cur != -2:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path
        for v in adj[u]:
            if parent[v] == -1 and v not in blocked:
                parent[v] = u
                q.append(v)
    return None

def _try_paths(adj, branches, order):
    """Try to find internally-disjoint paths for all pairs in given order.
    Returns dict of (i,j) -> path, or None if fails."""
    n = len(adj)
    blocked = set(branches)
    paths = {}
    for idx, (i, j) in enumerate(order):
        path = _find_path(adj, branches[i], branches[j], blocked)
        if path is None:
            return None
        # Mark interior vertices as blocked
        for v in path[1:-1]:
            blocked.add(v)
        paths[(i, j)] = path
    return paths

def _try_branch_set(adj, sq, branches):
    """Try to find disjoint paths for a set of branch vertices.
    Try multiple orderings to improve success rate."""
    k = len(branches)
    # Generate pair indices
    pairs = list(combinations(range(k), 2))
    
    # Try different orderings
    # 1. Natural order
    orders = [pairs]
    # 2. Reverse
    orders.append(list(reversed(pairs)))
    # 3. Sort by distance (shortest first) - might help
    def pair_dist(p):
        i, j = p
        # Use BFS distance
        dist = _bfs_distances(adj, branches[i])
        return dist[branches[j]]
    orders.append(sorted(pairs, key=pair_dist))
    orders.append(sorted(pairs, key=pair_dist, reverse=True))
    
    for order in orders:
        result = _try_paths(adj, branches, order)
        if result is not None:
            return result
    return None

def _greedy_clique_expand(adj, sq, initial_clique):
    """Try to expand a clique greedily."""
    clique = set(initial_clique)
    # Try adding vertices one by one
    changed = True
    while changed and time.time() < _DEADLINE:
        changed = False
        for v in range(P['n']):
            if v in clique:
                continue
            # Check if v can be added (connected to all in clique in sq)
            if all(u in sq[v] for u in clique):
                # Try the expanded set
                new_clique = clique | {v}
                branches = sorted(new_clique)
                result = _try_branch_set(adj, sq, branches)
                if result is not None:
                    clique = new_clique
                    changed = True
                    break
    return clique

def _local_search(adj, sq, initial_clique):
    """Local search: try to improve clique by swapping vertices."""
    current = set(initial_clique)
    branches = sorted(current)
    result = _try_branch_set(adj, sq, branches)
    if result is None:
        return None, None
    
    best_size = len(current)
    best_result = result
    best_clique = set(current)
    
    for _ in range(50):  # limited iterations
        if time.time() > _DEADLINE:
            break
        improved = False
        # Try adding a vertex
        for v in range(P['n']):
            if v in current:
                continue
            if all(u in sq[v] for u in current):
                new_clique = current | {v}
                new_branches = sorted(new_clique)
                new_result = _try_branch_set(adj, sq, new_branches)
                if new_result is not None:
                    current = new_clique
                    result = new_result
                    best_size = len(current)
                    best_result = new_result
                    best_clique = set(current)
                    improved = True
                    break
        if improved:
            continue
        # Try swapping: remove one, add another
        for to_remove in list(current):
            if time.time() > _DEADLINE:
                break
            for to_add in range(P['n']):
                if to_add in current or to_add == to_remove:
                    continue
                new_clique = (current - {to_remove}) | {to_add}
                if len(new_clique) <= best_size:
                    continue
                # Check clique property in sq
                if not all(a in sq[b] for a, b in combinations(sorted(new_clique), 2)):
                    continue
                new_branches = sorted(new_clique)
                new_result = _try_branch_set(adj, sq, new_branches)
                if new_result is not None:
                    current = new_clique
                    result = new_result
                    best_size = len(current)
                    best_result = new_result
                    best_clique = set(current)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    if best_result is None:
        return None, None
    return best_clique, best_result

def construct():
    adj = _build_graph()
    sq = _build_square(adj)
    
    # Find max clique in square
    clique = _max_clique_bk(sq)
    
    if not clique:
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    branches = sorted(clique)
    result = _try_branch_set(adj, sq, branches)
    
    if result is None:
        # Fallback: try smaller subsets
        for size in range(len(branches) - 1, 1, -1):
            if time.time() > _DEADLINE:
                break
            for subset in combinations(branches, size):
                sub_set = set(subset)
                sub_sorted = sorted(sub_set)
                r = _try_branch_set(adj, sq, sub_sorted)
                if r is not None:
                    result = r
                    branches = sub_sorted
                    break
            if result is not None:
                break
    
    if result is None:
        # Last resort: just an edge
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    # Build the final paths dict
    k = len(branches)
    paths_dict = {}
    for i in range(k):
        for j in range(i + 1, k):
            path = result[(i, j)]
            paths_dict[(int(branches[i]), int(branches[j]))] = [int(x) for x in path]
    
    return {
        'branches': [int(b) for b in branches],
        'paths': [paths_dict[(int(branches[i]), int(branches[j]))] for i in range(k) for j in range(i+1, k)]
    }
# EVOLVE-BLOCK-END
