import time
import random
import sys
from collections import deque

def _has_girth_at_least(adj, n, target_girth):
    """Check if girth >= target_girth using BFS. Returns False early if a shorter cycle is found."""
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            v = queue.popleft()
            dv = dist[v]
            for u in adj[v]:
                if dist[u] == -1:
                    dist[u] = dv + 1
                    parent[u] = v
                    queue.append(u)
                elif u != parent[v] and v != parent[u]:
                    if dv + dist[u] + 1 < target_girth:
                        return False
    return True

def _bfs_girth(adj, n):
    """Compute exact girth using BFS from each vertex."""
    min_girth = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            v = queue.popleft()
            for u in adj[v]:
                if dist[u] == -1:
                    dist[u] = dist[v] + 1
                    parent[u] = v
                    queue.append(u)
                elif u != parent[v] and v != parent[u]:
                    cycle_len = dist[v] + dist[u] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                    if min_girth <= 2:
                        return min_girth
    return min_girth

def _greedy_backtrack_bipartite(n, k, g, deadline):
    """
    Attempt to construct a k-regular bipartite graph with girth >= g using greedy backtracking.
    n is total vertices, so each part has n/2 vertices.
    """
    if n % 2 != 0:
        return None
    half = n // 2
    # adj[i] is list of neighbors in the other part
    # We build adjacency for the whole graph
    adj = [[] for _ in range(n)]
    # deg[i] is current degree
    deg = [0] * n
    # edges_set to avoid duplicates
    edges = set()
    
    # We need to add k * half edges.
    # Each edge connects i in [0, half) to j in [half, n).
    
    # Helper to check if adding edge (u, v) creates a cycle < g
    # u in [0, half), v in [half, n)
    # Cycle length = 2 * (shortest path length between u and v in existing graph) + 1?
    # No, in bipartite graph, if we add edge (u, v), a cycle is formed if there is a path between u and v.
    # The cycle length is len(path) + 1.
    # Since graph is bipartite, paths between u and v have odd length?
    # u is in part 0, v is in part 1.
    # Path from u to v has length L. Cycle length is L + 1.
    # We require cycle length >= g.
    # So L + 1 >= g => L >= g - 1.
    # Since g is even (16), g-1 is odd (15).
    # Paths between part 0 and part 1 have odd length.
    # So we need shortest path length >= 15.
    # This means we must NOT have a path of length 1, 3, 5, 7, 9, 11, 13.
    
    def can_add_edge(u, v):
        # u < half, v >= half
        # Check if shortest path between u and v is < g - 1
        target_path_len = g - 1
        # BFS from u to v, stop if dist >= target_path_len
        dist = [-1] * n
        dist[u] = 0
        queue = deque([u])
        while queue:
            curr = queue.popleft()
            if dist[curr] >= target_path_len:
                return True # Path is long enough (or no path found within limit)
            for neighbor in adj[curr]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[curr] + 1
                    if neighbor == v:
                        # Found path of length dist[v]
                        if dist[v] < target_path_len:
                            return False
                        else:
                            return True
                    queue.append(neighbor)
        # If v not reached, no path, so safe
        return True

    # State: list of edges added
    current_edges = []
    
    # Heuristic: Order vertices by degree (least degree first)
    # We need to add k * half edges.
    
    def backtrack(edge_index):
        if time.time() > deadline:
            return False
        if edge_index == k * half:
            # Check if all degrees are k
            for i in range(n):
                if deg[i] != k:
                    return False
            return True
        
        # Select a vertex with degree < k
        # To be efficient, we can just iterate through all possible pairs?
        # That's O(n^2).
        # Better: Pick a vertex u with deg[u] < k, and a vertex v with deg[v] < k.
        
        # Find a u with deg < k
        # Shuffle to avoid bias?
        u_candidates = [i for i in range(half) if deg[i] < k]
        if not u_candidates:
            return False
        random.shuffle(u_candidates)
        
        found = False
        for u in u_candidates:
            v_candidates = [j for j in range(half) if deg[half + j] < k and (u, j) not in edges]
            if not v_candidates:
                continue
            random.shuffle(v_candidates)
            
            for j in v_candidates:
                v = half + j
                if can_add_edge(u, v):
                    # Add edge
                    adj[u].append(v)
                    adj[v].append(u)
                    deg[u] += 1
                    deg[v] += 1
                    edges.add((u, j))
                    current_edges.append((u, v))
                    
                    if backtrack(edge_index + 1):
                        return True
                    
                    # Undo
                    adj[u].pop()
                    adj[v].pop()
                    deg[u] -= 1
                    deg[v] -= 1
                    edges.remove((u, j))
                    current_edges.pop()
        
        return False

    # Start backtracking
    # To speed up, we can fix the first few edges?
    # For now, just run.
    if backtrack(0):
        return n, current_edges
    
    return None

def _construct_cage_3_16():
    """Construct a 3-regular bipartite graph with girth >= 16."""
    n = 936
    half = n // 2
    deadline = time.time() + 105.0
    
    # Try greedy backtracking for smaller n first
    # Moore bound for (3,16) is 512.
    # Try n from 512 to 1000 in steps of 2 (bipartite)
    for n_try in range(512, 1200, 2):
        if time.time() > deadline - 30: # Leave time for fallback
            break
        res = _greedy_backtrack_bipartite(n_try, 3, 16, deadline - 30)
        if res:
            return res
            
    # Fallback to random matching (original)
    def _random_perfect_matching():
        perm = list(range(half))
        random.shuffle(perm)
        return perm
        
    for attempt in range(2000):
        if time.time() > deadline:
            break
        matchings = [_random_perfect_matching() for _ in range(3)]
        edge_set = set()
        valid = True
        for m in matchings:
            for i in range(half):
                edge = (i, m[i])
                if edge in edge_set:
                    valid = False
                    break
                edge_set.add(edge)
            if not valid:
                break
        if not valid:
            continue
        adj = [[] for _ in range(n)]
        for m in matchings:
            for i in range(half):
                j = m[i]
                adj[i].append(half + j)
                adj[half + j].append(i)
        if _has_girth_at_least(adj, n, 16):
            edges = []
            for i in range(half):
                for j in adj[i]:
                    if j >= half:
                        edges.append((i, j))
            return n, edges
    return None

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
    q = k
    deadline = time.time() + 38.0
    
    def _build_luw(d):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
        rule_list = [rule[c] for c in range(2, d + 1)]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for idx, c in enumerate(range(2, d + 1)):
                    a, b = rule_list[idx]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)
        
        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        i = 0
        while i < len(order):
            vertex = order[i]
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
            i += 1
        return len(order), edges

    for d in range(5, 20):
        if time.time() > deadline:
            break
        n, edges = _build_luw(d)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        if _has_girth_at_least(adj, n, g):
            return n, edges
        
    d_orig = max(8, g - 4)
    return _build_luw(d_orig)

def construct(k, g):
    start_time = time.time()
    
    if k == 3 and g == 16:
        result = _construct_cage_3_16()
        if result is not None:
            return result
        deadline = time.time() + 40.0
        return _luw_construct(k, g)
    
    return _luw_construct(k, g)
