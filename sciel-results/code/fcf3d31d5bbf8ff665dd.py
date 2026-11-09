"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_bicolor_forest_fast(colors, adj, n):
    """
    Check bicolored forest constraint using union-find.
    Returns True if valid, False otherwise.
    """
    colors_present = set(colors)
    if len(colors_present) < 2:
        return True
    
    color_list = sorted(colors_present)
    
    for i in range(len(color_list)):
        for j in range(i+1, len(color_list)):
            c1 = color_list[i]
            c2 = color_list[j]
            
            # Union-Find for vertices with color c1 or c2
            parent = {}
            rank = {}
            
            def find(x):
                if x not in parent:
                    parent[x] = x
                    rank[x] = 0
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True
            
            for u in range(n):
                if colors[u] == c1 or colors[u] == c2:
                    find(u)
            
            for u, v in P['edges']:
                if (colors[u] == c1 or colors[u] == c2) and (colors[v] == c1 or colors[v] == c2):
                    if not union(u, v):
                        return False
    
    return True

def _is_valid_coloring(colors, adj, n):
    """
    Check if coloring is valid:
    1. Proper coloring
    2. Bicolored forest constraint
    """
    # Check proper coloring
    for u, v in P['edges']:
        if colors[u] == colors[v]:
            return False
    
    # Check bicolored forest
    if not _check_bicolor_forest_fast(colors, adj, n):
        return False
    
    return True

def _count_distinct(colors):
    return len(set(colors))

def _try_pattern(pattern, n):
    """
    Try a periodic pattern. pattern is a list of color values for one period.
    Returns a full coloring if valid, else None.
    """
    period = len(pattern)
    colors = [pattern[i % period] for i in range(n)]
    
    if _is_valid_coloring(colors, _build_adjacency(), n):
        return colors
    return None

def _backtrack_color(k, adj, n, deadline, order):
    """
    Backtracking with forward checking and constraint propagation.
    """
    colors = [-1] * n
    
    # Track which colors are forbidden for each vertex due to proper coloring
    # and which color pairs create bicolored cycles
    
    def is_feasible(v, c):
        if time.time() > deadline:
            return False
            
        # Check proper coloring: no neighbor has color c
        for u in adj[v]:
            if colors[u] == c:
                return False
        
        # Check bicolored cycle constraint
        # For each neighbor color c2 != c, check if adding v with color c
        # creates a cycle in the {c, c2} subgraph
        neighbor_colors = set()
        for u in adj[v]:
            if colors[u] != -1:
                neighbor_colors.add(colors[u])
        
        for c2 in neighbor_colors:
            if c2 == c:
                continue
            
            # Find neighbors of v that have color c2
            neighbors_in_c2 = [u for u in adj[v] if colors[u] == c2]
            if len(neighbors_in_c2) < 2:
                continue
            
            # Check if there's a path between any two neighbors in c2
            # through vertices colored c or c2 (excluding v)
            start = neighbors_in_c2[0]
            target_set = set(neighbors_in_c2[1:])
            
            visited = set()
            queue = deque([start])
            visited.add(start)
            
            while queue:
                curr = queue.popleft()
                for nxt in adj[curr]:
                    if nxt == v:
                        continue
                    if colors[nxt] != c and colors[nxt] != c2:
                        continue
                    if nxt in visited:
                        continue
                    visited.add(nxt)
                    if nxt in target_set:
                        return False
                    queue.append(nxt)
                        
        return True

    def bt(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        
        v = order[idx]
        
        # Try colors 0 to k-1
        for c in range(k):
            if is_feasible(v, c):
                colors[v] = c
                if bt(idx + 1):
                    return True
                colors[v] = -1
        return False

    if bt(0):
        return colors
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    n = P['n']
    adj = _build_adjacency()
    
    best_result = None
    best_score = n + 1
    
    # Strategy 1: Try explicit periodic patterns
    # The graph is a 5x10 torus. Let's try various small patterns.
    
    # Pattern: row-based (period 10 along rows, but with wraparound)
    # Try 2-color patterns
    patterns_2 = [
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # alternating
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    ]
    
    for pat in patterns_2:
        res = _try_pattern(pat, n)
        if res is not None:
            distinct = _count_distinct(res)
            if distinct < best_score:
                best_score = distinct
                best_result = res[:]
    
    # Try 3-color patterns
    patterns_3 = [
        [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
        [0, 1, 2, 1, 0, 2, 1, 0, 2, 1],
        [0, 1, 0, 1, 2, 0, 1, 0, 1, 2],
    ]
    
    for pat in patterns_3:
        res = _try_pattern(pat, n)
        if res is not None:
            distinct = _count_distinct(res)
            if distinct < best_score:
                best_score = distinct
                best_result = res[:]
    
    # Strategy 2: Backtracking search with increasing k
    # Use a good vertex ordering: BFS from vertex 0
    from collections import deque as dq
    visited = [False] * n
    order = []
    q = dq([0])
    visited[0] = True
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                q.append(v)
    
    # Try k from 3 to 10
    for k in range(3, 11):
        if time.time() > deadline - 2.0:
            break
        
        # Try BFS order
        res = _backtrack_color(k, adj, n, deadline, order)
        if res is not None:
            if _is_valid_coloring(res, adj, n):
                distinct = _count_distinct(res)
                if distinct < best_score:
                    best_score = distinct
                    best_result = res[:]
                break
        
        # Try natural order
        res = _backtrack_color(k, adj, n, deadline, list(range(n)))
        if res is not None:
            if _is_valid_coloring(res, adj, n):
                distinct = _count_distinct(res)
                if distinct < best_score:
                    best_score = distinct
                    best_result = res[:]
                break
        
        # Try reversed order
        res = _backtrack_color(k, adj, n, deadline, list(range(n-1, -1, -1)))
        if res is not None:
            if _is_valid_coloring(res, adj, n):
                distinct = _count_distinct(res)
                if distinct < best_score:
                    best_score = distinct
                    best_result = res[:]
                break
    
    # Fallback: use all distinct colors
    if best_result is None:
        best_result = list(range(n))
        best_score = n
    
    return [int(c) for c in best_result]
# EVOLVE-BLOCK-END
