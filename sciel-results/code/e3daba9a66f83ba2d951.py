import time
import random
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
edges = P['edges']
adj = defaultdict(list)
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

# Precompute vertex ordering by degree (all degree 4 here, so use index)
vertex_order = list(range(N))

def backtracking_solve(K, deadline):
    """Backtracking search with forward checking for bicolored cycle constraint."""
    colors = [-1] * N
    # For each color pair (a,b), track edges that would create a cycle
    # We'll use a simpler approach: check validity incrementally
    
    # Order vertices: try to assign in an order that maximizes constraint propagation
    # Since all vertices have degree 4, use a heuristic order
    # Try vertices in order 0,1,2,... but also try other orderings
    
    def is_valid_assignment(v, c):
        """Check if assigning color c to vertex v is valid given current partial assignment."""
        # Check properness: no adjacent vertex has color c
        for u in adj[v]:
            if colors[u] == c:
                return False
        
        # Check bicolored cycle: for each already-assigned neighbor u with color cu != c,
        # adding edge (v,u) to the bicolored subgraph (c, cu) must not create a cycle
        # We need to check if v and u are already connected in the bicolored subgraph (c, cu)
        
        for u in adj[v]:
            cu = colors[u]
            if cu == -1 or cu == c:
                continue
            # Check if v and u are connected in the subgraph induced by colors {c, cu}
            # among already-assigned vertices
            # BFS/DFS from v through vertices with color c or cu
            visited = set()
            stack = [v]
            visited.add(v)
            while stack:
                x = stack.pop()
                if x == u:
                    return False  # cycle would be created
                for y in adj[x]:
                    if y in visited:
                        continue
                    if colors[y] == c or colors[y] == cu:
                        visited.add(y)
                        stack.append(y)
        
        return True
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == N:
            return True
        
        v = vertex_order[idx]
        
        # Try colors in order 0, 1, ..., K-1
        # Use forward checking: try each color and see if it's valid
        for c in range(K):
            if is_valid_assignment(v, c):
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                colors[v] = -1
        
        return False
    
    if backtrack(0):
        return colors
    return None

def check_validity(colors, K):
    """Verify a complete coloring is valid."""
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    used_colors = set(colors)
    if len(used_colors) < 2:
        return True
    color_list = list(used_colors)
    for i in range(len(color_list)):
        for j in range(i + 1, len(color_list)):
            a, b = color_list[i], color_list[j]
            parent = list(range(N))
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                px, py = find(x), find(y)
                if px == py:
                    return False
                parent[py] = px
                return True
            for u, v in edges:
                cu, cv = colors[u], colors[v]
                if (cu == a and cv == b) or (cu == b and cv == a):
                    if not union(u, v):
                        return False
    return True

def local_search(K, time_limit):
    """Local search fallback."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    for _ in range(200):
        if time.time() > deadline:
            break
        colors = [random.randint(0, K - 1) for _ in range(N)]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for v in range(N):
                old_c = colors[v]
                for c in range(K):
                    if c == old_c:
                        continue
                    # Check if changing v to c is valid
                    valid = True
                    for u in adj[v]:
                        if colors[u] == c:
                            valid = False
                            break
                    if not valid:
                        continue
                    # Check bicolored cycles
                    colors[v] = c
                    ok = True
                    used = set(colors)
                    cl = list(used)
                    for i in range(len(cl)):
                        for j in range(i+1, len(cl)):
                            a, b = cl[i], cl[j]
                            parent = list(range(N))
                            def find(x):
                                while parent[x] != x:
                                    parent[x] = parent[parent[x]]
                                    x = parent[x]
                                return x
                            def union(x, y):
                                px, py = find(x), find(y)
                                if px == py:
                                    return False
                                parent[py] = px
                                return True
                            for u, v2 in edges:
                                cu, cv = colors[u], colors[v2]
                                if (cu == a and cv == b) or (cu == b and cv == a):
                                    if not union(u, v2):
                                        ok = False
                                        break
                            if not ok:
                                break
                        if not ok:
                            break
                    if ok:
                        improved = True
                        break
                    colors[v] = old_c
        
        if check_validity(colors, K):
            return colors
    
    return None

def construct():
    deadline = time.time() + 55.0
    
    # Try K=3 with backtracking
    res = backtracking_solve(3, deadline - 5.0)
    if res is not None and check_validity(res, 3):
        return [int(c) for c in res]
    
    # Try K=4 with backtracking
    remaining = deadline - time.time()
    if remaining > 5:
        res = backtracking_solve(4, deadline - 3.0)
        if res is not None and check_validity(res, 4):
            return [int(c) for c in res]
    
    # Try K=4 with local search
    remaining = deadline - time.time()
    if remaining > 5:
        res = local_search(4, min(15.0, remaining - 3.0))
        if res is not None:
            return [int(c) for c in res]
    
    # Try K=5 with local search
    remaining = deadline - time.time()
    if remaining > 3:
        res = local_search(5, min(10.0, remaining - 2.0))
        if res is not None:
            return [int(c) for c in res]
    
    # Fallback: proper 4-coloring (should always be valid for degree-4 graph with forest constraint? No, need to check)
    # Use a simple proper coloring and hope it satisfies forest constraint
    colors = [i % 4 for i in range(N)]
    if check_validity(colors, 4):
        return colors
    
    # Last resort
    return list(range(N))

# EVOLVE-BLOCK-END
