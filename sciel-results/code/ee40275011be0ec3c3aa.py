import time
import random
import sys
from collections import defaultdict, deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
edges = P['edges']
adj = defaultdict(list)
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

# Precompute vertex ordering by degree (all degree 4 here, but use index as tiebreaker)
# For MRV heuristic, we'll dynamically compute remaining values
vertex_order = list(range(N))


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
            # BFS to check for cycles in bicolored subgraph
            visited = set()
            for start in range(N):
                if start in visited or colors[start] not in (a, b):
                    continue
                # BFS from start
                q = deque([start])
                visited.add(start)
                edge_count = 0
                vertex_count = 0
                while q:
                    node = q.popleft()
                    vertex_count += 1
                    for nb in adj[node]:
                        if colors[nb] in (a, b):
                            edge_count += 1
                            if nb not in visited:
                                visited.add(nb)
                                q.append(nb)
                edge_count //= 2  # each edge counted twice
                # For a forest: edges = vertices - components
                # A connected component with V vertices and E edges is a tree iff E = V-1
                # If E >= V, there's a cycle
                if edge_count >= vertex_count:
                    return False
    return True


def backtracking_search(K, time_limit, start_time):
    """
    Backtracking with forward checking for bicolored forest constraint.
    Returns a valid coloring or None.
    """
    colors = [-1] * N
    # Track for each color pair (a,b) with a<b, the DSU-like connectivity
    # We'll use a dict mapping (a,b) -> list of sets (connected components)
    # Actually, for efficiency, let's track edges per color pair and use BFS when needed
    
    # Better approach: maintain a list of edges per color pair
    # pair_edges[(a,b)] = list of (u,v) edges where {colors[u],colors[v]} = {a,b}
    pair_edges = defaultdict(list)  # (a,b) -> list of edges
    # Also track for quick cycle detection: for each pair, maintain DSU
    # But DSU doesn't support undo easily. Instead, we'll rebuild when needed.
    
    # Alternative: use a simple approach - for each assignment, check if adding
    # the new edges creates a cycle using BFS on the affected color pair subgraph
    
    deadline = start_time + time_limit
    nodes_visited = 0
    
    def get_mrv_vertex():
        """Find unassigned vertex with minimum remaining values."""
        best_v = -1
        best_count = K + 1
        for v in range(N):
            if colors[v] != -1:
                continue
            # Count valid colors
            valid_colors = 0
            neighbor_colors = set()
            for nb in adj[v]:
                if colors[nb] != -1:
                    neighbor_colors.add(colors[nb])
            for c in range(K):
                if c not in neighbor_colors:
                    valid_colors += 1
            if valid_colors < best_count:
                best_count = valid_colors
                best_v = v
                if best_count == 0:
                    return best_v
        return best_v
    
    def would_create_cycle(v, c):
        """Check if assigning color c to vertex v would create a bicolored cycle."""
        for nb in adj[v]:
            if colors[nb] == -1:
                continue
            nc = colors[nb]
            if nc == c:
                continue  # adjacent same color already handled by properness check
            # Check pair (min(c,nc), max(c,nc))
            a, b = min(c, nc), max(c, nc)
            # Check if v and nb are already connected in the (a,b) bicolored subgraph
            # Do a BFS from v in the subgraph induced by vertices with color a or b
            # But only through edges where both endpoints have colors in {a,b}
            # Start from v (which would have color c) and see if we can reach nb
            # through other vertices with colors in {a,b}
            
            # BFS from v
            if v == nb:
                continue
            visited = {v}
            queue = deque([v])
            found = False
            while queue:
                node = queue.popleft()
                for nb2 in adj[node]:
                    if nb2 == nb:
                        # Check if the edge (node, nb) is a bicolored edge for pair (a,b)
                        if colors[nb] in (a, b):  # nb has color nc which is in (a,b)
                            if node in (v,) or colors[node] in (a, b):
                                found = True
                                break
                    if nb2 in visited:
                        continue
                    if colors[nb2] == -1:
                        continue
                    if colors[nb2] in (a, b):
                        visited.add(nb2)
                        queue.append(nb2)
                if found:
                    break
            if found:
                return True
        return False
    
    def backtrack(depth):
        nonlocal nodes_visited
        if time.time() > deadline:
            return False
        nodes_visited += 1
        
        if depth == N:
            return True
        
        v = get_mrv_vertex()
        if v == -1:
            return depth == N  # All assigned
        
        # Try each color
        neighbor_colors = set()
        for nb in adj[v]:
            if colors[nb] != -1:
                neighbor_colors.add(colors[nb])
        
        # Order colors: try colors not used by neighbors first
        available = [c for c in range(K) if c not in neighbor_colors]
        random.shuffle(available)
        
        for c in available:
            # Check bicolored cycle constraint
            if would_create_cycle(v, c):
                continue
            
            # Assign
            colors[v] = c
            # Record edges for undo
            added_edges = []
            for nb in adj[v]:
                if colors[nb] != -1:
                    a, b = min(c, colors[nb]), max(c, colors[nb])
                    added_edges.append((a, b, v, nb))
            
            if backtrack(depth + 1):
                return True
            
            # Undo
            colors[v] = -1
            for a, b, u, w in added_edges:
                # We don't maintain pair_edges explicitly, so just undo colors
                pass
        
        return False
    
    if backtrack(0):
        return list(colors)
    return None


def local_search(K, time_limit, start_time):
    """Local search as fallback."""
    deadline = start_time + time_limit
    best_cost = float('inf')
    best_coloring = None
    
    for restart in range(100):
        if time.time() > deadline:
            break
        colors = [random.randint(0, K - 1) for _ in range(N)]
        
        current_cost = 0
        for u, v in edges:
            if colors[u] == colors[v]:
                current_cost += 100
        if current_cost > 0:
            continue
        
        # Check bicolored cycles
        used_colors = set(colors)
        color_list = list(used_colors)
        valid = True
        for i in range(len(color_list)):
            for j in range(i + 1, len(color_list)):
                a, b = color_list[i], color_list[j]
                visited = set()
                for start in range(N):
                    if start in visited or colors[start] not in (a, b):
                        continue
                    q = deque([start])
                    visited.add(start)
                    edge_count = 0
                    vertex_count = 0
                    while q:
                        node = q.popleft()
                        vertex_count += 1
                        for nb in adj[node]:
                            if colors[nb] in (a, b):
                                edge_count += 1
                                if nb not in visited:
                                    visited.add(nb)
                                    q.append(nb)
                    edge_count //= 2
                    if edge_count >= vertex_count:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return list(colors)
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Try K=3 with backtracking
    for seed in [42, 123, 456, 789, 1111, 2222, 3333]:
        remaining = deadline - time.time()
        if remaining < 3:
            break
        random.seed(seed)
        res = backtracking_search(3, min(15.0, remaining - 2.0), start_time)
        if res is not None:
            if check_validity(res, 3):
                return [int(c) for c in res]
    
    # Try K=4 with backtracking
    for seed in [42, 123, 456, 789]:
        remaining = deadline - time.time()
        if remaining < 3:
            break
        random.seed(seed)
        res = backtracking_search(4, min(15.0, remaining - 2.0), start_time)
        if res is not None:
            if check_validity(res, 4):
                return [int(c) for c in res]
    
    # Try K=5 with backtracking
    remaining = deadline - time.time()
    if remaining > 5:
        random.seed(42)
        res = backtracking_search(5, min(15.0, remaining - 2.0), start_time)
        if res is not None:
            if check_validity(res, 5):
                return [int(c) for c in res]
    
    # Fallback: local search
    remaining = deadline - time.time()
    if remaining > 5:
        for K in [3, 4, 5]:
            res = local_search(K, min(10.0, remaining - 2.0), start_time)
            if res is not None:
                return [int(c) for c in res]
            remaining = deadline - time.time()
            if remaining < 3:
                break
    
    # Last resort: use all distinct colors (always valid)
    return [int(i) for i in range(N)]


# EVOLVE-BLOCK-END
