"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _has_bicolor_cycle(adj, color, vertex, coloring):
    """Check if adding vertex with 'color' creates a bicolored cycle.
    A bicolored cycle exists if two neighbors of 'vertex' that also have 'color'
    are already connected in the subgraph induced by 'color'."""
    colored_neighbors = [nb for nb in adj[vertex] if coloring[nb] == color]
    if len(colored_neighbors) < 2:
        return False
    # BFS from the first colored neighbor within the color subgraph
    visited = {colored_neighbors[0]}
    queue = deque([colored_neighbors[0]])
    while queue:
        u = queue.popleft()
        for nb in adj[u]:
            if coloring[nb] == color and nb not in visited:
                visited.add(nb)
                queue.append(nb)
    for nb in colored_neighbors[1:]:
        if nb in visited:
            return True
    return False

def _is_valid_assignment(adj, vertex, color, coloring):
    """Check if assigning 'color' to 'vertex' is valid:
    1. No neighbor has the same color.
    2. No bicolored cycle is created."""
    for nb in adj[vertex]:
        if coloring[nb] == color:
            return False
    # Check for bicolored cycle
    if _has_bicolor_cycle(adj, color, vertex, coloring):
        return False
    return True

def _backtrack(adj, coloring, vertex_order, idx, max_color):
    """Recursive backtracking to find a valid coloring."""
    if idx == len(vertex_order):
        return True
    vertex = vertex_order[idx]
    for color in range(max_color + 1):
        if _is_valid_assignment(adj, vertex, color, coloring):
            coloring[vertex] = color
            if _backtrack(adj, coloring, vertex_order, idx + 1, max_color):
                return True
            coloring[vertex] = -1
    return False

def _find_min_colors(adj, n, deadline):
    """Find the minimum number of colors using backtracking with deadline."""
    # Try increasing number of colors
    for max_color in range(1, n):
        coloring = [-1] * n
        # Vertex ordering: use degree-based heuristic (highest degree first)
        degrees = [len(adj[v]) for v in range(n)]
        vertex_order = sorted(range(n), key=lambda v: -degrees[v])
        
        if _backtrack(adj, coloring, vertex_order, 0, max_color):
            return max_color + 1, coloring
        if time.time() > deadline:
            break
    return None, None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = P['n']
    edges = P['edges']
    adj = _build_adjacency(edges, n)
    
    # Try to find minimum colors
    min_colors, coloring = _find_min_colors(adj, n, deadline)
    
    if coloring is not None:
        # Verify the solution
        valid = True
        for u, v in edges:
            if coloring[u] == coloring[v]:
                valid = False
                break
        if valid:
            # Check bicolored cycles
            for c in range(min_colors):
                # Build subgraph for color c
                in_color = [v for v in range(n) if coloring[v] == c]
                in_color_set = set(in_color)
                # Check for cycles using DFS
                visited = set()
                for start in in_color:
                    if start in visited:
                        continue
                    # DFS
                    stack = [(start, -1)]
                    while stack:
                        node, parent = stack.pop()
                        if node in visited:
                            continue
                        visited.add(node)
                        for nb in adj[node]:
                            if nb in in_color_set:
                                if nb == parent:
                                    continue
                                if nb in visited:
                                    valid = False
                                    break
                                stack.append((nb, node))
                    if not valid:
                        break
            if valid:
                return [int(coloring[v]) for v in range(n)]
    
    # Fallback: use all different colors (always valid)
    return list(range(n))

# EVOLVE-BLOCK-END
