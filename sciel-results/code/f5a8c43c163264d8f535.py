import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    if k != 3:
        # Fallback for other k, though task is specific to k=3
        return _construct_luw(k, g)
        
    # Moore bound for n(3, 16)
    # n >= 2 * (1 + 2 + 4 + 8 + 16 + 32 + 64 + 128) = 510
    moore_bound = 510
    
    # Petersen graph definition
    # Vertices 0-9
    # Edges:
    # Outer cycle: 0-1, 1-2, 2-3, 3-4, 4-0
    # Inner star: 5-7, 7-9, 9-6, 6-8, 8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    # Build adjacency list for Petersen
    adj = [[] for _ in range(10)]
    for i, (u, v) in enumerate(petersen_edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
        
    # Find all non-backtracking closed walks of length < g in Petersen
    # Walk is a sequence of edges e_0, ..., e_{L-1}
    # Non-backtracking means e_{i-1} != e_i (as undirected edges)
    # Closed means start vertex == end vertex
    
    walks = []
    # Start from each vertex, do DFS/BFS for non-backtracking paths
    # We only care about closed walks.
    # Max length is g-1 = 15.
    
    for start_v in range(10):
        # State: (current_v, prev_edge_idx)
        # prev_edge_idx = -1 for start
        stack = [(start_v, -1, [])]
        while stack:
            v, pe, path = stack.pop()
            if len(path) >= g:
                continue
            for u, ei in adj[v]:
                if ei == pe:
                    continue
                new_path = path + [ei]
                if u == start_v and len(new_path) > 0:
                    # Closed walk found
                    # Store the tuple of edge indices
                    walks.append(tuple(new_path))
                # Continue expanding
                stack.append((u, ei, new_path))
                
    # Remove duplicates (walks can be found from different starts or rotations? 
    # Actually, a closed walk is defined by its sequence of edges. 
    # Rotations are different sequences if we fix the start, but for voltage sum, 
    # the sum is the same for any rotation of the same cycle.
    # However, the condition is that *no* closed walk sums to 0.
    # So we just need the set of all unique edge sequences of non-backtracking closed walks.
    unique_walks = set(walks)
    
    # Sort by length, shortest first
    unique_walks = sorted(unique_walks, key=len)
    
    best_n = float('inf')
    best_edges = None
    best_m = None
    
    # Scan m from Moore bound / 10 upwards
    # n = 10 * m
    min_m = (moore_bound + 9) // 10
    max_m = 150 # Upper limit for search
    
    for m in range(min_m, max_m + 1):
        if time.time() > deadline - 5.0:
            break
            
        # Try to find voltage assignment for this m
        # Randomized search
        found = False
        attempts = 0
        max_attempts = 1000
        
        while attempts < max_attempts and time.time() < deadline - 5.0:
            attempts += 1
            # Assign random voltages to edges
            # To ensure symmetry or specific properties, we might fix some voltages?
            # For now, fully random.
            voltages = [random.randint(0, m - 1) for _ in range(15)]
            
            # Check all walks
            valid = True
            for w in unique_walks:
                s = 0
                for ei in w:
                    s = (s + voltages[ei]) % m
                if s == 0:
                    valid = False
                    break
            
            if valid:
                found = True
                break
                
        if found:
            n = 10 * m
            if n < best_n:
                best_n = n
                best_m = m
                best_edges = _build_lift(petersen_edges, voltages, m)
                
    if best_edges is not None:
        return best_n, best_edges
    else:
        # Fallback to L-U-W
        return _construct_luw(k, g)

def _build_lift(base_edges, voltages, m):
    """
    Build the lift of the base graph with given voltages.
    Vertices are (v, offset) where v in 0..9, offset in 0..m-1.
    """
    n_base = 10
    n = n_base * m
    edges = []
    
    # Map (v, offset) to index
    # index = v * m + offset
    
    for i, (u, v) in enumerate(base_edges):
        w = voltages[i]
        # Edge u -> v with voltage w
        # Edge v -> u with voltage -w (or m-w)
        
        for offset in range(m):
            # From u, offset to v, offset + w
            u_idx = u * m + offset
            v_idx = v * m + ((offset + w) % m)
            edges.append((u_idx, v_idx))
            
            # From v, offset to u, offset - w
            # This is the same undirected edge, so we don't add it again
            # if we are iterating over all offsets for the base edge.
            # Wait, the base edge is undirected.
            # In the lift, for each base edge (u,v), we have m edges.
            # The edge connecting (u, a) is to (v, a+w).
            # The edge connecting (v, b) is to (u, b-w).
            # These are the same set of edges.
            # So iterating over base edges and offsets once is sufficient.
            
    return edges

def _construct_luw(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
# EVOLVE-BLOCK-END
