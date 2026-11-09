"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # 1. Generate the Grötzsch graph (Mycielski of C5) as the initial 4-chromatic core.
    # Vertices 0..10 form the Grötzsch graph. Vertices 11..15 are isolated.
    def get_groetzsch_edges():
        edges = []
        # C5 cycle: 0-1-2-3-4-0
        for i in range(5):
            edges.append([i, (i + 1) % 5])
        # Mycielski construction:
        # New vertices 5..9 connect to neighbors of 0..4 respectively
        for i in range(5):
            neighbors = [(i + 1) % 5, (i - 1) % 5]
            for n in neighbors:
                edges.append([5 + i, n])
        # New vertex 10 connects to 5..9
        for i in range(5):
            edges.append([10, 5 + i])
        return edges

    best_edges = get_groetzsch_edges()
    best_count = len(best_edges)
    
    # Helper to check if adding edge [u, v] maintains triangle-freeness
    def is_triangle_free(adj, u, v):
        # Check if u and v share a common neighbor
        # adj is a list of sets
        return len(adj[u] & adj[v]) == 0

    # Helper to check if graph is 4-chromatic (not 3-colorable) using bitsets for speed
    def is_not_3_colorable(adj):
        # Use bitmasks for colors: 0b001, 0b010, 0b100
        # colors[v] = 0 means uncolored, 1,2,3 for colors
        colors = [0] * 16
        
        def backtrack(v):
            if v == 16:
                return True
            
            # Find available colors based on neighbors
            available = 7  # 0b111, all three colors available
            for n in adj[v]:
                if colors[n]:
                    available &= ~(1 << (colors[n] - 1))
            
            if available == 0:
                return False
            
            # Try each available color
            for c in range(1, 4):
                if available & (1 << (c - 1)):
                    colors[v] = c
                    if backtrack(v + 1):
                        return True
                    colors[v] = 0
            return False
            
        return not backtrack(0)

    # Helper to add edge if valid
    def try_add_edge(adj, edge_set, u, v):
        if u == v or u > v:
            u, v = v, u
        if (u, v) in edge_set:
            return False
        if not is_triangle_free(adj, u, v):
            return False
        
        # Tentatively add
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
        
        if is_not_3_colorable(adj):
            return True
        else:
            # Revert
            adj[u].remove(v)
            adj[v].remove(u)
            edge_set.remove((u, v))
            return False

    # Helper to score an edge for ordering (prefer edges that connect low-degree vertices)
    def edge_score(adj, u, v):
        # Prefer edges that connect vertices with fewer existing edges
        # This helps maintain flexibility for future additions
        return (len(adj[u]) + len(adj[v]))

    # Main search loop
    while time.time() < deadline:
        # Start from best known or random restart from Grötzsch
        if random.random() < 0.3:
            # Restart from Grötzsch
            base_edges = get_groetzsch_edges()
        else:
            # Restart from current best
            base_edges = list(best_edges)
        
        # Build adjacency and edge set
        adj = [set() for _ in range(16)]
        edge_set = set()
        for e in base_edges:
            u, v = e
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u, v), max(u, v)))
        
        current_edges = list(base_edges)
        
        # Greedy addition phase with smart ordering
        # Generate all possible edges not currently present
        possible_edges = []
        for i in range(16):
            for j in range(i + 1, 16):
                if (i, j) not in edge_set:
                    possible_edges.append((i, j))
        
        # Sort by edge score (prefer edges connecting low-degree vertices)
        # Add some randomness to break ties
        random.shuffle(possible_edges)
        possible_edges.sort(key=lambda e: edge_score(adj, e[0], e[1]) + random.random() * 2)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add one edge
            for u, v in possible_edges:
                if (u, v) in edge_set:
                    continue
                if try_add_edge(adj, edge_set, u, v):
                    current_edges.append([u, v])
                    improved = True
                    if len(current_edges) > best_count:
                        best_count = len(current_edges)
                        best_edges = list(current_edges)
                    # Re-sort remaining possible edges after addition
                    possible_edges = [(i, j) for i in range(16) for j in range(i + 1, 16) if (i, j) not in edge_set]
                    random.shuffle(possible_edges)
                    possible_edges.sort(key=lambda e: edge_score(adj, e[0], e[1]) + random.random() * 2)
                    break # Restart scanning possible edges after an addition
        
        # If we found a better solution, update best
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = list(current_edges)
        
    return best_edges
# EVOLVE-BLOCK-END
