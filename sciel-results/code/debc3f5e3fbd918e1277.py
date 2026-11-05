import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    best_count = 0
    best_edges = []
    
    # Helper: Check if adding edge (u, v) creates a triangle
    # adj is a 16x16 matrix (0/1)
    def can_add_edge(adj, u, v):
        # If edge already exists
        if adj[u][v]:
            return False
        # Check for common neighbor
        for w in range(16):
            if adj[u][w] and adj[v][w]:
                return False
        return True

    # Helper: Add edge to adj
    def add_edge(adj, u, v):
        adj[u][v] = 1
        adj[v][u] = 1

    # Helper: Get list of edges from adj
    def get_edges(adj):
        edges = []
        for u in range(16):
            for v in range(u + 1, 16):
                if adj[u][v]:
                    edges.append([u, v])
        return edges

    # Helper: Check 3-colorability
    def is_3_colorable(adj):
        colors = [-1] * 16
        
        def backtrack(vertex):
            if vertex == 16:
                return True
            
            # MRV Heuristic: Pick the uncolored vertex with the fewest available colors
            # For simplicity in this small graph, we can just iterate 0..15
            # But MRV is better.
            # Let's stick to simple backtracking first, it's fast enough for 16 vertices.
            
            for c in range(3):
                # Check if color c is valid
                valid = True
                for u in range(16):
                    if adj[vertex][u] and colors[u] == c:
                        valid = False
                        break
                if valid:
                    colors[vertex] = c
                    if backtrack(vertex + 1):
                        return True
                    colors[vertex] = -1
            
            return False
        
        return backtrack(0)

    # Helper: Greedily maximize edges
    def maximize_edges(adj):
        # Try to add all possible edges
        changed = True
        while changed:
            changed = False
            for u in range(16):
                for v in range(u + 1, 16):
                    if not adj[u][v]:
                        if can_add_edge(adj, u, v):
                            add_edge(adj, u, v)
                            changed = True
        return adj

    # Main Loop
    random.seed(42) # For reproducibility in initial runs, but we want variety
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        # Generate a candidate graph
        # Strategy: Start with a C5 to ensure non-3-colorability
        # Pick 5 distinct vertices
        c5_vertices = random.sample(range(16), 5)
        adj = [[0]*16 for _ in range(16)]
        
        # Add C5 edges
        for i in range(5):
            u = c5_vertices[i]
            v = c5_vertices[(i + 1) % 5]
            add_edge(adj, u, v)
        
        # Randomly add other edges
        # To make it dense, we try to add many edges
        # We iterate through all pairs and add with some probability if valid
        # Or, we can just try to add ALL valid edges greedily?
        # If we add all valid edges, we get a maximal triangle-free graph.
        # Since it contains C5, it is not 3-colorable.
        # So, just maximizing from C5 should work!
        
        # Wait, if we just maximize from C5, do we get the global optimum?
        # The global optimum might not contain a specific C5.
        # But any non-3-colorable triangle-free graph contains an odd cycle.
        # So it contains a C5, C7, etc.
        # If we start with a random C5 and maximize, we explore the space of maximal triangle-free graphs containing that C5.
        # By trying many random C5s, we should find a dense one.
        
        # Let's try to add edges more aggressively/randomly to escape local maxima of "maximal" graphs?
        # Actually, the set of maximal triangle-free graphs is discrete.
        # Maximizing from C5 leads to a specific maximal graph.
        # Different C5s lead to different maximal graphs.
        
        # Let's just run the maximization.
        adj_max = maximize_edges(adj)
        
        # Verify it's still not 3-colorable (it should be, but just in case)
        if not is_3_colorable(adj_max):
            edges = get_edges(adj_max)
            count = len(edges)
            if count > best_count:
                best_count = count
                best_edges = edges
        else:
            # This shouldn't happen if C5 is present, but if C5 got "broken" somehow? No, edges only added.
            pass
        
        # To improve search, we can also try starting from other structures
        # e.g. C7
        if iterations % 2 == 0:
            c7_vertices = random.sample(range(16), 7)
            adj = [[0]*16 for _ in range(16)]
            for i in range(7):
                u = c7_vertices[i]
                v = c7_vertices[(i + 1) % 7]
                add_edge(adj, u, v)
            adj_max = maximize_edges(adj)
            if not is_3_colorable(adj_max):
                edges = get_edges(adj_max)
                count = len(edges)
                if count > best_count:
                    best_count = count
                    best_edges = edges

    return best_edges

if __name__ == "__main__":
    edges = construct()
    print(len(edges))
