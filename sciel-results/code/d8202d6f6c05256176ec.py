"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    best_n = 5
    best_edges = []
    
    # Try to find a 20-vertex graph with no regular induced subgraph on 6+ vertices
    # We'll use a greedy randomized approach
    
    def is_regular_subgraph(vertices, adj):
        """Check if the induced subgraph on 'vertices' is regular."""
        if len(vertices) < 6:
            return False
        v_set = set(vertices)
        degrees = []
        for v in vertices:
            deg = sum(1 for u in v_set if u != v and adj[v][u])
            degrees.append(deg)
        return all(d == degrees[0] for d in degrees)
    
    def has_regular_induced_subgraph(adj, n):
        """Check if any subset of 6+ vertices induces a regular subgraph."""
        for size in range(6, n + 1):
            for subset in combinations(range(n), size):
                if is_regular_subgraph(subset, adj):
                    return True
        return False
    
    def check_graph(adj, n):
        """Check if graph has no regular induced subgraph on 6+ vertices."""
        return not has_regular_induced_subgraph(adj, n)
    
    # Greedy construction with randomization
    def try_construct(n, max_attempts=500):
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Build graph greedily
            adj = [[0] * n for _ in range(n)]
            edges = []
            
            # Randomly add edges
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i][j] = 1
                        adj[j][i] = 1
                        edges.append([i, j])
            
            if check_graph(adj, n):
                return edges
        
        return None
    
    # Try larger graphs first
    for n in [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
        if time.time() > deadline:
            break
        
        # Use more attempts for smaller n since they're easier
        max_attempts = 2000 if n <= 15 else 500
        
        edges = try_construct(n, max_attempts)
        if edges is not None:
            return {'n': n, 'edges': edges}
    
    # Fallback: return a valid small graph
    # Empty graph on 5 vertices
    return {'n': 5, 'edges': []}
# EVOLVE-BLOCK-END
