import time
import random

def construct():
    """
    Find a triangle-free graph on 16 vertices with chromatic number >= 4,
    maximizing the number of edges.
    
    Returns:
        list of edges [u, v]
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    best_edges = None
    best_score = -1
    
    # Grötzsch graph edges (vertices 0-10)
    groetzsch_edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
        [5, 1], [5, 4],
        [6, 0], [6, 2],
        [7, 1], [7, 3],
        [8, 2], [8, 4],
        [9, 3], [9, 0],
        [10, 5], [10, 6], [10, 7], [10, 8], [10, 9]
    ]
    
    # Initialize adjacency matrix
    base_adj = [[0]*n for _ in range(n)]
    for u, v in groetzsch_edges:
        base_adj[u][v] = 1
        base_adj[v][u] = 1
        
    def is_3_colorable(adj):
        """
        Returns True if the graph is 3-colorable, False otherwise.
        """
        color = [-1]*n
        
        # Heuristic: order vertices by degree descending
        # We can just use 0..n-1 for simplicity, or sort indices
        # Sorting might add overhead, but reduce search space
        # Let's stick to 0..n-1 for speed in Python, n is small
        
        def backtrack(idx):
            if idx == n:
                return True
            v = idx
            # Check available colors
            used = set()
            for u in range(n):
                if adj[u][v] and color[u] != -1:
                    used.add(color[u])
            
            # Try colors
            for c in range(3):
                if c not in used:
                    color[v] = c
                    if backtrack(idx+1):
                        return True
                    color[v] = -1
            return False
            
        return backtrack(0)
        
    def can_add_edge(adj, u, v):
        """
        Check if adding edge (u, v) preserves triangle-freeness.
        """
        for w in range(n):
            if w != u and w != v:
                if adj[u][w] and adj[v][w]:
                    return False
        return True
        
    # Get all non-edges
    all_non_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if base_adj[i][j] == 0:
                all_non_edges.append((i, j))
                
    # Randomized Greedy Search
    # We will run multiple restarts until time runs out
    
    # To allow for more diversity, we can also randomly shuffle the base graph's isolated vertices?
    # No, the base structure is fixed.
    
    while time.time() < deadline:
        # Shuffle non-edges
        edges_to_try = all_non_edges[:]
        random.shuffle(edges_to_try)
        
        # Copy base graph
        adj = [row[:] for row in base_adj]
        
        # Try to add edges
        for u, v in edges_to_try:
            if adj[u][v]:
                continue
            if can_add_edge(adj, u, v):
                # Tentatively add
                adj[u][v] = 1
                adj[v][u] = 1
                # Check 3-colorability
                if not is_3_colorable(adj):
                    # Keep edge
                    pass
                else:
                    # Revert
                    adj[u][v] = 0
                    adj[v][u] = 0
                    
        # Count edges
        count = sum(adj[i][j] for i in range(n) for j in range(i+1, n))
        if count > best_score:
            best_score = count
            best_edges = [ [i, j] for i in range(n) for j in range(i+1, n) if adj[i][j] ]
            
    # Fallback
    if best_edges is None:
        return groetzsch_edges
        
    return best_edges
