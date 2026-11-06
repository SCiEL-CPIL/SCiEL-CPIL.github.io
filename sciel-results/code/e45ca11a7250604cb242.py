import random
import itertools

def is_valid_addition(adj, u, v):
    """
    Check if adding edge (u, v) to graph represented by adjacency matrix adj
    maintains K2,3-freeness.
    adj is a list of lists of ints, or a 2D array.
    """
    n = len(adj)
    # Check pairs involving u
    # For any x in N(v), codeg(u, x) must be < 2
    for x in range(n):
        if x == u or x == v:
            continue
        if adj[v][x]:
            # Count common neighbors of u and x
            # Common neighbors are y such that adj[u][y] and adj[x][y]
            # Note: v is in N(u) after addition? No, we are checking BEFORE addition.
            # If we add (u,v), v becomes neighbor of u.
            # So v is a common neighbor of u and x if adj[x][v] is True.
            # We already know adj[v][x] is True (symmetric).
            # So v IS a common neighbor.
            # We need to count OTHER common neighbors.
            count = 0
            for y in range(n):
                if y == u or y == v or y == x:
                    continue
                if adj[u][y] and adj[x][y]:
                    count += 1
            # Total common neighbors will be count + 1 (for v)
            if count + 1 > 2:
                return False
        
    # Check pairs involving v
    for y in range(n):
        if y == u or y == v:
            continue
        if adj[u][y]:
            count = 0
            for z in range(n):
                if z == u or z == v or z == y:
                    continue
                if adj[v][z] and adj[y][z]:
                    count += 1
            # u is a common neighbor of v and y
            if count + 1 > 2:
                return False
                
    return True

def construct():
    n = 20
    best_edges = []
    best_count = -1
    
    # Try multiple random orderings
    for seed in range(50):
        random.seed(seed)
        adj = [[0]*n for _ in range(n)]
        edges = []
        
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if is_valid_addition(adj, u, v):
                adj[u][v] = 1
                adj[v][u] = 1
                edges.append([u, v])
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            
    return best_edges
