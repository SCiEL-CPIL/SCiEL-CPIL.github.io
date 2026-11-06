import random
import numpy as np

def construct():
    n = 40
    m = 401
    
    # Initialize adjacency matrix
    adj = np.zeros((n, n), dtype=bool)
    edges = set()
    
    # 1. Greedy Construction
    # We want to add edges such that we minimize the maximum book size.
    # A good heuristic is to add edges that have few common neighbors.
    # We will iterate through all possible edges, sort by common neighbors, and add them.
    # But we need exactly 401 edges.
    
    # To ensure we don't get stuck in a bad local minimum, we can use a randomized greedy.
    # Or, we can just do a local search from a random start, which is simpler and often effective for N=40.
    
    # Let's go with Local Search from a random start.
    # It's a different family: "Stochastic Local Search" vs "Structured Perturbation".
    
    random.seed(123) # Deterministic seed for reproducibility of the search path
    
    # Start with a random graph of 401 edges
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_possible)
    initial_edges = all_possible[:m]
    
    for u, v in initial_edges:
        adj[u, v] = True
        adj[v, u] = True
        edges.add((u, v))
    
    def get_max_book(adj, edges):
        max_book = 0
        for u, v in edges:
            # Count common neighbors
            # adj[u] & adj[v] gives boolean vector of common neighbors
            # sum gives count
            common = np.sum(adj[u] & adj[v])
            if common > max_book:
                max_book = common
        return max_book
    
    current_max_book = get_max_book(adj, edges)
    
    # Local Search
    # We will try to swap edges to reduce the max book.
    # Number of iterations
    num_iterations = 20000
    
    for _ in range(num_iterations):
        # Pick a random edge to remove
        if not edges:
            break
        u, v = random.choice(list(edges))
        
        # Pick a random non-edge to add
        # We need to find a non-edge efficiently.
        # Since n=40, total possible edges = 780.
        # We can just pick a random pair and check if it's in edges.
        # But rejection sampling might be slow if graph is dense.
        # 401/780 is ~50% density. Rejection is fine.
        
        found = False
        while not found:
            x, y = random.sample(range(n), 2)
            if x > y:
                x, y = y, x
            if (x, y) not in edges:
                found = True
        
        # Perform swap: remove (u,v), add (x,y)
        # Check validity: (x,y) is not in edges. (u,v) is in edges.
        # Also (x,y) must not be same as (u,v) (guaranteed by non-edge check).
        
        # Tentatively apply
        adj[u, v] = False
        adj[v, u] = False
        edges.discard((u, v))
        
        adj[x, y] = True
        adj[y, x] = True
        edges.add((x, y))
        
        new_max_book = get_max_book(adj, edges)
        
        if new_max_book <= current_max_book:
            current_max_book = new_max_book
            # Keep the change
        else:
            # Revert
            adj[x, y] = False
            adj[y, x] = False
            edges.discard((x, y))
            
            adj[u, v] = True
            adj[v, u] = True
            edges.add((u, v))
            
            # Optionally, accept with some probability (Simulated Annealing)
            # For now, pure hill climbing.
            
    # Convert to list of lists
    result = [list(e) for e in edges]
    return result
