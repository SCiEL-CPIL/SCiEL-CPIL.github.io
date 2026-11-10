import time
import random
from itertools import combinations

def construct():
    """
    Find a triangle-free graph on 16 vertices with chromatic number >= 4,
    maximizing the number of edges.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    best_edges_count = 0
    best_edges_list = None
    
    # Grötzsch graph: 11 vertices, 20 edges, triangle-free, 4-chromatic
    # Vertices 0..10
    groetzsch_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # C5
        (5, 1), (5, 4),
        (6, 0), (6, 2),
        (7, 1), (7, 3),
        (8, 2), (8, 4),
        (9, 3), (9, 0),
        (10, 5), (10, 6), (10, 7), (10, 8), (10, 9)
    ]
    
    seed_edge_set = set(groetzsch_edges)
    
    # Precompute all possible edges
    all_edges = list(combinations(range(n), 2))
    
    def count_edges(adj):
        return sum(adj[i][j] for i in range(n) for j in range(i+1, n))
    
    def is_triangle_free_add(adj, u, v):
        """Check if adding edge (u,v) to adj creates a triangle."""
        # We assume adj[u][v] is 0
        # Check if there exists w such that adj[u][w] and adj[v][w]
        for w in range(n):
            if w != u and w != v:
                if adj[u][w] and adj[v][w]:
                    return False
        return True

    def greedy_fill(adj, seed_set):
        """
        Greedily add all possible edges that don't create triangles.
        Returns the new adjacency matrix.
        """
        new_adj = [row[:] for row in adj]
        
        # Get list of non-edges
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not new_adj[i][j]:
                    non_edges.append((i, j))
        
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if is_triangle_free_add(new_adj, u, v):
                new_adj[u][v] = 1
                new_adj[v][u] = 1
                
        return new_adj

    # Initial best: just the seed
    init_adj = [[0]*n for _ in range(n)]
    for u, v in groetzsch_edges:
        init_adj[u][v] = 1
        init_adj[v][u] = 1
        
    current_adj = greedy_fill(init_adj, seed_edge_set)
    best_edges_count = count_edges(current_adj)
    best_edges_list = [[u, v] for u in range(n) for v in range(u+1, n) if current_adj[u][v]]
    
    # Local Search / Simulated Annealing-like improvement
    # Strategy: Try to swap edges or remove an edge and refill to find a denser graph.
    
    while time.time() < deadline:
        # 1. Try to add any missing edge (should be none if greedy_fill was exhaustive, 
        #    but maybe we can try different orders? No, greedy_fill adds ALL possible.
        #    So current_adj is a maximal triangle-free supergraph of the seed.
        #    To improve, we must remove some edges and re-add others.
        
        # Randomly select a non-seed edge to remove
        removable_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if current_adj[i][j] and (i, j) not in seed_edge_set:
                    removable_edges.append((i, j))
        
        if not removable_edges:
            # If we can't remove any non-seed edge, we are at a very constrained state.
            # Try restarting with a different random seed for the greedy phase?
            # Or just break if we've been stuck for a while.
            break
            
        u, v = random.choice(removable_edges)
        
        # Tentatively remove (u, v)
        current_adj[u][v] = 0
        current_adj[v][u] = 0
        
        # Refill greedily
        new_adj = greedy_fill(current_adj, seed_edge_set)
        new_count = count_edges(new_adj)
        
        if new_count > best_edges_count:
            best_edges_count = new_count
            best_edges_list = [[i, j] for i in range(n) for j in range(i+1, n) if new_adj[i][j]]
            current_adj = new_adj
        else:
            # Revert
            # We need to restore the edge (u,v) and the state before refill?
            # Actually, greedy_fill modifies the passed matrix.
            # So we should have worked on a copy.
            # Let's re-implement the loop to work on copies to allow backtracking.
            pass
            
    # The above loop logic was slightly flawed with in-place modification.
    # Let's do a proper restart-based search which is safer and often effective for small n.
    
    best_edges_count = 0
    best_edges_list = None
    
    # Run multiple independent randomized greedy searches
    # Each search starts from the seed and fills greedily with a random order.
    
    # To diversify, we can also start from slightly different seeds?
    # No, we must keep the seed to guarantee 4-chromaticity.
    # But we can randomize the order of edge addition.
    
    for i in range(5000): # Max iterations, will stop earlier due to time
        if time.time() > deadline:
            break
            
        adj = [[0]*n for _ in range(n)]
        for u, v in groetzsch_edges:
            adj[u][v] = 1
            adj[v][u] = 1
            
        # Greedy fill
        non_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    non_edges.append((u, v))
        
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if is_triangle_free_add(adj, u, v):
                adj[u][v] = 1
                adj[v][u] = 1
                
        curr_count = count_edges(adj)
        
        if curr_count > best_edges_count:
            best_edges_count = curr_count
            best_edges_list = [[u, v] for u in range(n) for v in range(u+1, n) if adj[u][v]]
            
            # If we found a very good one, we might want to do local search from here
            # but for now, just record it.
            
    # If we didn't find anything (should not happen), return seed
    if best_edges_list is None:
        best_edges_list = [[u, v] for u, v in groetzsch_edges]
        
    return best_edges_list
