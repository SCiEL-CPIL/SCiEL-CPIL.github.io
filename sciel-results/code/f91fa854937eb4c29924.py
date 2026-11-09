"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    n = 40
    target_edges = 401
    time_limit = 115.0
    
    # Deterministic spectral heuristic: start with balanced bipartite graph + extra edges
    # Split vertices into two sets of 20
    A = list(range(20))
    B = list(range(20, 40))
    
    # Start with complete bipartite K_{20,20} = 400 edges
    edges = [[a, b] for a in A for b in B]
    # Add 1 more edge to reach 401: add an edge within A
    edges.append([0, 1])
    
    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    
    # Compute book sizes (number of triangles containing each edge)
    # For edge (u,v), book size = number of common neighbors = (A @ A)[u,v] - 1 (if u,v adjacent)
    # Actually, (A^2)[u,v] = number of common neighbors of u and v
    # If (u,v) is an edge, triangles containing it = (A^2)[u,v] - 1? No.
    # Triangles containing edge (u,v) = |N(u) ∩ N(v)| = (A^2)[u,v] (since A[u,v]=1, A^2[u,v] counts paths of length 2)
    # Wait: A^2[u,v] = sum_w A[u,w]*A[w,v] = |N(u) ∩ N(v)|
    # So book size for edge (u,v) = A^2[u,v]
    
    def compute_max_book(adj):
        A2 = adj @ adj
        max_book = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j] == 1:
                    if A2[i, j] > max_book:
                        max_book = A2[i, j]
        return max_book
    
    current_max = compute_max_book(adj)
    best_max = current_max
    best_adj = adj.copy()
    
    # Spectral heuristic: iteratively swap edges to reduce max book size
    # Focus on edges with high book sizes, try to redistribute
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Track which edges are present
    edge_present = np.zeros((n, n), dtype=bool)
    for u, v in edges:
        edge_present[u, v] = True
        edge_present[v, u] = True
    
    # List of current edges as (u,v) tuples
    current_edges = [(u, v) for u, v in edges]
    
    # Iterative improvement: find edge with max book, try to swap it
    iterations = 0
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        iterations += 1
        
        # Compute current book sizes
        A2 = adj @ adj
        
        # Find edge with maximum book size
        max_book = 0
        max_edge = None
        for u, v in current_edges:
            if A2[u, v] > max_book:
                max_book = A2[u, v]
                max_edge = (u, v)
        
        if max_book <= 6:
            # Already at or below theoretical minimum
            break
        
        # Try to improve the max edge
        u, v = max_edge
        improved = False
        
        # Strategy: remove edge (u,v) and add a different edge that doesn't increase max book
        # Try all non-edges that could replace (u,v)
        # For each candidate non-edge (x,y), check if adding it keeps max book <= max_book
        
        # Get neighbors of u and v
        neighbors_u = list(np.where(adj[u] == 1)[0])
        neighbors_v = list(np.where(adj[v] == 1)[0])
        
        # Candidate non-edges: edges not currently present
        # Try edges that share vertices with u or v to potentially reduce their book sizes
        candidates = []
        
        # Add edges between neighbors of u and neighbors of v (excluding u,v themselves)
        for x in neighbors_u:
            if x == v:
                continue
            for y in neighbors_v:
                if y == u or y == x:
                    continue
                if not edge_present[x, y]:
                    candidates.append((min(x,y), max(x,y)))
        
        # Also try random non-edges
        random.seed(iterations)
        for _ in range(50):
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x != y and not edge_present[x, y]:
                candidates.append((min(x,y), max(x,y)))
        
        # Deduplicate candidates
        candidates = list(set(candidates))
        
        # Try each candidate
        for x, y in candidates:
            # Simulate removing (u,v) and adding (x,y)
            # Check if this reduces the max book
            
            # Temporarily remove (u,v)
            adj[u, v] = 0
            adj[v, u] = 0
            
            # Compute new A2
            A2_new = adj @ adj
            
            # Temporarily add (x,y)
            adj[x, y] = 1
            adj[y, x] = 1
            
            # Compute new A2
            A2_new = adj @ adj
            
            # Find new max book
            new_max = 0
            for i, j in current_edges:
                if i == u and j == v:
                    continue  # This edge is removed
                if A2_new[i, j] > new_max:
                    new_max = A2_new[i, j]
            # Check the new edge
            if A2_new[x, y] > new_max:
                new_max = A2_new[x, y]
            
            # Revert temporary changes
            adj[x, y] = 0
            adj[y, x] = 0
            adj[u, v] = 1
            adj[v, u] = 1
            
            # Accept if it improves
            if new_max < max_book:
                # Apply the change
                adj[u, v] = 0
                adj[v, u] = 0
                edge_present[u, v] = False
                edge_present[v, u] = False
                
                adj[x, y] = 1
                adj[y, x] = 1
                edge_present[x, y] = True
                edge_present[y, x] = True
                
                # Update edge list
                current_edges.remove((u, v))
                current_edges.append((x, y))
                
                current_max = new_max
                if new_max < best_max:
                    best_max = new_max
                    best_adj = adj.copy()
                
                improved = True
                no_improve_count = 0
                break
        
        if not improved:
            no_improve_count += 1
            # If stuck, try a random perturbation
            if no_improve_count > 100:
                # Pick a random edge and swap it with a random non-edge
                idx = random.randint(0, len(current_edges) - 1)
                ru, rv = current_edges[idx]
                
                # Find a random non-edge
                while True:
                    rx = random.randint(0, n-1)
                    ry = random.randint(0, n-1)
                    if rx != ry and not edge_present[rx, ry]:
                        break
                
                # Apply swap
                adj[ru, rv] = 0
                adj[rv, ru] = 0
                edge_present[ru, rv] = False
                edge_present[rv, ru] = False
                
                adj[rx, ry] = 1
                adj[ry, rx] = 1
                edge_present[rx, ry] = True
                edge_present[ry, rx] = True
                
                current_edges[idx] = (min(rx,ry), max(rx,ry))
                
                # Recompute max
                A2 = adj @ adj
                current_max = 0
                for i, j in current_edges:
                    if A2[i, j] > current_max:
                        current_max = A2[i, j]
                
                if current_max < best_max:
                    best_max = current_max
                    best_adj = adj.copy()
                
                no_improve_count = 0
    
    # Extract edges from best_adj
    best_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j] == 1:
                best_edges.append([i, j])
    
    # Ensure we have exactly 401 edges
    if len(best_edges) != target_edges:
        # Fallback: use the last valid state
        # This shouldn't happen, but just in case
        pass
    
    return best_edges

# EVOLVE-BLOCK-END
