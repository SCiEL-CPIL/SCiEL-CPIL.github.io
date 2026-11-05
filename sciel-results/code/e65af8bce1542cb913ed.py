"""Find maximum-edge triangle-free non-3-colorable graph on 16 vertices."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    n = 16
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Generate Clebsch graph (triangle-free, 4-chromatic, 40 edges)
    def get_clebsch_adj():
        vectors = [i for i in range(32) if bin(i).count('1') % 2 == 0]
        adj = [0] * n
        for u_idx in range(n):
            for v_idx in range(u_idx + 1, n):
                xor = vectors[u_idx] ^ vectors[v_idx]
                if bin(xor).count('1') == 4:
                    adj[u_idx] |= (1 << v_idx)
                    adj[v_idx] |= (1 << u_idx)
        return adj
    
    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2
    
    def is_triangle_free(adj):
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    if (adj[i] & adj[j]) != 0:
                        return False
        return True
    
    def is_3_colorable(adj):
        """Backtracking 3-colorability check with bitmasks."""
        colors = [-1] * n
        
        def backtrack(v):
            if v == n:
                return True
            # Try colors 0, 1, 2
            for c in range(3):
                # Check if color c is available for v
                # Get neighbors of v with index < v (since we color in order)
                # Actually, we need to check all already colored neighbors
                valid = True
                # Check neighbors
                nb_mask = adj[v]
                for u in range(v):
                    if nb_mask & (1 << u):
                        if colors[u] == c:
                            valid = False
                            break
                if valid:
                    colors[v] = c
                    if backtrack(v + 1):
                        return True
                    colors[v] = -1
            return False
        
        return backtrack(0)
    
    def creates_triangle(adj, u, v):
        """Check if adding edge (u,v) creates a triangle."""
        # If there exists w such that adj[u] has w and adj[v] has w
        if (adj[u] & adj[v]) != 0:
            return True
        return False
    
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    # Start with Clebsch graph
    best_adj = get_clebsch_adj()
    best_count = count_edges(best_adj)
    
    # Verify it's 4-chromatic (not 3-colorable)
    if is_3_colorable(best_adj):
        # Should not happen for Clebsch, but just in case
        pass
    
    # Simulated annealing approach
    # State: adjacency list (bitmasks)
    # Moves: add an edge, remove an edge, or swap an edge
    # Objective: maximize edges, subject to triangle-free and not 3-colorable
    
    current_adj = [row[:] for row in best_adj]
    current_count = best_count
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    cooling = 0.995
    
    # List of available edges (not currently in graph)
    def get_available_edges(adj):
        avail = []
        for u, v in all_edges:
            if not (adj[u] & (1 << v)):
                avail.append((u, v))
        return avail
    
    # Main annealing loop
    T = T_start
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            # Restart with fresh random perturbation
            current_adj = [row[:] for row in best_adj]
            current_count = best_count
            no_improve = 0
            T = T_start * 0.5
        
        # Pick a random move
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add an edge
            avail = get_available_edges(current_adj)
            if not avail:
                continue
            u, v = random.choice(avail)
            if creates_triangle(current_adj, u, v):
                continue
            
            # Apply move
            add_edge(current_adj, u, v)
            new_count = current_count + 1
            is_3col = is_3_colorable(current_adj)
            
            # Evaluate
            if not is_3col:
                # Valid: triangle-free and not 3-colorable
                if new_count > best_count:
                    best_count = new_count
                    best_adj = [row[:] for row in current_adj]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Invalid: became 3-colorable. Accept with probability exp(-1/T)
                if random.random() < (1.0 / T):
                    no_improve += 1
                else:
                    # Reject: undo
                    remove_edge(current_adj, u, v)
                    no_improve += 1
            
        elif move_type < 0.7:
            # Try to remove an edge
            if current_count == 0:
                continue
            # Pick a random edge in current graph
            # Find all edges
            edges_in_graph = []
            for u, v in all_edges:
                if current_adj[u] & (1 << v):
                    edges_in_graph.append((u, v))
            if not edges_in_graph:
                continue
            u, v = random.choice(edges_in_graph)
            
            remove_edge(current_adj, u, v)
            new_count = current_count - 1
            
            # Removing an edge might make it 3-colorable or not
            # If it was not 3-colorable, removing an edge keeps it not 3-colorable? No, removing edges can make it 3-colorable
            is_3col = is_3_colorable(current_adj)
            
            if not is_3col:
                # Still valid
                if new_count > best_count:
                    best_count = new_count
                    best_adj = [row[:] for row in current_adj]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Became 3-colorable (bad)
                if random.random() < (1.0 / T):
                    no_improve += 1
                else:
                    # Reject: undo
                    add_edge(current_adj, u, v)
                    no_improve += 1
            
        else:
            # Edge swap: remove one edge, add another
            # First, find an edge to remove
            edges_in_graph = []
            for u, v in all_edges:
                if current_adj[u] & (1 << v):
                    edges_in_graph.append((u, v))
            if not edges_in_graph:
                continue
            
            # Pick random edge to remove
            u1, v1 = random.choice(edges_in_graph)
            remove_edge(current_adj, u1, v1)
            
            # Find an edge to add
            avail = get_available_edges(current_adj)
            if not avail:
                add_edge(current_adj, u1, v1)
                continue
            
            # Filter out edges that would create triangles
            valid_adds = []
            for u2, v2 in avail:
                if not creates_triangle(current_adj, u2, v2):
                    valid_adds.append((u2, v2))
            
            if not valid_adds:
                add_edge(current_adj, u1, v1)
                continue
            
            u2, v2 = random.choice(valid_adds)
            add_edge(current_adj, u2, v2)
            
            new_count = current_count  # Same number of edges
            is_3col = is_3_colorable(current_adj)
            
            if not is_3col:
                if new_count > best_count:
                    best_count = new_count
                    best_adj = [row[:] for row in current_adj]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                if random.random() < (1.0 / T):
                    no_improve += 1
                else:
                    # Reject: undo both
                    remove_edge(current_adj, u2, v2)
                    add_edge(current_adj, u1, v1)
                    no_improve += 1
        
        # Cool down
        T *= cooling
        if T < T_end:
            T = T_end
    
    # Final local search: greedily add edges if possible
    final_adj = [row[:] for row in best_adj]
    final_count = best_count
    
    # Try to add more edges greedily
    improved = True
    while improved and time.time() < deadline:
        improved = False
        avail = get_available_edges(final_adj)
        for u, v in avail:
            if creates_triangle(final_adj, u, v):
                continue
            add_edge(final_adj, u, v)
            if not is_3_colorable(final_adj):
                final_count += 1
                if final_count > best_count:
                    best_count = final_count
                    best_adj = [row[:] for row in final_adj]
                improved = True
            else:
                remove_edge(final_adj, u, v)
                break
    
    # Convert to edge list
    result = []
    for u, v in all_edges:
        if best_adj[u] & (1 << v):
            result.append([u, v])
    
    return result
