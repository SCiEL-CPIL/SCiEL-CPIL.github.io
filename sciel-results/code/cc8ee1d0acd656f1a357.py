import time
import random
import math

def construct():
    n = 16
    # Clebsch graph construction
    # Vertices are pairs (i, j) where i, j in {0, 1}^2? No, Clebsch is on 16 vertices.
    # It is the complement of the Shrikhande graph? No.
    # Clebsch graph is the folded 5-cube.
    # Vertices can be represented as binary strings of length 4.
    # Two vertices are adjacent if their Hamming distance is 1 or 4? No.
    # In the folded 5-cube, vertices are {0,1}^5 / (00000 ~ 11111). 
    # Actually, simpler: The Clebsch graph is triangle-free, 4-chromatic, 5-regular, 16 vertices.
    # Edges: (u, v) if dist(u,v) in {1, 4} in the 4-cube? 
    # Let's use the standard construction:
    # Vertices 0..15.
    # Adjacency: u ~ v if popcount(u xor v) in {1, 4}?
    # Let's check: 
    # If u=0 (0000), neighbors are 1,2,4,8 (dist 1) and 15 (dist 4).
    # Degree = 5.
    # Is it triangle free?
    # If u~v and v~w, can u~w?
    # This is a known graph. Let's just hardcode the edges or generate them.
    
    def generate_clebsch():
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                d = bin(u ^ v).count('1')
                if d == 1 or d == 4:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    adj = generate_clebsch()
    
    # Helper: get edge count
    def edge_count(adj):
        return sum(bin(a).count('1') for a in adj) // 2

    # Helper: check 3-colorability
    def is_3_colorable(adj):
        colors = [-1] * n
        
        def backtrack(v):
            if v == n:
                return True
            used = 0
            # Get neighbors
            nb_mask = adj[v]
            # Iterate neighbors to find used colors
            # Optimization: Precompute neighbor lists? Or just iterate 0..15
            for u in range(n):
                if nb_mask & (1 << u):
                    if colors[u] != -1:
                        used |= (1 << colors[u])
            
            # Try colors 0, 1, 2
            # Heuristic: Try colors that appear least? Or just 0,1,2
            for c in range(3):
                if not (used & (1 << c)):
                    colors[v] = c
                    if backtrack(v + 1):
                        return True
            colors[v] = -1
            return False
        
        return backtrack(0)

    # Initial state
    current_adj = adj[:]
    current_edges = edge_count(current_adj)
    
    # Check if initial state is valid (it should be)
    if is_3_colorable(current_adj):
        # Fallback if generation is wrong, but Clebsch is 4-chromatic
        pass
        
    best_adj = current_adj[:]
    best_edges = current_edges
    
    # SA Parameters
    start_time = time.time()
    time_limit = 110 # Leave 10s buffer
    T_start = 10.0
    T_end = 0.001
    # Number of iterations per temperature step? Or continuous decay?
    # Let's do fixed number of iterations with linear decay.
    total_iterations = 500000
    current_T = T_start
    
    rng = random.Random(42)
    
    # Precompute all possible pairs
    pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    for i in range(total_iterations):
        if time.time() - start_time > time_limit:
            break
            
        # Update temperature
        current_T = T_start * (T_end / T_start) ** (i / total_iterations)
        
        # Pick random move
        u, v = rng.choice(pairs)
        
        # Determine if edge exists
        if current_adj[u] & (1 << v):
            # Remove edge
            new_adj = current_adj[:]
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
            new_edges = current_edges - 1
            # Removing an edge cannot create a triangle, so triangle-free is preserved.
            # It might make it 3-colorable (if it was 4-chromatic due to that edge? Unlikely, but possible if it was critical).
            # Or it might stay 4-chromatic.
            
            # Check 3-colorability
            valid = not is_3_colorable(new_adj)
            
            # Calculate energy delta
            # We want to maximize edges. So "energy" = -edges.
            # Delta E = E_new - E_old = (-new_edges) - (-current_edges) = current_edges - new_edges = 1
            delta_E = 1
            
            if valid:
                # If valid, we accept if it's better (more edges) or with prob exp(-delta/T)
                # Here new_edges < current_edges, so it's worse in terms of objective.
                # But it might be necessary to escape.
                # However, if it's valid (non-3-colorable), it's a candidate for best.
                if new_edges > best_edges:
                    best_adj = new_adj[:]
                    best_edges = new_edges
                    
                # Acceptance
                # Since it's worse (fewer edges), accept with prob exp(-1/T)
                if rng.random() < math.exp(-delta_E / current_T):
                    current_adj = new_adj
                    current_edges = new_edges
            else:
                # Invalid (3-colorable). 
                # We accept this "bad" state to escape local optima?
                # The directive says: "accepts edge additions ... even if they temporarily make the graph 3-colorable"
                # So yes, we can accept invalid states during search, but we only track valid ones for the final answer.
                # If we accept an invalid state, we are in a 3-colorable graph.
                # From a 3-colorable graph, adding an edge might make it 4-chromatic again.
                # So we should allow transitions to invalid states.
                
                # Acceptance for invalid state:
                # It is "worse" in terms of validity (penalty) and edges (fewer).
                # Let's assign a large penalty for being 3-colorable.
                # Penalty = 100?
                # Delta E = (Penalty + 1) 
                # This makes it very unlikely to accept unless T is high.
                # But at high T, we want to explore.
                penalty = 50
                delta_E = penalty + 1
                if rng.random() < math.exp(-delta_E / current_T):
                    current_adj = new_adj
                    current_edges = new_edges
                    
        else:
            # Add edge
            # Check triangle constraint
            # If adj[u] & adj[v] != 0, adding (u,v) creates triangle. Reject immediately.
            if current_adj[u] & current_adj[v]:
                continue
            
            new_adj = current_adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            new_edges = current_edges + 1
            
            # Check 3-colorability
            valid = not is_3_colorable(new_adj)
            
            if valid:
                # Better in edges (new_edges > current_edges)
                # And valid.
                # Always accept? 
                # Yes, if it's valid and has more edges, it's a strict improvement in objective and constraint.
                current_adj = new_adj
                current_edges = new_edges
                
                if new_edges > best_edges:
                    best_adj = new_adj[:]
                    best_edges = new_edges
            else:
                # Invalid (3-colorable) but more edges.
                # Delta E = -1 (better edges) + Penalty (worse validity)
                # We want to accept this if T is high enough to overcome the penalty.
                penalty = 50
                delta_E = -1 + penalty # Net penalty of 49
                # Wait, if it's 3-colorable, is it really "worse"? 
                # The task requires the *final* graph to be non-3-colorable.
                # During search, being 3-colorable is a barrier.
                # If we accept it, we might add another edge later that makes it 4-chromatic again.
                # So yes, we should allow it.
                
                # Acceptance probability
                if rng.random() < math.exp(-delta_E / current_T):
                    current_adj = new_adj
                    current_edges = new_edges
                    # Note: We do NOT update best_adj here because it's invalid.
                    # But we might return to a valid state later.

    # Final check: ensure best_adj is valid
    # It should be, as we only updated it when valid.
    
    # Convert best_adj back to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
