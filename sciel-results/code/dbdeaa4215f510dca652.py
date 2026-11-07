"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin for 60s limit
    
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = graph['n']
    
    # Build adjacency list for quick neighbor lookup and edge index mapping
    # adj[u][v] = index of edge (u,v) or (v,u) in the edges list
    adj = {i: {} for i in range(n_vertices)}
    for i, (u, v) in enumerate(edges):
        adj[u][v] = i
        adj[v][u] = i
        
    # Initial state: Try to reproduce the seed's best random sample or a good heuristic
    # The seed uses np.random.default_rng(3) to sample budget times.
    # We can just start with the best of a few random samples or the alternating pattern.
    # Let's start with the alternating pattern from the current program as it's a known valid start.
    # Or better, start with the seed's best result if we can simulate it quickly?
    # The seed samples 32 times. Let's just do a quick random search to find a valid start.
    
    def compute_radius(signs):
        """Compute spectral radius of the signed adjacency matrix."""
        A = np.zeros((n_vertices, n_vertices))
        for i, (u, v) in enumerate(edges):
            A[u, v] = signs[i]
            A[v, u] = signs[i]
        # Spectral radius is the largest absolute eigenvalue
        # Since A is symmetric, eigenvalues are real
        # np.linalg.eigvalsh is faster and more stable for symmetric matrices
        try:
            eigvals = np.linalg.eigvalsh(A)
            return np.max(np.abs(eigvals))
        except:
            return np.inf

    def is_valid(signs):
        """Check if the lift is connected."""
        # The lift is connected iff the product of signs around some cycle is -1.
        # Equivalent: The signed graph is not bipartite in the sense of the lift?
        # Actually, the 2-lift is connected iff the sign assignment is not a "coboundary" 
        # i.e., there is no vertex labeling s: V -> {+1, -1} such that sign(u,v) = s(u)s(v) for all edges.
        # If such an s exists, the lift splits into two components (one where lift_vertex = v*s(v), etc).
        # So we check if the system of equations s(u)*s(v) = sign(u,v) is consistent.
        # This is a 2-SAT / bipartite coloring problem on the signed graph.
        
        # BFS/DFS to check consistency
        color = [-1] * n_vertices # 0 for +1, 1 for -1 relative to start node? 
        # Let's use 0 and 1 for the two sides of the lift relative to a root.
        # If we assign color 0 to node 0, then for edge (u,v) with sign s:
        # color(v) = color(u) if s == +1
        # color(v) = 1 - color(u) if s == -1
        # If we encounter a conflict, it's consistent (connected lift).
        # If we finish without conflict, it's inconsistent (disconnected lift).
        
        color[0] = 0
        queue = [0]
        visited = [False] * n_vertices
        visited[0] = True
        
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                edge_idx = adj[u][v]
                s = signs[edge_idx]
                expected_color = color[u] if s == 1 else 1 - color[u]
                if not visited[v]:
                    color[v] = expected_color
                    visited[v] = True
                    queue.append(v)
                else:
                    if color[v] != expected_color:
                        return True # Conflict found -> Consistent -> Connected Lift
        
        return False # No conflict -> Inconsistent -> Disconnected Lift

    # Generate initial sign vector
    # Strategy: Start with all +1 (invalid) and flip random edges until valid?
    # Or start with the alternating matching pattern which is known to be valid for m=4,5,7,9?
    # Let's verify the alternating pattern validity.
    # For m=4: Pattern -1, +1, -1, +1 on matching. Cycle 0-1-m+1-m-0.
    # Edges: (0,1) top (+1), (1, m+1) match (+1, idx 1 odd), (m+1, m) bottom (+1), (m, 0) match (-1, idx 0 even).
    # Product: 1*1*1*(-1) = -1. Valid.
    # For m=5: Cycle 0-1-m+1-m-0.
    # Edges: (0,1) top (+1), (1, m+1) match (+1), (m+1, m) bottom (+1), (m, 0) match (-1).
    # Product -1. Valid.
    # It seems the alternating pattern is always valid for these prisms because it creates a -1 product on the 4-cycles.
    
    signs = np.ones(n_edges, dtype=int)
    for i, (u, v) in enumerate(edges):
        if abs(u - v) == m:
            top = u if u < m else v
            if top % 2 == 0:
                signs[i] = -1
            else:
                signs[i] = +1
        else:
            signs[i] = +1
            
    if not is_valid(signs):
        # Fallback: random search for valid start
        rng = np.random.default_rng(3)
        for _ in range(1000):
            signs = rng.choice([-1, 1], size=n_edges)
            if is_valid(signs):
                break
        if not is_valid(signs):
             # Should not happen, but if it does, return something
             return signs

    best_radius = compute_radius(signs)
    best_signs = signs.copy()
    
    # Local Search: Hill Climbing
    # We flip one edge at a time if it reduces the radius.
    # To speed up, we can precompute the effect of a flip?
    # Flipping sign of edge (u,v) changes A by adding 2*sign*(e_u e_v^T + e_v e_u^T).
    # This is a rank-2 update. We could use Sherman-Morrison, but eigenvalues don't update simply.
    # Given small n (max 18 vertices), full eigvalsh is fast enough (~18x18).
    
    current_signs = best_signs.copy()
    current_radius = best_radius
    
    no_improvement_count = 0
    max_no_improvement = 50 # Restart or stop if stuck
    
    while time.time() < deadline:
        improved = False
        # Shuffle edge order to avoid bias
        perm = np.random.permutation(n_edges)
        
        for idx in perm:
            if time.time() > deadline:
                break
                
            # Flip sign at idx
            old_sign = current_signs[idx]
            new_sign = -old_sign
            current_signs[idx] = new_sign
            
            # Check validity
            if not is_valid(current_signs):
                current_signs[idx] = old_sign
                continue
            
            # Compute new radius
            new_radius = compute_radius(current_signs)
            
            if new_radius < current_radius:
                current_radius = new_radius
                best_signs = current_signs.copy()
                best_radius = current_radius
                improved = True
                no_improvement_count = 0
            else:
                # Revert
                current_signs[idx] = old_sign
        
        if not improved:
            no_improvement_count += 1
            if no_improvement_count >= max_no_improvement:
                # Try a random restart from best_signs with some noise
                noise_count = max(1, n_edges // 10)
                current_signs = best_signs.copy()
                flip_indices = np.random.choice(n_edges, size=noise_count, replace=False)
                for idx in flip_indices:
                    current_signs[idx] *= -1
                    # Ensure validity after noise? 
                    # If invalid, revert this specific flip or restart
                    if not is_valid(current_signs):
                        current_signs[idx] *= -1 # revert
                        # If still invalid, break and try another restart
                        break
                no_improvement_count = 0
                current_radius = compute_radius(current_signs) if is_valid(current_signs) else best_radius
                if not is_valid(current_signs):
                    current_signs = best_signs.copy()
                    current_radius = best_radius
                    
    return best_signs
# EVOLVE-BLOCK-END
