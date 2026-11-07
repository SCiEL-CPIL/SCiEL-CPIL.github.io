"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift
    
    deadline = time.time() + 85.0
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    
    # Build adjacency structure for fast spectral radius computation
    # The graph is a prism: two m-cycles + matching
    # We need to build the signed adjacency matrix and compute its spectral radius
    
    # Pre-compute edge indices for fast matrix building
    edge_idx = []
    for u, v in edges:
        edge_idx.append((u, v))
    
    def compute_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        S = np.zeros((n, n), dtype=np.float64)
        for i, (u, v) in enumerate(edge_idx):
            s = signs[i]
            S[u, v] = s
            S[v, u] = s
        # For symmetric matrix, spectral radius is max abs eigenvalue
        eigvals = np.linalg.eigvalsh(S)
        return np.max(np.abs(eigvals))
    
    # Check if a sign assignment gives a valid (connected) lift
    def is_valid_lift(signs):
        """Check if the 2-lift is connected."""
        # The lift has vertices (v, 0) and (v, 1) for v in 0..n-1
        # +1 on edge (u,v) connects (u,0)-(v,0) and (u,1)-(v,1)
        # -1 on edge (u,v) connects (u,0)-(v,1) and (u,1)-(v,0)
        
        # Build adjacency for the lift using BFS
        from collections import deque
        
        lift_n = 2 * n
        adj = [[] for _ in range(lift_n)]
        
        for i, (u, v) in enumerate(edge_idx):
            s = signs[i]
            if s == 1:
                # Equal sides
                adj[u * 2].append(v * 2)
                adj[v * 2].append(u * 2)
                adj[u * 2 + 1].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2 + 1)
            else:
                # Opposite sides
                adj[u * 2].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2)
                adj[u * 2 + 1].append(v * 2)
                adj[v * 2].append(u * 2 + 1)
        
        # BFS from (0, 0)
        visited = [False] * lift_n
        queue = deque([0])
        visited[0] = True
        count = 1
        
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    count += 1
                    queue.append(neighbor)
        
        return count == lift_n
    
    # Start with random search using the seed's approach
    rng = np.random.default_rng(3)
    
    best_signs = None
    best_radius = np.inf
    
    # First, try the seed's approach: sample budget random vectors
    for _ in range(budget):
        signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        
        if not is_valid_lift(signs):
            continue
        
        try:
            radius, details = check_lift(graph, signs)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
        except:
            pass
    
    if best_signs is None:
        # Fallback: all +1 should give a valid lift (two disconnected cycles)
        # Wait, all +1 gives two disconnected copies, so it's NOT connected.
        # We need at least one -1 to connect them.
        # Let's try a simple pattern: one -1 edge
        for i in range(num_edges):
            signs = np.ones(num_edges, dtype=np.int8)
            signs[i] = -1
            if is_valid_lift(signs):
                try:
                    radius, details = check_lift(graph, signs)
                    if radius < best_radius:
                        best_radius = radius
                        best_signs = signs.copy()
                except:
                    pass
    
    if best_signs is None:
        # Emergency fallback
        return np.ones(num_edges, dtype=np.int8)
    
    # Now do local search to improve
    current_signs = best_signs.copy()
    current_radius = best_radius
    
    # Local search: flip one edge at a time
    while time.time() < deadline:
        improved = False
        
        for i in range(num_edges):
            if time.time() >= deadline:
                break
            
            # Flip edge i
            test_signs = current_signs.copy()
            test_signs[i] = -test_signs[i]
            
            if not is_valid_lift(test_signs):
                continue
            
            try:
                radius, details = check_lift(graph, test_signs)
                if radius < current_radius - 1e-10:
                    current_signs = test_signs
                    current_radius = radius
                    best_signs = test_signs.copy()
                    best_radius = radius
                    improved = True
            except:
                pass
        
        if not improved:
            # Try 2-flip moves
            two_flip_improved = False
            for i in range(num_edges):
                if time.time() >= deadline:
                    break
                for j in range(i + 1, num_edges):
                    if time.time() >= deadline:
                        break
                    
                    test_signs = current_signs.copy()
                    test_signs[i] = -test_signs[i]
                    test_signs[j] = -test_signs[j]
                    
                    if not is_valid_lift(test_signs):
                        continue
                    
                    try:
                        radius, details = check_lift(graph, test_signs)
                        if radius < current_radius - 1e-10:
                            current_signs = test_signs
                            current_radius = radius
                            best_signs = test_signs.copy()
                            best_radius = radius
                            two_flip_improved = True
                            break
                    except:
                        pass
                if two_flip_improved:
                    break
            
            if not two_flip_improved:
                # Try random restarts
                for _ in range(100):
                    if time.time() >= deadline:
                        break
                    
                    signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
                    
                    if not is_valid_lift(signs):
                        continue
                    
                    try:
                        radius, details = check_lift(graph, signs)
                        if radius < current_radius - 1e-10:
                            current_signs = signs
                            current_radius = radius
                            best_signs = signs.copy()
                            best_radius = radius
                            break
                    except:
                        pass
                
                if time.time() >= deadline:
                    break
                # If no improvement after random restarts, stop
                if radius >= current_radius - 1e-10:
                    break
    
    return best_signs
# EVOLVE-BLOCK-END
