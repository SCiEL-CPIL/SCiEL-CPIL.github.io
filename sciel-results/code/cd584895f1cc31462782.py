"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def sign_edges(graph, budget):
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Build signed adjacency matrix structure
    # For each edge (u,v), we'll assign a sign
    # Signed adjacency: A[u,v] = sign, A[v,u] = sign (symmetric)
    
    # Precompute edge indices for quick lookup
    edge_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i
    
    def build_signed_matrix(signs):
        A = np.zeros((n, n))
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
        return A
    
    def spectral_radius(signs):
        A = build_signed_matrix(signs)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_connected_lift(signs):
        # Check if the 2-lift is connected
        # Build the lift graph with 2n vertices
        # Vertex (v, layer) where layer in {0, 1}
        # Edge from (u, l) to (v, l') where l' = l if sign=+1 else 1-l
        lift_adj = [[] for _ in range(2 * n)]
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            for l in [0, 1]:
                l2 = l if s == 1 else 1 - l
                lift_adj[u * 2 + l].append(v * 2 + l2)
                lift_adj[v * 2 + l2].append(u * 2 + l)
        
        # BFS from (0, 0)
        visited = [False] * (2 * n)
        queue = [0]  # vertex 0, layer 0
        visited[0] = True
        while queue:
            node = queue.pop(0)
            for nb in lift_adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        return all(visited)
    
    def is_valid(signs):
        if not is_connected_lift(signs):
            return False
        # Check 3-regular: each vertex in base graph has degree 3
        # In prism graph, each vertex has degree 3 (2 cycle + 1 matching)
        # The lift is always 3-regular if base is 3-regular
        return True
    
    # Local search
    rng = np.random.default_rng(42)
    
    best_signs = None
    best_radius = float('inf')
    
    # Try multiple random restarts
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random initial signs
        signs = rng.choice([-1, 1], size=num_edges)
        
        # Ensure valid start - if not, try to fix
        if not is_valid(signs):
            # Try flipping edges until valid
            valid = False
            for _ in range(100):
                if time.time() > deadline:
                    break
                flip_idx = rng.integers(num_edges)
                signs[flip_idx] *= -1
                if is_valid(signs):
                    valid = True
                    break
            if not valid:
                continue
        
        cur_radius = spectral_radius(signs)
        if cur_radius < best_radius:
            best_radius = cur_radius
            best_signs = signs.copy()
        
        # Local search: try flipping single edges
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            # Shuffle edge order for diversity
            order = rng.permutation(num_edges)
            for idx in order:
                if time.time() > deadline:
                    break
                old_sign = signs[idx]
                signs[idx] = -old_sign
                if is_valid(signs):
                    new_radius = spectral_radius(signs)
                    if new_radius < cur_radius:
                        cur_radius = new_radius
                        improved = True
                        if cur_radius < best_radius:
                            best_radius = cur_radius
                            best_signs = signs.copy()
                    else:
                        signs[idx] = old_sign
                else:
                    signs[idx] = old_sign
    
    # If no valid solution found, return all +1 (may not be connected but valid array)
    if best_signs is None:
        # Try to find any valid configuration
        signs = np.ones(num_edges, dtype=int)
        if not is_valid(signs):
            # Try all -1
            signs = -np.ones(num_edges, dtype=int)
            if not is_valid(signs):
                # Random search for valid
                for _ in range(1000):
                    signs = rng.choice([-1, 1], size=num_edges)
                    if is_valid(signs):
                        break
        best_signs = signs
    
    return best_signs
# EVOLVE-BLOCK-END
