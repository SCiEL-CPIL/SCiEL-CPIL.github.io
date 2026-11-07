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
    
    # Precompute edge endpoints as arrays for speed
    eu = np.array([u for u, v in edges], dtype=np.int32)
    ev = np.array([v for u, v in edges], dtype=np.int32)
    
    def build_signed_matrix(signs):
        A = np.zeros((n, n), dtype=np.float64)
        np.add.at(A, (eu, ev), signs.astype(np.float64))
        np.add.at(A, (ev, eu), signs.astype(np.float64))
        return A
    
    def spectral_radius(A):
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_connected_lift(signs):
        # Fast connectivity check using BFS on the 2-lift
        visited = np.zeros(2 * n, dtype=bool)
        queue = [0]  # (vertex 0, layer 0)
        visited[0] = True
        
        # Precompute for each edge: sign as 0 (+1) or 1 (-1)
        twist = (signs == -1).astype(np.int8)
        
        # For each vertex, we need to know its 3 neighbors and the corresponding edge signs
        vert_edges = [[] for _ in range(n)]
        for i in range(num_edges):
            vert_edges[eu[i]].append(i)
            vert_edges[ev[i]].append(i)
        
        while queue:
            node = queue.pop(0)
            v = node // 2
            layer = node % 2
            for ei in vert_edges[v]:
                u = eu[ei] if ev[ei] == v else ev[ei]
                t = twist[ei]
                nl = layer ^ t
                nn = u * 2 + nl
                if not visited[nn]:
                    visited[nn] = True
                    queue.append(nn)
        
        return all(visited)
    
    def is_valid(signs):
        return is_connected_lift(signs)
    
    # Continuous phase-based optimization
    # For each edge, represent sign as cos(theta) where theta is 0 or pi
    # Minimize spectral radius of matrix with entries cos(theta_i)
    # Use gradient descent on the phase variables
    
    def spectral_radius_continuous(thetas):
        """Compute spectral radius for continuous phases (entries are cos(theta_i))"""
        A = np.zeros((n, n), dtype=np.float64)
        c = np.cos(thetas)
        np.add.at(A, (eu, ev), c)
        np.add.at(A, (ev, eu), c)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def gradient_spectral_radius(thetas):
        """Compute gradient of spectral radius w.r.t. thetas using finite differences"""
        eps = 1e-6
        grad = np.zeros_like(thetas)
        base_r = spectral_radius_continuous(thetas)
        for i in range(num_edges):
            theta_plus = thetas.copy()
            theta_plus[i] += eps
            r_plus = spectral_radius_continuous(theta_plus)
            grad[i] = (r_plus - base_r) / eps
        return grad
    
    def sign_from_phases(thetas):
        """Convert continuous phases to discrete signs"""
        # Round to nearest 0 or pi
        # cos(theta) > 0 means sign +1, cos(theta) < 0 means sign -1
        signs = np.where(np.cos(thetas) >= 0, 1, -1).astype(np.int8)
        return signs
    
    rng = np.random.default_rng(42)
    
    best_signs = None
    best_radius = float('inf')
    
    # Run continuous optimization from multiple random starts
    num_restarts = 40
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random initial phases in [0, 2*pi)
        thetas = rng.uniform(0, 2 * np.pi, size=num_edges)
        
        # Gradient descent
        lr = 0.1
        for step in range(200):
            if time.time() > deadline:
                break
            
            r = spectral_radius_continuous(thetas)
            grad = gradient_spectral_radius(thetas)
            
            # Update
            thetas = thetas - lr * grad
            # Keep in [0, 2*pi)
            thetas = thetas % (2 * np.pi)
        
        # Convert to discrete signs
        signs = sign_from_phases(thetas)
        
        # Check validity
        if is_valid(signs):
            A = build_signed_matrix(signs)
            r = spectral_radius(A)
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
    
    # Also try local search from the best continuous solution
    if best_signs is not None:
        signs = best_signs.copy()
        A = build_signed_matrix(signs)
        cur_radius = spectral_radius(A)
        
        # Local search with single and double flips
        def flip_and_eval(signs, A, idx):
            u, v = eu[idx], ev[idx]
            old_s = float(signs[idx])
            signs[idx] = -signs[idx]
            new_s = float(signs[idx])
            
            A[u, v] = new_s
            A[v, u] = new_s
            
            if not is_valid(signs):
                signs[idx] = -signs[idx]
                A[u, v] = old_s
                A[v, u] = old_s
                return None
            
            r = spectral_radius(A)
            return r
        
        improved = True
        stall_count = 0
        while improved and time.time() < deadline:
            improved = False
            order = rng.permutation(num_edges)
            
            for idx in order:
                if time.time() > deadline:
                    break
                new_r = flip_and_eval(signs, A, idx)
                if new_r is not None and new_r < cur_radius:
                    cur_radius = new_r
                    improved = True
                    if cur_radius < best_radius:
                        best_radius = cur_radius
                        best_signs = signs.copy()
            
            if not improved:
                stall_count += 1
            else:
                stall_count = 0
    
    # Fallback: random search
    if best_signs is None:
        for _ in range(1000):
            if time.time() > deadline:
                break
            signs = rng.choice([-1, 1], size=num_edges)
            if is_valid(signs):
                A = build_signed_matrix(signs)
                r = spectral_radius(A)
                if r < best_radius:
                    best_radius = r
                    best_signs = signs.copy()
    
    if best_signs is None:
        signs = np.ones(num_edges, dtype=int)
        if not is_valid(signs):
            signs = -np.ones(num_edges, dtype=int)
            if not is_valid(signs):
                for _ in range(2000):
                    signs = rng.choice([-1, 1], size=num_edges)
                    if is_valid(signs):
                        break
        best_signs = signs
    
    return best_signs
# EVOLVE-BLOCK-END
