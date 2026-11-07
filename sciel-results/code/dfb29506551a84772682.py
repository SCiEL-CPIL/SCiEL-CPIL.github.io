# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import deque

def _compute_radius_and_validity(graph, signs):
    """
    Compute spectral radius of signed adjacency matrix and check lift validity.
    Returns (radius, is_valid).
    """
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    
    # Build signed adjacency matrix B (n x n)
    B = np.zeros((n, n))
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        B[u, v] = s
        B[v, u] = s
    
    # Compute eigenvalues
    try:
        eigvals = np.linalg.eigvalsh(B)
        radius = np.max(np.abs(eigvals))
    except Exception:
        return None, False
    
    # Check connectivity of the lift
    # Lift vertices: (v, side) for v in 0..n-1, side in 0,1
    # Total 2n vertices.
    # Edge (u, v) with sign s:
    #   if s > 0: (u,0)-(v,0) and (u,1)-(v,1)
    #   if s < 0: (u,0)-(v,1) and (u,1)-(v,0)
    # Note: For connectivity check, we only care about the sign (topology), not magnitude.
    # But we must ensure the discrete signs are -1 or +1.
    # The input 'signs' here is assumed to be discrete for this function in the search loop,
    # but in gradient descent, we might pass continuous values. 
    # However, connectivity is a topological property determined by the sign of the continuous value.
    # So we use np.sign(signs).
    
    discrete_signs = np.sign(signs)
    # If any sign is 0, it's invalid for the lift definition (must be -1 or 1).
    # But in relaxation, we snap at the end. For intermediate steps in GD, we just need topology.
    # If a sign is 0, treat it as +1 for connectivity check? No, 0 is not a valid sign.
    # We will only call this on snapped signs or ensure signs are not 0.
    
    lift_n = 2 * n
    adj = [[] for _ in range(lift_n)]
    for i, (u, v) in enumerate(edges):
        s = discrete_signs[i]
        if s == 0:
            # Fallback for zero: treat as +1 to avoid crash, but this state is invalid
            s = 1 
            
        if s > 0:
            # (u,0)-(v,0) and (u,1)-(v,1)
            adj[u].append(v)          
            adj[v].append(u)
            adj[u + n].append(v + n)   
            adj[v + n].append(u + n)
        else:
            # (u,0)-(v,1) and (u,1)-(v,0)
            adj[u].append(v + n)       
            adj[v + n].append(u)
            adj[u + n].append(v)       
            adj[v].append(u + n)
    
    # BFS from vertex 0
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
    
    is_connected = (count == lift_n)
    
    return radius, is_connected

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Precompute edge structure for fast matrix updates
    # We will maintain a continuous vector x of length num_edges, initialized to +/- 1.
    # We want to minimize the spectral radius of B(x).
    
    # Helper to build B from continuous x
    def build_B(x):
        B = np.zeros((n, n))
        for i, (u, v) in enumerate(edges):
            B[u, v] = x[i]
            B[v, u] = x[i]
        return B

    # Gradient estimation via power method
    # The spectral radius rho(B) is the max abs eigenvalue.
    # For symmetric B, rho(B) = max(|lambda_max|, |lambda_min|).
    # Gradient of rho wrt B_ij is related to the eigenvector.
    # Let v be the eigenvector corresponding to the eigenvalue with largest magnitude.
    # If lambda_max is the one with largest magnitude, grad rho = v v^T (scaled).
    # Actually, d rho / d B = v v^T where v is the normalized eigenvector of the dominant eigenvalue.
    # We need to be careful with the sign if the dominant eigenvalue is negative.
    # Let's just compute the gradient numerically or analytically.
    
    # Analytical gradient:
    # Let lambda* be the eigenvalue with max |lambda|.
    # Let v* be the corresponding normalized eigenvector.
    # The gradient of |lambda*| with respect to B is v* v*^T if lambda* > 0, and -v* v*^T if lambda* < 0?
    # Actually, f(B) = max_i |lambda_i(B)|.
    # If lambda* is unique and dominant, grad f(B) = sign(lambda*) * v* v*^T.
    
    def compute_gradient_and_radius(x):
        B = build_B(x)
        eigvals, eigvecs = np.linalg.eigh(B)
        # Find index of max abs eigenvalue
        abs_eigvals = np.abs(eigvals)
        idx = np.argmax(abs_eigvals)
        radius = abs_eigvals[idx]
        v = eigvecs[:, idx]
        
        # Gradient of radius wrt B
        # If eigvals[idx] > 0, grad = v v^T
        # If eigvals[idx] < 0, grad = -v v^T (because we are minimizing |lambda| = -lambda)
        # Wait, if lambda is negative, |lambda| = -lambda. d(-lambda)/dB = - d(lambda)/dB = - v v^T.
        # So grad_B = sign(eigvals[idx]) * np.outer(v, v)
        
        grad_B = np.sign(eigvals[idx]) * np.outer(v, v)
        
        # Now map grad_B to grad_x
        # B[u, v] = x[i] for edge i.
        # d(radius)/d(x[i]) = d(radius)/d(B[u,v]) * d(B[u,v])/d(x[i]) + d(radius)/d(B[v,u]) * d(B[v,u])/d(x[i])
        # Since B is symmetric, B[u,v] = B[v,u] = x[i].
        # So grad_x[i] = 2 * grad_B[u, v]
        
        grad_x = np.zeros(num_edges)
        for i, (u, v) in enumerate(edges):
            grad_x[i] = 2.0 * grad_B[u, v]
            
        return radius, grad_x

    # Multi-start gradient descent
    best_radius = float('inf')
    best_signs = None
    
    rng = np.random.default_rng(3)
    
    # Number of restarts
    num_restarts = 100
    steps_per_restart = 200
    lr = 0.1
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize x randomly to +/- 1
        x = np.ones(num_edges)
        # Randomly flip some
        flip_mask = rng.random(num_edges) < 0.5
        x[flip_mask] = -1.0
        
        # Ensure validity of the initial discrete sign?
        # Not strictly necessary for the relaxation, but we want to end up at a valid discrete sign.
        # The connectivity constraint is discrete. We will check it at the end of each restart.
        
        current_radius = float('inf')
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            radius, grad = compute_gradient_and_radius(x)
            
            # Update x
            x -= lr * grad
            
            # Project back to [-1, 1]
            x = np.clip(x, -1.0, 1.0)
            
            # Optional: Periodically snap to discrete signs to check validity and potentially restart from there
            if step % 50 == 49:
                discrete_x = np.sign(x)
                # Handle zeros
                discrete_x[discrete_x == 0] = 1.0
                rad, valid = _compute_radius_and_validity(graph, discrete_x)
                if valid and rad < best_radius:
                    best_radius = rad
                    best_signs = discrete_x.copy()
        
        # Final snap for this restart
        discrete_x = np.sign(x)
        discrete_x[discrete_x == 0] = 1.0
        rad, valid = _compute_radius_and_validity(graph, discrete_x)
        if valid and rad < best_radius:
            best_radius = rad
            best_signs = discrete_x.copy()
            
    # If no valid solution found via GD, fall back to random search
    if best_signs is None:
        # Simple random search
        for _ in range(1000):
            if time.time() > deadline:
                break
            signs = np.ones(num_edges)
            flip_mask = rng.random(num_edges) < 0.5
            signs[flip_mask] = -1.0
            rad, valid = _compute_radius_and_validity(graph, signs)
            if valid and rad < best_radius:
                best_radius = rad
                best_signs = signs.copy()
                
    if best_signs is None:
        # Fallback: All +1 except one -1 to ensure connectivity
        signs = np.ones(num_edges)
        signs[0] = -1
        return signs
        
    return best_signs
# EVOLVE-BLOCK-END
