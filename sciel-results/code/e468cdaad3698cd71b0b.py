import time
import numpy as np
from collections import deque

def sign_edges(graph, budget):
    deadline = time.time() + 85.0
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Precompute edge indices for specific types
    # Top: (i, i+1) for i in 0..m-1 (mod m)
    # Bottom: (m+i, m+i+1) for i in 0..m-1 (mod m)
    # Match: (i, m+i) for i in 0..m-1
    
    edge_list = [(u, v) for u, v in edges]
    
    # Build adjacency list for connectivity check
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edge_list):
        adj[u].append((v, i))
        adj[v].append((u, i))
        
    # Identify cycle edge sets for faster "cycle flip" moves
    # We need to map edge indices to their positions
    # This depends on the specific edge ordering in the input, 
    # but usually it's consistent. 
    # To be safe, we will just use single and pair flips, 
    # and random kicks. Cycle flips require knowing which edges form cycles.
    
    def build_signed_matrix(signs):
        A = np.zeros((n, n))
        for i, (u, v) in enumerate(edge_list):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
        return A
    
    def spectral_radius(signs):
        A = build_signed_matrix(signs)
        evals = np.linalg.eigvalsh(A)
        return np.max(np.abs(evals))
    
    def is_connected_lift(signs):
        # Check if signing is a coboundary
        # t_v assignment: t_root = 1, t_v = t_u * s_e for edge (u,v)
        t = np.zeros(n, dtype=int)  # 0 = unassigned, 1 or -1
        t[0] = 1
        
        visited = [False] * n
        visited[0] = True
        queue = deque([0])
        while queue:
            u = queue.popleft()
            for v, ei in adj[u]:
                if not visited[v]:
                    t[v] = t[u] * signs[ei]
                    visited[v] = True
                    queue.append(v)
                else:
                    # Check consistency
                    if t[v] != t[u] * signs[ei]:
                        return True  # Not a coboundary -> connected lift
        return False  # Is a coboundary -> disconnected lift

    def random_valid_signing(rng):
        while True:
            signs = rng.choice([-1, 1], size=num_edges)
            if is_connected_lift(signs):
                return signs

    best_radius = float('inf')
    best_signs = None
    
    rng = np.random.default_rng(3)
    
    # Structured initial guesses
    candidates = []
    
    # 1. Random valid
    for _ in range(10):
        candidates.append(random_valid_signing(rng))
        
    # 2. Alternating patterns (if m allows)
    # Try to create frustration
    if m % 2 == 0:
        # Alternating on top
        signs = np.ones(num_edges)
        # We need to know which edges are which.
        # Since edge order is not guaranteed to be grouped, 
        # we will rely on the local search to find the structure.
        pass
        
    # Local Search Function
    def local_search(initial_signs, max_iterations=5000):
        signs = initial_signs.copy()
        current_radius = spectral_radius(signs)
        if not is_connected_lift(signs):
            return None, float('inf')
            
        no_improve_count = 0
        for _ in range(max_iterations):
            if time.time() > deadline - 2.0:
                break
                
            improved = False
            
            # 1. Single flip
            # Shuffle order to avoid bias
            edge_order = rng.permutation(num_edges)
            for i in edge_order:
                if time.time() > deadline - 1.0:
                    break
                old_sign = signs[i]
                signs[i] = -old_sign
                if is_connected_lift(signs):
                    new_radius = spectral_radius(signs)
                    if new_radius < current_radius:
                        current_radius = new_radius
                        improved = True
                        no_improve_count = 0
                    else:
                        signs[i] = old_sign
                else:
                    signs[i] = old_sign
            
            # 2. Pair flip (2-opt)
            if not improved:
                # Try a few random pairs
                for _ in range(5):
                    if time.time() > deadline - 1.0:
                        break
                    i, j = rng.choice(num_edges, 2, replace=False)
                    signs[i] *= -1
                    signs[j] *= -1
                    if is_connected_lift(signs):
                        new_radius = spectral_radius(signs)
                        if new_radius < current_radius:
                            current_radius = new_radius
                            improved = True
                            no_improve_count = 0
                        else:
                            signs[i] *= -1
                            signs[j] *= -1
                    else:
                        signs[i] *= -1
                        signs[j] *= -1
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > 50:
                    # Kick: flip 3 random edges
                    kick_edges = rng.choice(num_edges, 3, replace=False)
                    for k in kick_edges:
                        signs[k] *= -1
                    if is_connected_lift(signs):
                        current_radius = spectral_radius(signs)
                    else:
                        # If invalid, revert
                        for k in kick_edges:
                            signs[k] *= -1
                    no_improve_count = 0
                    
        return signs, current_radius

    # Run local search from multiple starts
    for restart in range(50):
        if time.time() > deadline - 5.0:
            break
            
        start_signs = random_valid_signing(rng)
        found_signs, radius = local_search(start_signs)
        
        if found_signs is not None and radius < best_radius:
            best_radius = radius
            best_signs = found_signs.copy()
            
        # Also try to improve the best found so far
        if best_signs is not None:
            improved_signs, improved_radius = local_search(best_signs, max_iterations=1000)
            if improved_signs is not None and improved_radius < best_radius:
                best_radius = improved_radius
                best_signs = improved_signs.copy()

    if best_signs is None:
        # Fallback: return a random valid one
        best_signs = random_valid_signing(rng)
        
    return best_signs
