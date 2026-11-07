import time
import numpy as np

def sign_edges(graph, budget):
    deadline = time.time() + 85.0
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Precompute edge list
    edge_list = [(u, v) for u, v in edges]
    
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
        # BFS
        from collections import deque
        adj = [[] for _ in range(n)]
        for i, (u, v) in enumerate(edge_list):
            adj[u].append((v, i))
            adj[v].append((u, i))
        
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
    
    # Multiple restarts
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        signs = random_valid_signing(rng)
        current_radius = spectral_radius(signs)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = signs.copy()
        
        # Local search: try flipping each edge
        improved = True
        while improved:
            if time.time() > deadline - 3.0:
                break
            improved = False
            for i in range(num_edges):
                if time.time() > deadline - 2.0:
                    break
                old_sign = signs[i]
                signs[i] = -old_sign
                if is_connected_lift(signs):
                    new_radius = spectral_radius(signs)
                    if new_radius < current_radius:
                        current_radius = new_radius
                        improved = True
                        if current_radius < best_radius:
                            best_radius = current_radius
                            best_signs = signs.copy()
                    else:
                        signs[i] = old_sign
                else:
                    signs[i] = old_sign
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = signs.copy()
    
    return best_signs
