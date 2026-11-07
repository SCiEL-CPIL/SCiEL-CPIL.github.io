import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    DEADLINE = 85.0
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    def build_signed_adj(signs):
        A = np.zeros((n, n))
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
        return A
    
    def spectral_radius(A):
        eigvals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(eigvals)))
    
    def is_connected_lift(signs):
        adj = [[] for _ in range(2 * n)]
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            if s == 1:
                adj[2*u].append(2*v)
                adj[2*v].append(2*u)
                adj[2*u+1].append(2*v+1)
                adj[2*v+1].append(2*u+1)
            else:
                adj[2*u].append(2*v+1)
                adj[2*v+1].append(2*u)
                adj[2*u+1].append(2*v)
                adj[2*v].append(2*u+1)
        
        visited = [False] * (2 * n)
        stack = [0]
        visited[0] = True
        count = 1
        while stack:
            node = stack.pop()
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    count += 1
                    stack.append(neighbor)
        return count == 2 * n
    
    def evaluate(signs):
        if not is_connected_lift(signs):
            return False, float('inf')
        A = build_signed_adj(signs)
        r = spectral_radius(A)
        return True, r
    
    best_signs = None
    best_radius = float('inf')
    
    def update_best(signs, r):
        nonlocal best_signs, best_radius
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()
    
    def local_search(signs, max_iters=3000):
        nonlocal best_signs, best_radius
        
        current = signs.copy()
        valid, current_r = evaluate(current)
        if not valid:
            return current, current_r
        
        update_best(current, current_r)
        
        for it in range(max_iters):
            if time.time() > DEADLINE:
                break
            
            improved = False
            for idx in range(num_edges):
                if time.time() > DEADLINE:
                    break
                
                old = current[idx]
                current[idx] = -old
                valid, r = evaluate(current)
                
                if valid and r < current_r - 1e-12:
                    current_r = r
                    improved = True
                    update_best(current, current_r)
                else:
                    current[idx] = old
            
            if not improved:
                break
        
        return current, current_r
    
    rng = np.random.default_rng(3)
    
    # Phase 1: Random sampling
    for i in range(1000):
        if time.time() > DEADLINE:
            break
        signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        valid, r = evaluate(signs)
        if valid:
            update_best(signs, r)
    
    # Phase 2: Local search from best
    if best_signs is not None:
        local_search(best_signs)
    
    # Phase 3: More restarts with local search
    for i in range(50):
        if time.time() > DEADLINE:
            break
        signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        local_search(signs, max_iters=2000)
    
    # Phase 4: Final polish
    if best_signs is not None:
        local_search(best_signs, max_iters=5000)
    
    if best_signs is None:
        # Construct a valid signing
        best_signs = np.ones(num_edges, dtype=np.int8)
        for idx, (u, v) in enumerate(edges):
            if abs(u - v) == m:
                best_signs[idx] = -1
    
    return best_signs.astype(np.int8)
