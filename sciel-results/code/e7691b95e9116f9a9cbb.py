import numpy as np
import time
import random

def sign_edges(graph, budget):
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    
    # Determine m
    m = n // 2
    
    # Build adjacency list with edge indices
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))

    def is_balanced(signs):
        """
        Returns True if the signed graph is balanced (lift is disconnected).
        """
        colors = [0] * n # 0 unvisited, 1 or -1 for colors
        
        for start in range(n):
            if colors[start] != 0:
                continue
            colors[start] = 1
            queue = [start]
            while queue:
                u = queue.pop(0)
                for v, idx in adj[u]:
                    s = signs[idx]
                    # Constraint: colors[u] * colors[v] == s
                    # colors[v] == s * colors[u]
                    expected = s * colors[u]
                    if colors[v] == 0:
                        colors[v] = expected
                        queue.append(v)
                    else:
                        if colors[v] != expected:
                            return False # Not balanced (Good)
        return True # Balanced (Bad)

    def evaluate(signs):
        """
        Returns spectral radius if lift is connected, else inf.
        """
        if is_balanced(signs):
            return float('inf')
        
        # Build signed adjacency matrix
        A = np.zeros((n, n), dtype=float)
        for idx, (u, v) in enumerate(edges):
            s = float(signs[idx])
            A[u, v] = s
            A[v, u] = s
        
        # Compute spectral radius
        try:
            eigvals = np.linalg.eigvalsh(A)
            return max(abs(eigvals[0]), abs(eigvals[-1]))
        except:
            return float('inf')

    best_signs = None
    best_val = float('inf')
    
    # Heuristic Initialization: All -1
    init_signs = np.ones(num_edges, dtype=int) * -1
    
    # If init is balanced, flip a random edge to make it unbalanced
    if is_balanced(init_signs):
        idx_to_flip = random.randint(0, num_edges - 1)
        init_signs[idx_to_flip] *= -1
        # Check again, just in case (unlikely but possible if graph is small)
        if is_balanced(init_signs):
             # Flip another
             init_signs[random.randint(0, num_edges - 1)] *= -1

    s = init_signs.copy()
    v = evaluate(s)
    if v < best_val:
        best_val = v
        best_signs = s.copy()

    def local_search(current_signs, current_val):
        nonlocal best_signs, best_val
        
        improved = True
        while improved:
            improved = False
            order = list(range(num_edges))
            random.shuffle(order)
            
            for idx in order:
                if time.time() - start_time > time_limit:
                    return
                
                old_sign = current_signs[idx]
                current_signs[idx] = -old_sign
                
                new_val = evaluate(current_signs)
                
                if new_val < current_val:
                    current_val = new_val
                    improved = True
                    if current_val < best_val:
                        best_val = current_val
                        best_signs = current_signs.copy()
                else:
                    current_signs[idx] = old_sign
                    
            # If no improvement in a full pass, stop

    local_search(s, v)
    
    # Random restarts
    rng = np.random.default_rng(42)
    while time.time() - start_time < time_limit - 5:
        # Random signs
        s = rng.choice([-1, 1], size=num_edges)
        
        # Ensure connectivity
        if is_balanced(s):
            # Flip a random edge
            idx = rng.integers(0, num_edges)
            s[idx] *= -1
            if is_balanced(s):
                idx2 = rng.integers(0, num_edges)
                s[idx2] *= -1
                if is_balanced(s):
                    # Force
                    s[0] *= -1

        v = evaluate(s)
        if v < best_val:
            best_val = v
            best_signs = s.copy()
        
        local_search(s, v)

    if best_signs is None:
        return init_signs
        
    return best_signs
