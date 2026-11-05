import random
import time
import numpy as np

def creates_c6(u, v, adj):
    """
    Check if adding edge (u, v) creates a C6.
    This is true if there is a simple path of length 5 between u and v in the current graph.
    Path: u - a - b - c - d - v
    """
    # Optimization: if u and v are already connected, this function shouldn't be called for that edge
    # But we call it for non-edges.
    
    # We need to find a, b, c, d such that:
    # a in N(u)
    # b in N(a) \ {u}
    # c in N(b) \ {u, a}
    # d in N(c) \ {u, a, b}
    # (d, v) in E and d != v
    
    neighbors_u = adj[u]
    neighbors_v = adj[v]
    
    # If u and v have no common neighbors, can they be part of a C6?
    # Path u-a-b-c-d-v.
    # a is neighbor of u. d is neighbor of v.
    # If N(u) and N(v) are disjoint, we still need the middle part b-c.
    
    for a in neighbors_u:
        if a == v:
            continue
        # b is neighbor of a, not u
        for b in adj[a]:
            if b == u or b == a:
                continue
            # c is neighbor of b, not u, a
            for c in adj[b]:
                if c == u or c == a or c == b:
                    continue
                # d is neighbor of c, not u, a, b
                # And d must be neighbor of v
                # So we just check neighbors of c
                for d in adj[c]:
                    if d == u or d == a or d == b or d == c:
                        continue
                    # Check if d is connected to v
                    if d in neighbors_v:
                        return True
    return False

def construct():
    n = 40
    best_edges = []
    best_m = 0
    start_time = time.time()
    timeout = 115  # Leave 5s for safety
    
    # Pre-generate all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    for trial in range(50):
        if time.time() - start_time > timeout:
            break
        
        # 1. Spectral Heuristic Initialization
        # Start with a random graph, roughly regular degree 5
        A = np.zeros((n, n))
        for u in range(n):
            candidates = [v for v in range(n) if v != u]
            random.shuffle(candidates)
            for v in candidates[:5]:
                if A[u, v] == 0:
                    A[u, v] = 1
                    A[v, u] = 1
        
        # 2. Projected Gradient Ascent on continuous relaxation
        # Objective: Maximize sum(A) - gamma * lambda_max(A)
        gamma = 0.1
        eta = 0.1
        
        for step in range(20):
            eigvals, eigvecs = np.linalg.eig(A)
            idx = np.argmax(eigvals)
            x = eigvecs[:, idx]
            x = x / np.linalg.norm(x)
            
            grad = np.ones((n, n)) - gamma * np.outer(x, x)
            A = A + eta * grad
            A = np.clip(A, 0, 1)
            A = (A + A.T) / 2
            np.fill_diagonal(A, 0)
        
        # 3. Round to binary
        A_bin = (A > 0.5).astype(int)
        
        # Convert to adj sets
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if A_bin[u, v] == 1:
                    adj[u].add(v)
                    adj[v].add(u)
        
        # 4. Repair Phase: Remove edges that form C6s
        edges_to_check = [(u, v) for u in range(n) for v in adj[u] if u < v]
        random.shuffle(edges_to_check)
        
        for u, v in edges_to_check:
            if u in adj[v]:
                adj[u].remove(v)
                adj[v].remove(u)
                if creates_c6(u, v, adj):
                    pass # Keep removed
                else:
                    adj[u].add(v)
                    adj[v].add(u)
        
        # 5. Greedy Addition of remaining edges
        candidates = [e for e in all_edges if e[0] not in adj[e[1]]]
        random.shuffle(candidates)
        
        for u, v in candidates:
            if creates_c6(u, v, adj):
                continue
            adj[u].add(v)
            adj[v].add(u)
            
        # Count edges
        m = sum(len(s) for s in adj) // 2
        if m > best_m:
            best_m = m
            best_edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        best_edges.append([u, v])
                        
    return best_edges
