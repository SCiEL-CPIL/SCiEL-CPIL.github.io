import time
import numpy as np
from itertools import product
from collections import deque

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin for 60s limit
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    E = len(edges)
    
    # Precompute edge endpoints as arrays
    edge_u = np.array([e[0] for e in edges], dtype=np.int32)
    edge_v = np.array([e[1] for e in edges], dtype=np.int32)
    
    # Build adjacency structure for base graph
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    # BFS to find spanning tree
    visited = [False] * n
    tree_edges = set()
    queue = deque([0])
    visited[0] = True
    
    while queue:
        u = queue.popleft()
        for v, ei in adj[u]:
            if not visited[v]:
                visited[v] = True
                tree_edges.add(ei)
                queue.append(v)
    
    # Non-tree edges
    non_tree_edges = [i for i in range(E) if i not in tree_edges]
    num_non_tree = len(non_tree_edges)
    
    # Precompute base matrix with tree edges set to +1, non-tree set to 0
    base_A = np.zeros((n, n), dtype=np.float64)
    for i in range(E):
        if i in tree_edges:
            u, v = edge_u[i], edge_v[i]
            base_A[u, v] = 1.0
            base_A[v, u] = 1.0
    
    # Precompute: which positions in the edge array are non-tree
    non_tree_indices = np.array(non_tree_edges, dtype=np.int32)
    non_tree_u = edge_u[non_tree_indices]
    non_tree_v = edge_v[non_tree_indices]
    
    def is_coboundary(signs):
        phi = [0] * n
        phi[0] = 1
        q = deque([0])
        while q:
            u = q.popleft()
            for v, ei in adj[u]:
                expected = phi[u] * signs[ei]
                if phi[v] == 0:
                    phi[v] = expected
                    q.append(v)
                else:
                    if phi[v] != expected:
                        return False
        return True

    best_signs = None
    best_radius = np.inf
    
    if num_non_tree <= 20:
        for combo in product([-1, 1], repeat=num_non_tree):
            if time.time() > deadline - 2.0:
                break
            
            signs = np.ones(E, dtype=np.int8)
            signs[non_tree_indices] = np.array(combo, dtype=np.int8)
            
            if is_coboundary(signs):
                continue
                
            # Build signed matrix: base_A + non-tree contributions
            A = base_A.copy()
            A[non_tree_u, non_tree_v] = combo
            A[non_tree_v, non_tree_u] = combo
            
            eigvals = np.linalg.eigvalsh(A)
            radius = np.max(np.abs(eigvals))
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
    else:
        rng = np.random.default_rng(42)
        for _ in range(10000):
            if time.time() > deadline - 2.0:
                break
            
            signs = rng.choice([-1, 1], size=E).astype(np.int8)
            if is_coboundary(signs):
                continue
            
            A = base_A.copy()
            A[non_tree_u, non_tree_v] = signs[non_tree_indices]
            A[non_tree_v, non_tree_u] = signs[non_tree_indices]
            
            eigvals = np.linalg.eigvalsh(A)
            radius = np.max(np.abs(eigvals))
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
    
    if best_signs is None:
        if non_tree_edges:
            signs = np.ones(E, dtype=np.int8)
            signs[non_tree_edges[0]] = -1
            if not is_coboundary(signs):
                best_signs = signs
            else:
                rng = np.random.default_rng(123)
                for _ in range(100):
                    signs = rng.choice([-1, 1], size=E).astype(np.int8)
                    if not is_coboundary(signs):
                        best_signs = signs
                        break
        if best_signs is None:
            best_signs = np.ones(E, dtype=np.int8)
    
    return best_signs
