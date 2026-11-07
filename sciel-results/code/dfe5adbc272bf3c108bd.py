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
    
    # Identify edge types
    bottom_edges = []
    top_edges = []
    match_edges = []
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            bottom_edges.append(i)
        elif u >= m and v >= m:
            top_edges.append(i)
        else:
            match_edges.append(i)
    
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
    
    # Pre-compute base unsigned adjacency matrix structure?
    # We need to build the signed matrix for each candidate.
    
    # Fast validity check: Is the sign pattern a coboundary?
    # If yes, lift is disconnected.
    # If no, lift is connected (for connected base).
    
    def is_coboundary(signs):
        # Check if there exists phi such that signs[i] = phi[u] * phi[v]
        # Fix phi[0] = 1
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

    def compute_radius(signs):
        A = np.zeros((n, n), dtype=np.float64)
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
        try:
            eigvals = np.linalg.eigvalsh(A)
            return np.max(np.abs(eigvals))
        except:
            return np.inf

    best_signs = None
    best_radius = np.inf
    
    # Enumerate all 2^(num_non_tree) sign assignments for non-tree edges
    # Tree edges are fixed to +1
    
    if num_non_tree <= 20:
        for combo in product([-1, 1], repeat=num_non_tree):
            if time.time() > deadline - 2.0:
                break
            
            signs = np.ones(E, dtype=np.int8)
            for j, ei in enumerate(non_tree_edges):
                signs[ei] = combo[j]
            
            # Check validity (connectivity)
            if is_coboundary(signs):
                continue
                
            radius = compute_radius(signs)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
    else:
        # Fallback: random search
        rng = np.random.default_rng(42)
        for _ in range(10000):
            if time.time() > deadline - 2.0:
                break
            
            signs = rng.choice([-1, 1], size=E).astype(np.int8)
            if is_coboundary(signs):
                continue
            radius = compute_radius(signs)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
    
    # Fallback if no valid lift found
    if best_signs is None:
        # Try all +1 (which is a coboundary, so disconnected, so invalid)
        # We need a valid one.
        # Try a pattern that is likely not a coboundary.
        # Flip one non-tree edge to -1.
        if non_tree_edges:
            signs = np.ones(E, dtype=np.int8)
            signs[non_tree_edges[0]] = -1
            if not is_coboundary(signs):
                best_signs = signs
            else:
                # Try random
                rng = np.random.default_rng(123)
                for _ in range(100):
                    signs = rng.choice([-1, 1], size=E).astype(np.int8)
                    if not is_coboundary(signs):
                        best_signs = signs
                        break
        if best_signs is None:
            # Last resort, all +1 (even if invalid, better than crash? No, must return valid)
            # But if all are invalid, we are stuck.
            # For prism graphs, there are always valid lifts.
            best_signs = np.ones(E, dtype=np.int8)
    
    return best_signs
