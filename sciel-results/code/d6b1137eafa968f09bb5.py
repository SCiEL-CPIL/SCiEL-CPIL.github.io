"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    deadline = time.time() + 50.0
    
    # Classify edges: cycle1 (0..m-1), cycle2 (m..2m-1), matching
    # For a prism graph with 2m vertices:
    # Cycle 1: vertices 0..m-1
    # Cycle 2: vertices m..2m-1
    # Matching: (i, i+m) for i in 0..m-1
    
    # Build adjacency info for the base graph
    # We need to identify which edges belong to which cycle and matching
    
    edge_cycle1 = []  # indices of edges in cycle 1
    edge_cycle2 = []  # indices of edges in cycle 2
    edge_matching = []  # indices of matching edges
    
    for idx, (u, v) in enumerate(edges):
        u_min = min(u, v)
        u_max = max(u, v)
        if u_max < m:
            edge_cycle1.append(idx)
        elif u_min >= m:
            edge_cycle2.append(idx)
        else:
            edge_matching.append(idx)
    
    def build_signed_matrix(signs):
        """Build the signed adjacency matrix of the base graph."""
        A = np.zeros((n, n), dtype=float)
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
        return A
    
    def spectral_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        A = build_signed_matrix(signs)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_lift_connected(signs):
        """Check if the 2-lift is connected using BFS."""
        # Build the lift graph
        # Vertices: (v, 0) and (v, 1) for v in 0..n-1
        # Edge (u,v) with sign +1: (u,0)-(v,0), (u,1)-(v,1)
        # Edge (u,v) with sign -1: (u,0)-(v,1), (u,1)-(v,0)
        
        lift_adj = [[] for _ in range(2 * n)]
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            if s == 1:
                lift_adj[u * 2].append(v * 2)
                lift_adj[v * 2].append(u * 2)
                lift_adj[u * 2 + 1].append(v * 2 + 1)
                lift_adj[v * 2 + 1].append(u * 2 + 1)
            else:
                lift_adj[u * 2].append(v * 2 + 1)
                lift_adj[v * 2 + 1].append(u * 2)
                lift_adj[u * 2 + 1].append(v * 2)
                lift_adj[v * 2].append(u * 2 + 1)
        
        # BFS from (0, 0)
        visited = [False] * (2 * n)
        queue = deque([0])
        visited[0] = True
        count = 1
        while queue:
            node = queue.popleft()
            for neighbor in lift_adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    count += 1
                    queue.append(neighbor)
        return count == 2 * n
    
    def evaluate(signs):
        """Evaluate a sign vector: return spectral radius if valid, else None."""
        if not is_lift_connected(signs):
            return None
        return spectral_radius(signs)
    
    # Local search: flip one edge at a time
    def local_search(signs, max_iterations=1000):
        current_signs = signs.copy()
        current_val = evaluate(current_signs)
        if current_val is None:
            return current_signs, None
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_flip_idx = -1
            best_val = current_val
            
            for idx in range(num_edges):
                trial_signs = current_signs.copy()
                trial_signs[idx] = -trial_signs[idx]
                val = evaluate(trial_signs)
                if val is not None and val < best_val:
                    best_val = val
                    best_flip_idx = idx
            
            if best_flip_idx >= 0:
                current_signs[best_flip_idx] = -current_signs[best_flip_idx]
                current_val = best_val
                improved = True
        
        return current_signs, current_val
    
    # Initialize with random signs
    rng = np.random.default_rng(3)
    best_signs = None
    best_val = float('inf')
    
    # Try multiple random starts
    num_starts = 20
    for start in range(num_starts):
        if time.time() >= deadline:
            break
        
        signs = rng.choice([-1, 1], size=num_edges)
        
        # Check if valid
        val = evaluate(signs)
        if val is not None:
            signs, val = local_search(signs)
            if val is not None and val < best_val:
                best_val = val
                best_signs = signs.copy()
    
    # If no valid solution found, try to fix connectivity
    if best_signs is None:
        # Try all +1 first (should be connected for prism)
        signs = np.ones(num_edges, dtype=int)
        val = evaluate(signs)
        if val is not None:
            signs, val = local_search(signs)
            if val is not None:
                best_signs = signs
                best_val = val
        else:
            # Fallback: return all +1
            best_signs = np.ones(num_edges, dtype=int)
    
    return best_signs.astype(int)
# EVOLVE-BLOCK-END
