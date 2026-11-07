"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def sign_edges(graph, budget):
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # Identify edge types by position in the ordered edge list
    # Top cycle: edges 0..m-1 (connecting i to i+1 mod m for i in 0..m-1)
    # Bottom cycle: edges m..2m-1 (connecting m+i to m+i+1 mod m)
    # Matching: edges 2m..3m-1 (connecting i to m+i)
    top_edges = list(range(m))
    bottom_edges = list(range(m, 2*m))
    match_edges = list(range(2*m, 3*m))
    
    # Build edge index map
    edge_index = {}
    for i, (u, v) in enumerate(edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    # Spanning tree for gauge fixing:
    # Use matching edges + first m-1 top cycle edges
    # Tree edges: all m matching + top_edges[0..m-2]
    # Chord edges: top_edges[m-1] + all m bottom edges
    tree_edge_indices = match_edges + top_edges[:m-1]
    chord_edge_indices = [top_edges[m-1]] + bottom_edges
    num_chords = len(chord_edge_indices)
    
    def build_signed_adj(signs):
        """Build the signed adjacency matrix."""
        A = np.zeros((n, n), dtype=float)
        for i, (u, v) in enumerate(edges):
            s = float(signs[i])
            A[u, v] = s
            A[v, u] = s
        return A
    
    def compute_radius(A):
        """Compute spectral radius from eigenvalues."""
        try:
            eigvals = np.linalg.eigvalsh(A)
            return np.max(np.abs(eigvals))
        except Exception:
            return float('inf')
    
    def check_connectivity(signs):
        """Check if the 2-lift is connected using BFS."""
        lift_n = 2 * n
        adj = [[] for _ in range(lift_n)]
        
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            if s == 1:
                adj[u*2].append(v*2)
                adj[v*2].append(u*2)
                adj[u*2+1].append(v*2+1)
                adj[v*2+1].append(u*2+1)
            else:
                adj[u*2].append(v*2+1)
                adj[v*2+1].append(u*2)
                adj[u*2+1].append(v*2)
                adj[v*2].append(u*2+1)
        
        visited = [False] * lift_n
        visited[0] = True
        queue = deque([0])
        count = 1
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    count += 1
                    queue.append(nb)
        
        return count == lift_n
    
    def evaluate(signs):
        """Evaluate a sign assignment: return spectral radius if valid, else inf."""
        if not check_connectivity(signs):
            return float('inf')
        A = build_signed_adj(signs)
        return compute_radius(A)
    
    def gauge_fix(signs):
        """Apply gauge transformation to make all tree edges +1."""
        signs = np.array(signs, dtype=int, copy=True)
        vertex_sign = np.ones(n, dtype=int)
        
        # Propagate through matching edges: (i, i+m)
        for i in range(m):
            vertex_sign[i+m] = vertex_sign[i] * signs[match_edges[i]]
        
        # Propagate through top cycle edges: (i, i+1) for i=0..m-2
        for i in range(m-1):
            vertex_sign[i+1] = vertex_sign[i] * signs[top_edges[i]]
        
        # Apply gauge transformation
        for i, (u, v) in enumerate(edges):
            signs[i] = vertex_sign[u] * vertex_sign[v] * signs[i]
        
        return signs
    
    best_signs = None
    best_radius = float('inf')
    
    def try_signs(signs):
        nonlocal best_signs, best_radius
        r = evaluate(signs)
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()
        return r
    
    # Phase 1: Exhaustive search over all 2^(m+1) chord assignments
    # For m=4: 32, m=5: 64, m=7: 256, m=9: 1024
    for mask in range(2 ** num_chords):
        if time.time() > deadline:
            break
        
        base_signs = np.ones(num_edges, dtype=int)
        for j, edge_idx in enumerate(chord_edge_indices):
            if (mask >> j) & 1:
                base_signs[edge_idx] = -1
        
        signs = gauge_fix(base_signs)
        try_signs(signs)
    
    # Phase 2: Local search with single and double flips
    if best_signs is not None:
        current = best_signs.copy()
        current_radius = best_radius
        
        # Single flip local search
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for i in range(num_edges):
                if time.time() > deadline:
                    break
                trial = current.copy()
                trial[i] *= -1
                r = evaluate(trial)
                if r < current_radius - 1e-12:
                    current = trial
                    current_radius = r
                    improved = True
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = current.copy()
        
        # Double flip local search
        if time.time() < deadline - 10:
            for i in range(num_edges):
                if time.time() > deadline:
                    break
                for j in range(i+1, num_edges):
                    if time.time() > deadline:
                        break
                    trial = best_signs.copy()
                    trial[i] *= -1
                    trial[j] *= -1
                    r = evaluate(trial)
                    if r < best_radius - 1e-12:
                        best_radius = r
                        best_signs = trial.copy()
    
    # Phase 3: Random restarts with local search
    if time.time() < deadline - 15:
        rng = np.random.default_rng(42)
        for restart in range(500):
            if time.time() > deadline:
                break
            
            signs = rng.choice([-1, 1], size=num_edges).astype(int)
            r = evaluate(signs)
            
            # Quick local search
            for _ in range(100):
                if time.time() > deadline:
                    break
                improved = False
                for i in range(num_edges):
                    trial = signs.copy()
                    trial[i] *= -1
                    r2 = evaluate(trial)
                    if r2 < r - 1e-12:
                        signs = trial
                        r = r2
                        improved = True
                        break
                if not improved:
                    break
            
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
    
    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=int)
    
    return best_signs
# EVOLVE-BLOCK-END
