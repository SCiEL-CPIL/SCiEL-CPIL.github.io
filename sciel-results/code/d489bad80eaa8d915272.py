import time
import random
import numpy as np
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s margin for safety

    def is_c4_free(adj):
        """Check if a graph is C4-free using matrix multiplication."""
        A = adj.astype(np.int32)
        A2 = A @ A
        # For C4-free, no two distinct vertices can have 2+ common neighbors
        # A2[i,j] for i!=j is the number of common neighbors of i and j
        # If any A2[i,j] >= 2 for i != j, there's a C4
        n = adj.shape[0]
        # Mask out diagonal
        mask = ~np.eye(n, dtype=bool)
        return not np.any(A2[mask] >= 2)

    def get_min_degree(adj):
        if adj.size == 0:
            return 0
        return int(np.min(adj.sum(axis=1)))

    def random_graph_c4_free(n, target_deg, max_time=5.0):
        """
        Construct a C4-free graph on n vertices with min degree >= target_deg.
        
        Strategy: Start with empty graph, iteratively add edges.
        An edge (u,v) can be added iff u and v have at most 1 common neighbor.
        We prioritize adding edges to the vertex with currently minimum degree.
        Use randomized tie-breaking to explore different graph structures.
        """
        adj = np.zeros((n, n), dtype=bool)
        degrees = np.zeros(n, dtype=int)
        
        # Precompute all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # We'll do multiple passes, shuffling edge order each time
        # to break symmetry and explore different solutions
        attempts = 0
        max_attempts = 50
        
        while attempts < max_attempts:
            attempts += 1
            if time.time() - start_time > deadline - 15:
                break
            
            adj = np.zeros((n, n), dtype=bool)
            degrees = np.zeros(n, dtype=int)
            
            # Random shuffle of edges for this attempt
            random.shuffle(all_edges)
            
            # Greedy: process edges in shuffled order
            for u, v in all_edges:
                if time.time() - start_time > deadline - 15:
                    break
                
                # Check if adding (u,v) would create a C4
                # This happens iff u and v have >= 2 common neighbors
                common = np.sum(adj[u] & adj[v])
                if common <= 1:
                    # Safe to add
                    adj[u, v] = True
                    adj[v, u] = True
                    degrees[u] += 1
                    degrees[v] += 1
                
                # Early exit if all vertices meet target degree
                if np.min(degrees) >= target_deg:
                    break
            
            # Check if we achieved the target
            if np.min(degrees) >= target_deg:
                # Verify C4-free (should be, but double-check)
                if is_c4_free(adj):
                    return adj
        
        return None

    def random_graph_with_priority(n, target_deg, max_time=5.0):
        """
        Alternative: Priority-based edge addition.
        At each step, find the vertex with minimum degree that hasn't met target,
        and try to connect it to a vertex with fewest common neighbors.
        """
        adj = np.zeros((n, n), dtype=bool)
        degrees = np.zeros(n, dtype=int)
        
        # Track which vertices still need edges
        needs_edge = np.ones(n, dtype=bool)
        
        while np.any(needs_edge):
            if time.time() - start_time > deadline - 15:
                break
            
            # Find vertex with minimum degree among those needing edges
            deficient = np.where(needs_edge)[0]
            if len(deficient) == 0:
                break
            
            u = deficient[np.argmin(degrees[deficient])]
            
            if degrees[u] >= target_deg:
                needs_edge[u] = False
                continue
            
            # Find best v to connect to u
            # v must not be adjacent to u, and common_neighbors(u,v) <= 1
            candidates = []
            for v in range(n):
                if v == u or adj[u, v]:
                    continue
                common = np.sum(adj[u] & adj[v])
                if common <= 1:
                    candidates.append((common, v))
            
            if not candidates:
                # Can't add any edge to u. Try removing an edge elsewhere?
                # For now, just mark as stuck and break
                break
            
            # Pick candidate with fewest common neighbors (random tie-break)
            min_common = min(c[0] for c in candidates)
            best_candidates = [v for c, v in candidates if c == min_common]
            v = random.choice(best_candidates)
            
            adj[u, v] = True
            adj[v, u] = True
            degrees[u] += 1
            degrees[v] += 1
            
            if degrees[u] >= target_deg:
                needs_edge[u] = False
            if degrees[v] >= target_deg:
                needs_edge[v] = False
        
        if np.min(degrees) >= target_deg and is_c4_free(adj):
            return adj
        
        return None

    # Try to construct for n=47, then n=46, etc.
    best_result = None
    
    for n in [47, 46, 45, 44]:
        if time.time() - start_time > deadline - 10:
            break
        
        target_deg = n - 39
        if target_deg < 0:
            target_deg = 0
        
        # Try multiple methods
        adj = None
        
        # Method 1: Random shuffle greedy
        adj = random_graph_c4_free(n, target_deg)
        
        if adj is None:
            # Method 2: Priority-based
            adj = random_graph_with_priority(n, target_deg)
        
        if adj is not None:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i, j]:
                        edges.append([i, j])
            best_result = {'n': n, 'edges': edges}
            break
    
    # Fallback: Return a valid small graph
    if best_result is None:
        # n=39, empty graph: min degree 0 >= 39-39=0
        best_result = {'n': 39, 'edges': []}
    
    return best_result
