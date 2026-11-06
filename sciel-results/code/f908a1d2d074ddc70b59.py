import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 285  # Safety margin

    # Seed: Paley 37
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges_37 = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]

    target_n = 43

    # Precompute all 5-combinations as a NumPy array for vectorized checking
    comb_43 = np.array(list(combinations(range(target_n), 5)), dtype=np.int32)  # shape (962598, 5)

    # Build adjacency matrix as a NumPy boolean matrix for fast checking
    def build_adj_matrix(adj_list):
        """Convert adjacency list (bitmasks) to a numpy boolean adjacency matrix."""
        n = len(adj_list)
        mat = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                if adj_list[i] & (1 << j):
                    mat[i, j] = True
                    mat[j, i] = True
        return mat

    def mat_to_adj_list(mat):
        """Convert numpy boolean adjacency matrix back to adjacency list (bitmasks)."""
        n = mat.shape[0]
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if mat[i, j]:
                    adj[i] |= 1 << j
                    adj[j] |= 1 << i
        return adj

    def check_violation(mat):
        """
        Check if mat (numpy bool adjacency matrix) has a K5 or independent 5-set.
        Returns (is_valid, violation_5set or None).
        Uses vectorized edge counting over all 5-subsets.
        """
        # comb_43 has shape (N, 5)
        # For each 5-subset, count edges among the 5 vertices
        # Edge between vertices a,b is mat[a,b]
        # We need to check all 10 pairs in each 5-set
        
        v0 = comb_43[:, 0]
        v1 = comb_43[:, 1]
        v2 = comb_43[:, 2]
        v3 = comb_43[:, 3]
        v4 = comb_43[:, 4]
        
        # Count edges for all 10 pairs
        e01 = mat[v0, v1]
        e02 = mat[v0, v2]
        e03 = mat[v0, v3]
        e04 = mat[v0, v4]
        e12 = mat[v1, v2]
        e13 = mat[v1, v3]
        e14 = mat[v1, v4]
        e23 = mat[v2, v3]
        e24 = mat[v2, v4]
        e34 = mat[v3, v4]
        
        edge_counts = (e01 + e02 + e03 + e04 + e12 + e13 + e14 + e23 + e24 + e34).astype(np.int32)
        
        # Find violations: edge_count == 10 (clique) or edge_count == 0 (independent)
        clique_mask = (edge_counts == 10)
        indep_mask = (edge_counts == 0)
        
        if np.any(clique_mask):
            idx = np.argmax(clique_mask)  # first True
            return False, tuple(comb_43[idx])
        if np.any(indep_mask):
            idx = np.argmax(indep_mask)
            return False, tuple(comb_43[idx])
        
        return True, None

    # Initialize with Paley(37) + random vertices
    def init_graph(n_vertices, density=0.5):
        adj = [0] * n_vertices
        # Fill 0-36 with Paley
        for i in range(37):
            for j in range(i + 1, 37):
                if (j - i) % 37 in residues:
                    adj[i] |= 1 << j
                    adj[j] |= 1 << i
        # Fill 37-42 with random edges
        for i in range(37, n_vertices):
            for j in range(n_vertices):
                if i == j:
                    continue
                if random.random() < density:
                    adj[i] |= 1 << j
                    adj[j] |= 1 << i
        return adj

    def local_search(adj, max_iters=200):
        """
        Local search: repeatedly find a violation and flip an edge in it.
        Uses vectorized checking for speed.
        """
        mat = build_adj_matrix(adj)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            valid, viol = check_violation(mat)
            if valid:
                return adj, True
            
            # Flip a random edge in the violation
            v0, v1, v2, v3, v4 = viol
            pairs = [(v0, v1), (v0, v2), (v0, v3), (v0, v4),
                     (v1, v2), (v1, v3), (v1, v4),
                     (v2, v3), (v2, v4), (v3, v4)]
            u, v = random.choice(pairs)
            
            # Flip edge in both mat and adj
            mat[u, v] = not mat[u, v]
            mat[v, u] = not mat[v, u]
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
        
        return adj, False

    # Try multiple restarts with different densities
    densities = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    for density in densities:
        if time.time() > deadline - 30:
            break
        
        # Try a few restarts at this density
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            adj = init_graph(target_n, density)
            adj, found = local_search(adj, max_iters=500)
            
            if found:
                # Convert to edge list
                edges = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if adj[i] & (1 << j):
                            edges.append((i, j))
                return target_n, edges
    
    # Fallback to 37
    return 37, edges_37
