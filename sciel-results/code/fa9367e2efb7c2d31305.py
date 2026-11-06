import time
import itertools
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Helper: Check if a graph is valid (no regular induced subgraph of size >= 6)
    # Returns True if valid, False otherwise.
    # Uses NumPy vectorization for speed.
    def is_valid(n, adj_matrix):
        # adj_matrix: n x n boolean numpy array
        # We need to check all subsets S of size k >= 6.
        # If any S is regular, return False.
        
        # Optimization: 
        # For a subset S of size k, let A_S be the kxk submatrix.
        # Degrees are diag(A_S).
        # We check if all diag elements are equal.
        
        # Iterate k from 6 to n
        for k in range(6, n + 1):
            # Generate all combinations of size k
            # This is the bottleneck. 
            # For n=20, k=6: C(20,6) = 38760
            # For n=20, k=10: C(20,10) = 184756
            # Total subsets ~ 600k.
            # NumPy indexing for each is fast, but overhead might be high.
            
            # Alternative: Iterate over all masks? No, combinations is better for small k.
            # But for large k, we can use complements? No, regularity isn't preserved by complement easily.
            
            # Let's try to vectorize the check for a fixed k.
            # We can't easily vectorize the combination generation itself in pure numpy.
            # We will loop combinations, but use numpy for the degree check.
            
            # Precompute? 
            # Actually, for n=20, 600k subsets * (k operations) is too slow in Python loops even with numpy inside.
            # We need a smarter check.
            
            # Observation:
            # If a graph has a regular induced subgraph of size k, it must have specific degree sequences.
            
            # Let's stick to the loop but optimize the inner check.
            # For each subset, we need to check if degrees are uniform.
            
            # A faster way: 
            # Precompute all masks? No.
            
            # Let's try a different angle: 
            # Random sampling of subsets? No, must be exact.
            
            # Given the time limit, and that n=20 is the target, 
            # we should only run this full check on candidate graphs that pass a faster heuristic.
            
            # For now, this function is for final verification.
            # It might be too slow for n=20 if called many times.
            # We will only call it on the final candidate or very few candidates.
            
            for subset in itertools.combinations(range(n), k):
                mask = np.zeros(n, dtype=bool)
                for v in subset:
                    mask[v] = True
                
                # Extract subgraph
                # Degrees within subset
                degs = (adj_matrix * mask[:, None]).sum(axis=1)
                # Only care about vertices in subset
                degs_sub = degs[mask]
                
                # Check if all equal
                if len(set(degs_sub)) == 1:
                    return False
        
        return True

    # Helper: Construct a specific graph.
    # Strategy: 
    # 1. Start with a known valid small graph (e.g., Paley 17 or 13).
    # 2. Use a greedy incremental approach to add vertices.
    # 3. When adding a vertex, only check subsets containing the new vertex.
    # 4. Use a fast check for the new vertex addition.
    
    def make_paley(p):
        if p % 4 != 1:
            return None
        if p < 2: return None
        for i in range(2, int(p**0.5)+1):
            if p % i == 0: return None
        qrs = set((i*i) % p for i in range(1, p))
        edges = []
        for u in range(p):
            for v in range(u+1, p):
                if (v - u) % p in qrs:
                    edges.append((u, v))
        return edges

    # Base candidate: Paley 17
    p17_edges = make_paley(17)
    if p17_edges is None:
        return {'n': 5, 'edges': []}
        
    n_base = 17
    # Build adjacency matrix for base
    adj_base = np.zeros((n_base, n_base), dtype=bool)
    for u, v in p17_edges:
        adj_base[u, v] = True
        adj_base[v, u] = True
        
    # Verify base graph is valid (it should be, but let's be sure with a fast check)
    # For n=17, full check is feasible.
    if not is_valid(n_base, adj_base):
        # Fallback to smaller
        p13_edges = make_paley(13)
        if p13_edges:
            n_base = 13
            adj_base = np.zeros((n_base, n_base), dtype=bool)
            for u, v in p13_edges:
                adj_base[u, v] = True
                adj_base[v, u] = True
            if not is_valid(n_base, adj_base):
                return {'n': 5, 'edges': []}
        else:
            return {'n': 5, 'edges': []}
    
    best_n = n_base
    best_adj = adj_base.copy()
    
    # Target n=20 or 21
    target_n = 21
    
    # Greedy extension
    # We want to add vertices one by one.
    # For each new vertex, we try a set of candidate masks (connections to existing vertices).
    # We check if adding this vertex creates a regular induced subgraph of size >= 6.
    
    # Fast check for adding a vertex:
    # Let G be current graph on n vertices.
    # Add vertex v_new with mask m (length n).
    # New graph G' on n+1 vertices.
    # We only need to check subsets S containing v_new.
    # S = {v_new} U T, where T is a subset of {0..n-1}.
    # |S| >= 6 => |T| >= 5.
    # For S to be regular, all vertices in S must have the same degree within S.
    
    # deg(v_new) in S = popcount(m & mask_T)
    # deg(u) in S for u in T = deg(u in T) + (1 if u in m else 0)
    
    # We need to check all T of size 5..n.
    # This is still expensive if done naively.
    # But we can prune:
    # If we fix m, we can compute the "potential" regularity.
    
    # Let's implement a fast check using NumPy for the new vertex addition.
    
    def can_add_vertex(adj, n, m):
        # adj: n x n boolean array
        # m: length n boolean array (connections of new vertex)
        # Returns True if adding vertex is safe (no new regular subgraph of size >= 6)
        
        # We need to check all subsets T of {0..n-1} with |T| >= 5.
        # Let's iterate k from 5 to n.
        
        for k in range(5, n + 1):
            # For each subset T of size k
            # We need to check if {new} U T is regular.
            
            # Optimization: 
            # deg_new = popcount(m & mask_T)
            # degs_T = (adj * mask_T[:, None]).sum(axis=1) + m[mask_T]
            # We need deg_new == degs_T[i] for all i in T.
            
            # This is equivalent to:
            # Let d = deg_new.
            # For all u in T, deg_u_in_T + (1 if u in m else 0) == d.
            
            # We can vectorize this check for a given T.
            
            for T in itertools.combinations(range(n), k):
                mask_T = np.zeros(n, dtype=bool)
                for v in T:
                    mask_T[v] = True
                
                # deg_new
                d_new = (m & mask_T).sum()
                
                # degs of vertices in T within the induced subgraph {new} U T
                # deg_u = (adj[u, T] sum) + (m[u] if u in T else 0)
                # adj[u, T] is a slice.
                
                # Vectorized calculation of degrees for all vertices in T
                # degs_all = (adj * mask_T[:, None]).sum(axis=1)
                # degs_T = degs_all[mask_T] + m[mask_T]
                
                degs_all = (adj * mask_T[:, None]).sum(axis=1)
                degs_T = degs_all[mask_T] + m[mask_T]
                
                # Check if all degs_T are equal to d_new
                if (degs_T == d_new).all():
                    return False
        
        return True

    # Now, try to extend to target_n
    current_n = n_base
    current_adj = best_adj.copy()
    
    random.seed(42)
    
    # We will try to add vertices one by one until we reach target_n or fail.
    # For each new vertex, we try several random masks and specific masks.
    
    success_reached = current_n
    current_edges = []
    # Rebuild edges from adj for output
    def get_edges(adj, n):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    edges.append([u, v])
        return edges
    
    # We need to track the actual edges to return.
    # It's easier to track the adjacency matrix and convert at the end.
    
    for new_v_idx in range(current_n, target_n):
        if time.time() > deadline - 20:
            break
            
        found_valid_mask = False
        # Try a few strategies for the mask:
        # 1. Random masks
        # 2. Sparse masks (few connections)
        # 3. Dense masks
        
        attempts = 0
        while not found_valid_mask and attempts < 50:
            attempts += 1
            if time.time() > deadline - 20:
                break
                
            # Generate a random mask
            # Bias towards random?
            m = np.random.rand(current_n) < 0.5
            
            if can_add_vertex(current_adj, current_n, m):
                # Accept this mask
                # Update adjacency matrix
                new_adj = np.zeros((current_n + 1, current_n + 1), dtype=bool)
                new_adj[:current_n, :current_n] = current_adj
                new_adj[:current_n, current_n] = m
                new_adj[current_n, :current_n] = m
                # Note: no self-loop, so new_adj[current_n, current_n] = 0
                
                current_adj = new_adj
                current_n += 1
                success_reached = current_n
                found_valid_mask = True
        
        if not found_valid_mask:
            # Try a few more specific masks if random failed?
            # For now, break.
            break
        
    # Final verification of the best graph found
    # This is crucial to ensure validity.
    # If the greedy check was correct, this should pass.
    # But let's be safe.
    
    if is_valid(current_n, current_adj):
        final_edges = get_edges(current_adj, current_n)
        return {'n': current_n, 'edges': final_edges}
    else:
        # Fallback to base if final check fails (shouldn't happen)
        base_edges = get_edges(best_adj, n_base)
        return {'n': n_base, 'edges': base_edges}
