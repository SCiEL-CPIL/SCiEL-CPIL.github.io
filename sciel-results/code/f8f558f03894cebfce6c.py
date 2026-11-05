import itertools
import time
import numpy as np

def _is_regular_induced(subset_mask, adj_masks, n):
    """
    Check if the induced subgraph on vertices in subset_mask is regular.
    """
    vertices = []
    m = subset_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        vertices.append(v)
        m ^= lsb
        
    k = len(vertices)
    if k < 6:
        return False
        
    # Calculate degrees within the induced subgraph
    degs = []
    for v in vertices:
        common = adj_masks[v] & subset_mask
        d = common.bit_count()
        degs.append(d)
        
    first_deg = degs[0]
    for d in degs[1:]:
        if d != first_deg:
            return False
    return True

def _check_graph_validity(n, adj_masks):
    """
    Check if the graph has no regular induced subgraph of size >= 6.
    """
    for k in range(6, n + 1):
        for subset in itertools.combinations(range(n), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if _is_regular_induced(mask, adj_masks, n):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Strategy: 
    # 1. Try a known valid 20-vertex graph. 
    #    The existence of such graphs is established. 
    #    We can construct a candidate using a specific deterministic pattern or 
    #    a pre-computed adjacency list. 
    #    
    #    A common source for such graphs is the complement of a design or specific 
    #    strongly regular graph modifications. 
    #    However, rather than relying on external data, we can use a robust 
    #    randomized local search that is extremely fast due to NumPy acceleration 
    #    for the verification step, or simply hardcode a valid graph if we can 
    #    derive one reliably.
    #
    #    Let's use a fast local search with NumPy for the initial check.
    #    If we find a valid graph for n=20, return it.
    #    If not, try n=19, etc.
    
    def fast_check_validity(adj_matrix, n):
        """
        Use NumPy to check for regular induced subgraphs of size 6.
        If no size 6 regular subgraph exists, we still need to check larger sizes,
        but size 6 is the primary bottleneck for validity in random-like graphs.
        Actually, we must check ALL sizes >= 6.
        However, for a random graph, if it avoids size 6, it's very likely to avoid larger ones 
        or we can check larger ones separately.
        
        For n=20, C(20,6) = 38760. 
        We can vectorize the degree calculation for all size-6 subsets.
        """
        # Precompute all size 6 combinations
        # This is done once per n
        combos_6 = list(itertools.combinations(range(n), 6))
        if not combos_6:
            return True
            
        # Convert to numpy array of indices
        # shape (num_combos, 6)
        idx_6 = np.array(combos_6)
        
        # For each subset, we need the degree of each vertex in the subset.
        # deg[v] in subset S = sum(adj_matrix[v, u] for u in S)
        # We can compute this by multiplying the adjacency matrix with indicator vectors.
        
        # Create indicator matrix for all subsets? 
        # num_combos x n
        # This might be large: 38760 x 20 = 775,200 floats. Manageable.
        
        # Actually, simpler:
        # For each vertex v, we want to know how many neighbors it has in each subset.
        # Let A be the adjacency matrix (n x n).
        # Let X be a matrix where X[i, v] is 1 if v is in combo i.
        # Degrees D[i, v] = sum_u A[v, u] * X[i, u] = A[v, :] dot X[i, :].
        # So D = X @ A.T ? No.
        # D[i, v] = sum_u X[i, u] * A[u, v]. 
        # So D = X @ A.
        
        # Build X
        X = np.zeros((len(combos_6), n), dtype=np.int8)
        for i, combo in enumerate(combos_6):
            for v in combo:
                X[i, v] = 1
                
        # A is n x n
        # D is (num_combos, n)
        D = X @ adj_matrix
        
        # For each combo i, the degrees of the vertices in the combo are D[i, combo]
        # We need to check if D[i, combo] are all equal.
        
        # Extract degrees for the specific vertices in each combo
        # D_subset[i, j] = D[i, idx_6[i, j]]
        # This is a fancy indexing operation.
        D_subset = D[np.arange(len(combos_6))[:, None], idx_6]
        
        # Check if all columns in each row are equal
        # row is regular if max(row) == min(row)
        maxes = np.max(D_subset, axis=1)
        mins = np.min(D_subset, axis=1)
        
        regular_6 = (maxes == mins)
        
        if np.any(regular_6):
            return False
            
        # If no size 6 regular subgraphs, check larger sizes?
        # For n=20, checking all sizes >= 6 is expensive in Python loops.
        # But if a graph is "random-like" and has no size 6 regular subgraphs, 
        # it is highly likely to be valid. 
        # However, to be strictly correct, we must verify.
        # Let's do a full check in Python for the candidates that pass the NumPy filter.
        
        return True

    def full_check_python(n, adj_masks):
        return _check_graph_validity(n, adj_masks)

    def to_adj_masks(adj_matrix):
        n = adj_matrix.shape[0]
        masks = [0] * n
        for i in range(n):
            m = 0
            for j in range(n):
                if adj_matrix[i, j]:
                    m |= (1 << j)
            masks[i] = m
        return masks

    def masks_to_edges(masks, n):
        edges = []
        for i in range(n):
            m = masks[i]
            for j in range(i + 1, n):
                if m & (1 << j):
                    edges.append([i, j])
        return edges

    # Search for n=20 first
    # We will use a randomized local search to find a valid graph.
    # Start with a random graph.
    # Objective: Minimize number of regular induced subgraphs of size 6.
    # If 0, check full validity.
    
    import random
    
    for target_n in [20, 19, 18]:
        if time.time() > deadline:
            break
            
        # Try multiple random restarts
        found = False
        for restart in range(100):
            if time.time() > deadline:
                break
                
            # Initialize random graph
            # Probability 0.5
            adj_matrix = (np.random.rand(target_n, target_n) < 0.5).astype(np.int8)
            # Ensure symmetry and no self-loops
            adj_matrix = np.triu(adj_matrix, 1)
            adj_matrix = adj_matrix + adj_matrix.T
            np.fill_diagonal(adj_matrix, 0)
            
            # Check size 6 violations using NumPy
            if fast_check_validity(adj_matrix, target_n):
                # Potential candidate. Verify fully with Python
                masks = to_adj_masks(adj_matrix)
                if full_check_python(target_n, masks):
                    edges = masks_to_edges(masks, target_n)
                    return {'n': target_n, 'edges': edges}
                    
            # Local Search: Flip edges to reduce violations
            # We can't easily use the NumPy check for incremental updates without rebuilding X or D.
            # So we stick to restarts for simplicity and speed of implementation.
            # 100 restarts for n=20 is fast.
            
    # If we failed to find 20-18, try a smaller n with more effort or just return a valid small graph
    # n=5 is always valid (no subset of size 6)
    # Let's try to find a valid n=10 just in case, though 20 should be found.
    
    for target_n in [10, 9, 8, 7, 6, 5]:
        if time.time() > deadline:
            break
        # For small n, random search is very likely to succeed or we can just return n=5
        if target_n == 5:
            return {'n': 5, 'edges': []}
            
        for restart in range(50):
            adj_matrix = (np.random.rand(target_n, target_n) < 0.5).astype(np.int8)
            adj_matrix = np.triu(adj_matrix, 1)
            adj_matrix = adj_matrix + adj_matrix.T
            np.fill_diagonal(adj_matrix, 0)
            
            masks = to_adj_masks(adj_matrix)
            if full_check_python(target_n, masks):
                edges = masks_to_edges(masks, target_n)
                return {'n': target_n, 'edges': edges}
                
    # Fallback
    return {'n': 5, 'edges': []}
