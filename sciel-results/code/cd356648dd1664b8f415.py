# EVOLVE-BLOCK-START
import numpy as np

def sign_edges(graph, budget):
    """
    Assign signs to edges of a prism graph to minimize the spectral radius
    of the signed adjacency matrix, while ensuring the 2-lift is connected.
    
    Strategy: Enforce that the product of signs around fundamental cycles is -1
    to maximize frustration and minimize spectral radius.
    """
    from autoresearch_suite.science_tcs_expansion import check_lift
    
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Classify edges into top, bottom, and matching
    top_edge_indices = {}  # edge_idx -> (i, i+1 mod m)
    bottom_edge_indices = {}  # edge_idx -> (i, i+1 mod m) local
    match_edge_indices = {}  # edge_idx -> i
    
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            # Top cycle edge
            i = u % m
            j = v % m
            top_edge_indices[idx] = (i, j)
        elif u >= m and v >= m:
            # Bottom cycle edge
            i = (u - m) % m
            j = (v - m) % m
            bottom_edge_indices[idx] = (i, j)
        else:
            # Matching edge
            if u < m:
                match_edge_indices[idx] = u
            else:
                match_edge_indices[idx] = v - m
    
    # Build lookup maps
    top_idx_by_edge = {}
    for idx, (i, j) in top_edge_indices.items():
        top_idx_by_edge[(i, j)] = idx
        top_idx_by_edge[(j, i)] = idx
    
    bottom_idx_by_edge = {}
    for idx, (i, j) in bottom_edge_indices.items():
        bottom_idx_by_edge[(i, j)] = idx
        bottom_idx_by_edge[(j, i)] = idx
    
    match_idx_by_i = {}
    for idx, i in match_edge_indices.items():
        match_idx_by_i[i] = idx
    
    def build_signed_adjacency(signs):
        A = np.zeros((n, n))
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
        return A
    
    def spectral_radius(signs):
        A = build_signed_adjacency(signs)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_connected(signs):
        """
        Check if the 2-lift is connected by verifying the sign vector is not a coboundary.
        A sign vector is a coboundary iff all cycle products are +1.
        We check: top cycle, bottom cycle, and all ladder cycles.
        """
        # Top cycle product
        prod_top = 1
        for idx in top_edge_indices:
            prod_top *= signs[idx]
        
        # Bottom cycle product
        prod_bot = 1
        for idx in bottom_edge_indices:
            prod_bot *= signs[idx]
        
        # Ladder cycles: for each i, cycle is i -> i+1 -> m+i+1 -> m+i -> i
        # Edges: top(i, i+1), match(i+1), bottom(i, i+1), match(i)
        for i in range(m):
            i_next = (i + 1) % m
            try:
                t_idx = top_idx_by_edge[(i, i_next)]
                b_idx = bottom_idx_by_edge[(i, i_next)]
                m_i_idx = match_idx_by_i[i]
                m_next_idx = match_idx_by_i[i_next]
            except KeyError:
                continue
            
            prod_ladder = signs[t_idx] * signs[m_next_idx] * signs[b_idx] * signs[m_i_idx]
            if prod_ladder == -1:
                return True
        
        if prod_top == -1 or prod_bot == -1:
            return True
        
        return False
    
    # Generate candidate sign assignments based on cycle product constraints
    candidates = []
    
    # Strategy 1: For even m, try to set all fundamental cycle products to -1
    if m % 2 == 0:
        # Set alternating matching signs: match(i) = (-1)^i
        # Then for ladder i: top(i) * (-1)^(i+1) * bottom(i) * (-1)^i = top(i) * bottom(i) * (-1)
        # For ladder product to be -1: top(i) * bottom(i) * (-1) = -1 => top(i) * bottom(i) = 1
        # So set bottom(i) = top(i)
        # For top cycle product to be -1: product of all top(i) = -1
        # For bottom cycle product to be -1: product of all bottom(i) = product of all top(i) = -1 ✓
        
        s = np.ones(num_edges, dtype=int)
        for idx, i in match_edge_indices.items():
            if i % 2 == 1:
                s[idx] = -1
        
        top_idxs = list(top_edge_indices.keys())
        if top_idxs:
            s[top_idxs[0]] = -1
        
        for idx, (i, j) in bottom_edge_indices.items():
            try:
                t_idx = top_idx_by_edge[(i, j)]
                s[idx] = s[t_idx]
            except KeyError:
                s[idx] = 1
        
        if is_connected(s):
            candidates.append(s)
        
        s2 = np.ones(num_edges, dtype=int)
        for idx, i in match_edge_indices.items():
            if (i + 1) % 2 == 1:
                s2[idx] = -1
        
        top_idxs = list(top_edge_indices.keys())
        if top_idxs:
            s2[top_idxs[0]] = -1
        
        for idx, (i, j) in bottom_edge_indices.items():
            try:
                t_idx = top_idx_by_edge[(i, j)]
                s2[idx] = s2[t_idx]
            except KeyError:
                s2[idx] = 1
        
        if is_connected(s2):
            candidates.append(s2)
    
    # Strategy 2: For odd m, try various patterns
    s_a = np.ones(num_edges, dtype=int)
    match_idxs = list(match_edge_indices.keys())
    if match_idxs:
        s_a[match_idxs[0]] = -1
    if is_connected(s_a):
        candidates.append(s_a)
    
    s_b = np.ones(num_edges, dtype=int)
    for idx, i in match_edge_indices.items():
        if i % 2 == 1:
            s_b[idx] = -1
    top_idxs = list(top_edge_indices.keys())
    if top_idxs:
        s_b[top_idxs[0]] = -1
    if is_connected(s_b):
        candidates.append(s_b)
    
    s_c = np.ones(num_edges, dtype=int)
    for idx, i in match_edge_indices.items():
        if i % 2 == 1:
            s_c[idx] = -1
    if is_connected(s_c):
        candidates.append(s_c)
    
    rng = np.random.default_rng(3)
    for _ in range(200):
        s_d = rng.choice([-1, 1], size=num_edges).astype(int)
        if is_connected(s_d):
            candidates.append(s_d)
            if len(candidates) >= 10:
                break
    
    best_signs = None
    best_radius = float('inf')
    
    for s in candidates:
        try:
            r = spectral_radius(s)
            if r < best_radius:
                best_radius = r
                best_signs = s.copy()
        except Exception:
            continue
    
    if best_signs is None:
        s_fallback = np.ones(num_edges, dtype=int)
        if match_edge_indices:
            first_idx = next(iter(match_edge_indices.keys()))
            s_fallback[first_idx] = -1
        return s_fallback
    
    current_signs = best_signs.copy()
    current_radius = best_radius
    
    for iteration in range(200):
        improved = False
        order = np.random.permutation(num_edges)
        
        for idx in order:
            new_signs = current_signs.copy()
            new_signs[idx] = -new_signs[idx]
            
            if not is_connected(new_signs):
                continue
            
            r = spectral_radius(new_signs)
            if r < current_radius - 1e-12:
                current_signs = new_signs
                current_radius = r
                improved = True
                break
        
        if not improved:
            break
    
    return current_signs
# EVOLVE-BLOCK-END
