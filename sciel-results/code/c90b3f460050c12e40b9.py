import numpy as np
import time

def sign_edges(graph, budget):
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    # Identify edge types: top cycle, bottom cycle, matching
    top_edges = []
    bottom_edges = []
    match_edges = []
    
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_edges.append(idx)
        elif u >= m and v >= m:
            bottom_edges.append(idx)
        else:
            match_edges.append(idx)
            
    top_edges.sort()
    bottom_edges.sort()
    match_edges.sort()
    
    num_top = len(top_edges)
    num_bottom = len(bottom_edges)
    
    # Pre-compute the base adjacency matrix structure
    # We will modify the signs of cycle edges and keep matching edges fixed at +1
    # Actually, we can vary matching signs too, but symmetry suggests fixing them helps?
    # Let's stick to the strategy of the current program which fixed matching to +1 and searched cycles.
    # However, the inspiration suggests searching all 2^(2m) cycle signs.
    # Let's optimize the spectral radius calculation.
    
    # The signed adjacency matrix A is symmetric.
    # We can construct it quickly.
    # A[i][j] = sign(edge_ij) if (i,j) is an edge, else 0.
    
    # To speed up, we can pre-allocate A and only update the cycle edges.
    # But since we are iterating over many masks, we need a very fast loop.
    
    # Key Insight: The spectrum of the signed adjacency matrix of a prism graph
    # can be related to the spectra of the signed cycles.
    # However, exhaustive search over 2^(2m) is feasible for m=9 (2^18 = 262k).
    # We need to make each iteration extremely fast.
    
    # Let's pre-compute the positions of top and bottom edges in the edge list.
    # And pre-compute the matching edges.
    
    # We will fix matching signs to +1.
    # We search over signs for top and bottom edges.
    
    # Pre-allocate the full sign vector
    signs = np.ones(num_edges, dtype=int)
    for idx in match_edges:
        signs[idx] = 1
        
    # Combined list of cycle edges to vary
    cycle_edges = top_edges + bottom_edges
    num_cycle = len(cycle_edges)
    
    # We fix the first cycle edge to +1 to break global sign symmetry
    # Flipping all signs in the lift results in the same spectrum (eigenvalues change sign, radius same)
    # So we can fix one sign.
    if num_cycle > 0:
        fixed_edge = cycle_edges[0]
        signs[fixed_edge] = 1
        variable_edges = cycle_edges[1:]
    else:
        variable_edges = []
        
    num_var = len(variable_edges)
    
    best_radius = float('inf')
    best_signs = None
    
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Pre-compute the base adjacency matrix with all +1
    # Then we just flip specific entries.
    # Actually, building A from scratch is O(E) = O(m), which is small.
    # eigvalsh is O(n^3) = O(18^3) ~ 5800 ops.
    # 262k * 5800 ~ 1.5e9 flops. NumPy is fast, but Python overhead is the killer.
    
    # Optimization:
    # We can use the fact that A is sparse and small.
    # But np.linalg.eigvalsh on a dense 18x18 matrix is very fast in C.
    # The bottleneck is the Python loop and array construction.
    
    # Let's try to vectorize the sign assignment if possible, or use a tight loop.
    
    # We'll iterate over masks.
    # For each mask, update the signs of variable_edges, build A, compute eigvalsh.
    
    # To further speed up, we can pre-compute the indices (u, v) for each variable edge.
    var_edge_info = [(edges[i][0], edges[i][1]) for i in variable_edges]
    
    # Also, we need to know which edges are top and which are bottom for connectivity check?
    # The task says: "The lift must be connected".
    # For a prism graph, the 2-lift is connected iff the product of signs on the top cycle is -1
    # OR the product of signs on the bottom cycle is -1.
    # (If both are +1, the lift splits into two components: (top,0)+bottom,0 and top,1+bottom,1? No.)
    # Let's verify:
    # Vertices: (v,0), (v,1).
    # Edge (u,v) with sign s: (u,0)-(v,s), (u,1)-(v,1-s). (Using 0/1 for sides, s=0 for +1, 1 for -1)
    # If all signs are +1 (s=0), then (u,0)-(v,0) and (u,1)-(v,1). The graph splits into two copies.
    # If product of top signs is -1, then traversing the top cycle flips the side, connecting the two halves.
    # So connectivity condition: prod_top == -1 OR prod_bottom == -1.
    
    # We can compute prod_top and prod_bottom quickly during the loop.
    # Or, since we are doing exhaustive search, we can just check connectivity by building the graph?
    # No, that's too slow.
    # The formula is: connected iff (prod_top * prod_bottom == -1) ?
    # Let's check.
    # If prod_top = 1 and prod_bottom = 1, disconnected.
    # If prod_top = -1 and prod_bottom = 1, connected.
    # If prod_top = 1 and prod_bottom = -1, connected.
    # If prod_top = -1 and prod_bottom = -1, connected.
    # So condition: NOT (prod_top == 1 AND prod_bottom == 1).
    
    # We can track prod_top and prod_bottom incrementally or just compute from the mask.
    # Since num_var is small (<= 17), we can compute products from the bits.
    
    # Let's map variable edges to top/bottom.
    top_var_indices = [] # indices in variable_edges that are top
    bottom_var_indices = [] # indices in variable_edges that are bottom
    
    for i, idx in enumerate(variable_edges):
        if idx in top_edges:
            top_var_indices.append(i)
        elif idx in bottom_edges:
            bottom_var_indices.append(i)
            
    # The fixed edge (if any) is in cycle_edges[0].
    # We need to know if it's top or bottom to include in the product.
    fixed_prod_top = 1
    fixed_prod_bottom = 1
    if num_cycle > 0:
        if fixed_edge in top_edges:
            # fixed sign is +1
            pass
        else:
            pass
        # Since fixed sign is +1, it doesn't change the product.
        # So we just need the product of the variable edges.
        
    # In the loop, for a given mask:
    # prod_top = product of signs of top variable edges
    # prod_bottom = product of signs of bottom variable edges
    # Sign is +1 if bit is 0, -1 if bit is 1.
    # So prod = (-1)^(number of 1s in the relevant bits).
    
    # Let's pre-compute for each mask the number of 1s in top bits and bottom bits?
    # No, just compute on the fly.
    
    # Optimization: Pre-compute the bitmasks for top and bottom variable indices.
    # Or just iterate.
    
    # Let's try the loop.
    
    for mask in range(1 << num_var):
        if time.time() > deadline:
            break
            
        # Update signs
        for i, edge_idx in enumerate(variable_edges):
            if (mask >> i) & 1:
                signs[edge_idx] = -1
            else:
                signs[edge_idx] = 1
                
        # Check connectivity
        # Count 1s in top_var_indices and bottom_var_indices
        # prod_top = (-1)^(count_top_ones)
        # prod_bottom = (-1)^(count_bottom_ones)
        
        # Quick count
        count_top = 0
        for i in top_var_indices:
            count_top += (mask >> i) & 1
            
        count_bottom = 0
        for i in bottom_var_indices:
            count_bottom += (mask >> i) & 1
            
        # If both counts are even, prod is +1, disconnected.
        if (count_top % 2 == 0) and (count_bottom % 2 == 0):
            continue
            
        # Build A
        A = np.zeros((n, n), dtype=float)
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
            
        # Compute spectral radius
        eigvals = np.linalg.eigvalsh(A)
        radius = max(abs(eigvals[0]), abs(eigvals[-1]))
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            
    # Fallback
    if best_signs is None:
        # This should not happen if we found any valid configuration.
        # If num_var == 0, we didn't loop.
        # For m=4, num_cycle=8, num_var=7. 2^7=128.
        # There must be valid configurations.
        # If we timed out before finding any, return a default valid one.
        # A known valid configuration: alternate signs on top cycle.
        # e.g., +1, -1, +1, -1...
        # Product will be (-1)^(m/2) if m even?
        # For m=4, +1,-1,+1,-1 -> prod = 1. Disconnected if bottom is also prod 1.
        # Let's just pick one that is known to work or random.
        # Random search with budget?
        # The task says budget=32. We can use the seed's approach if we fail.
        # But we should have found something in 2^17 iterations.
        
        # Emergency: return all +1 except one edge to make it connected?
        # If we set one edge to -1, prod becomes -1.
        signs = np.ones(num_edges, dtype=int)
        if num_edges > 0:
            signs[0] = -1
        best_signs = signs
        
    return best_signs
