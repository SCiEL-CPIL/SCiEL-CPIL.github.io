import time
import numpy as np
from itertools import combinations
import scipy.optimize as opt

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_n = 0
    best_edges = []
    min_edge_count = float('inf')
    
    # We want to minimize edges. 
    # n=9: 126 edges, 512 colorings.
    # n=10: 252 edges, 1024 colorings.
    # n=11: 462 edges, 2048 colorings.
    # n=12: 792 edges, 4096 colorings.
    
    # The known best is 51 edges. 
    # Smaller n might force more edges? 
    # Actually, for fixed k, increasing n usually allows for sparser non-2-colorable hypergraphs 
    # or at least doesn't strictly increase the minimum edge count immediately, but the search space grows.
    # However, the "best known construction" of 51 edges is likely for a specific n.
    # Let's try n=11 first as it's a sweet spot for ILP size vs problem difficulty.
    # If n=11 yields >51, try n=12? Or n=10?
    # Let's try n=11, then n=10, then n=12, then n=9.
    
    for n in [11, 10, 12, 9]:
        if time.time() > deadline:
            break
            
        vertices = list(range(n))
        all_edges = list(combinations(vertices, 5))
        num_edges = len(all_edges)
        num_colorings = 1 << n
        
        # Build the incidence matrix A (num_colorings x num_edges)
        # A[c, e] = 1 if coloring c is monochromatic on edge e
        # We need to cover all colorings: sum_e A[c, e] * x[e] >= 1 for all c
        
        # Optimization: Represent colorings as integers 0..2^n-1.
        # Edge e is a tuple of 5 vertices.
        # Coloring c is monochromatic on e if all bits in e are 0 or all are 1.
        
        # We can compute the mask for each edge efficiently.
        # Instead of a dense matrix, we can use the ILP directly with the constraints generated.
        # But milp expects a matrix.
        
        # For n=12, 4096 x 792 matrix is ~3.2M floats. 25MB. Acceptable.
        # For n=13, 8192 x 1287 is ~10M floats. 80MB. Might be slow.
        
        # Let's construct the matrix.
        # To speed up, we can iterate colorings and check edges, or edges and check colorings.
        # Checking edges is better if we precompute vertex masks.
        
        # A[c, e] = 1 iff (c & mask_e) == 0 or (c & mask_e) == mask_e
        
        # We will create the matrix column by column (per edge) or row by row (per coloring).
        # Row by row (per coloring) might be faster in Python loops if vectorized per row?
        # Actually, let's use numpy broadcasting if possible, but memory might blow up.
        # num_colorings * num_edges is small enough for dense.
        
        # Let's precompute vertex bit masks for all edges
        edge_masks = []
        for e in all_edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
        
        # Create the constraint matrix A
        # A has shape (num_colorings, num_edges)
        # We fill it by iterating over edges (columns) because the condition depends on the edge mask.
        
        # Initialize A as zeros
        # Note: milp requires 'c' (objective), 'A' (constraints), 'b' (bounds), 'integrality', 'bounds'
        
        # Objective: min sum(x)
        c_obj = np.ones(num_edges)
        
        # Constraints: A @ x >= 1
        # We need A such that A[c, e] = 1 if c is mono on e.
        
        # Efficient construction:
        # For each edge e, we can find which colorings are monochromatic.
        # The set of monochromatic colorings for edge e (vertices v1..v5) is:
        # { c | c restricted to {v1..v5} is 00000 or 11111 }
        # There are 2^(n-5) * 2 = 2^(n-4) such colorings for each edge.
        
        # Let's fill A column by column.
        A = np.zeros((num_colorings, num_edges), dtype=np.float64)
        
        for j, mask in enumerate(edge_masks):
            # Find all colorings c such that (c & mask) == 0 or (c & mask) == mask
            # We can iterate through all colorings? 4096 * 792 = 3.2M ops. Fast.
            # Or generate them?
            # Iterating is simpler and fast enough.
            
            # Vectorized check for this column?
            # all_colorings = np.arange(num_colorings)
            # vals = all_colorings & mask
            # A[:, j] = ((vals == 0) | (vals == mask)).astype(float)
            
            # Let's do it vectorized for speed
            all_colorings = np.arange(num_colorings, dtype=np.int32)
            vals = (all_colorings & mask).astype(np.int32)
            A[:, j] = ((vals == 0) | (vals == mask)).astype(np.float64)
            
        # Solve ILP
        # Variables x[j] are binary
        integrality = np.ones(num_edges)
        bounds = [(0, 1)] * num_edges
        
        # Constraints: A x >= 1
        # milp uses standard form: A_ub x <= b_ub, A_eq x = b_eq
        # We have A x >= 1  =>  -A x <= -1
        A_ub = -A
        b_ub = -np.ones(num_colorings)
        
        try:
            res = opt.milp(
                c=c_obj,
                integrality=integrality,
                bounds=bounds,
                constraints=[opt.LinearConstraint(A_ub, ub=b_ub)]
            )
            
            if res.status == 0: # Optimal solution found
                # res.x contains the optimal values
                # Due to floating point, we might need to round
                x_opt = res.x
                # Select edges where x_opt[j] > 0.5
                selected_indices = np.where(x_opt > 0.5)[0]
                selected_edges = [all_edges[i] for i in selected_indices]
                
                # Verify the solution just in case (optional but safe)
                # Check if union of masks covers all colorings
                covered = np.zeros(num_colorings, dtype=bool)
                for idx in selected_indices:
                    mask = edge_masks[idx]
                    # Add colorings mono on this edge
                    # We can reuse the logic or just trust the ILP if status is optimal
                    # For safety, let's do a quick check if time permits
                    pass # Skipping verification for speed, ILP is exact
                
                if len(selected_edges) < min_edge_count:
                    min_edge_count = len(selected_edges)
                    best_n = n
                    best_edges = selected_edges
                    
        except Exception as e:
            # If ILP fails, we could fall back to greedy, but ILP is preferred.
            # For now, just continue to next n.
            pass

    if best_edges:
        return {'n': best_n, 'edges': [list(e) for e in best_edges]}
    else:
        # Fallback: Return a known valid structure if ILP failed for all n
        # This is unlikely given the sizes, but good for robustness.
        # Using n=9 full graph as a guaranteed non-2-colorable hypergraph (126 edges)
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
