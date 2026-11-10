import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import sparse
import itertools
import time
import random

def construct():
    # 1. Setup vertices and edges
    n = 16
    edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    m = len(edges) # 120
    
    # 2. Define Q3 structure
    # Vertices 0..7
    q3_vertices = list(range(8))
    q3_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]
    
    # 3. Generate unique isomorphism patterns for Q3
    # We need 840 unique sets of 12 edges on 8 vertices
    patterns = set()
    # Precompute permutations? 8! = 40320.
    for p in itertools.permutations(q3_vertices):
        # p is a bijection from 0..7 to 0..7
        # Map q3_edges using p
        mapped_edges = []
        for u, v in q3_edges:
            mu, mv = p[u], p[v]
            if mu > mv:
                mu, mv = mv, mu
            mapped_edges.append((mu, mv))
        patterns.add(tuple(sorted(mapped_edges)))
    
    patterns = list(patterns)
    # Should be 840
    
    # 4. Precompute all Q3 subgraphs in K16 (for fallback and ILP)
    # This list will contain tuples of 12 edge indices
    q3_subgraphs = []
    # To speed up ILP matrix construction, we can also store the rows
    # But let's just generate the list first.
    
    # Time check
    start_time = time.time()
    
    # Generate all 8-subsets
    # For each subset, generate the 840 patterns mapped to actual edge indices
    # This is the bottleneck.
    
    # Optimization: 
    # Instead of building a huge list of 10M items in Python memory first,
    # we can try to build the sparse matrix directly or use the list for greedy.
    
    # Let's try to build the list of subgraphs.
    # 12870 subsets * 840 patterns = 10.8M items.
    # Each item is a tuple of 12 ints.
    # Memory: 10.8M * 12 * 8 bytes (if numpy) or ~100 bytes (python tuple) -> ~1GB.
    # This might be too much for standard Python lists.
    
    # Strategy:
    # 1. Try ILP.
    #    To do ILP, we need the matrix.
    #    We can build the matrix row by row? No, scipy sparse needs all data.
    #    We can use `lil_matrix` and append? Slow.
    #    We can use `coo_matrix` with pre-allocated arrays?
    
    # Let's estimate if we can afford the list for the greedy method.
    # If we use NumPy arrays for the edges in each subgraph:
    # `subgraph_edges` = np.array of shape (10800000, 12) of dtype int32.
    # 10.8M * 12 * 4 bytes = 518 MB.
    # This is feasible.
    
    # Let's generate this array.
    num_subsets = len(list(itertools.combinations(range(n), 8))) # 12870
    num_patterns = len(patterns) # 840
    total_q3 = num_subsets * num_patterns
    
    # Pre-allocate
    # We need to map (subset, pattern) to edge indices.
    
    # This generation loop is 10.8M iterations.
    # Inside: 12 lookups in dict.
    # 10.8M * 12 = 129.6M dict lookups.
    # In Python, this is ~ 10-20 seconds.
    
    # If time is tight, we might skip ILP and go straight to greedy?
    # But ILP might give a better score.
    # Let's try to do it in 30 seconds. If it takes longer, we abort ILP.
    
    # We will generate the data for the sparse matrix and the greedy list simultaneously.
    
    row_data = []
    col_data = []
    val_data = []
    
    # For greedy, we need the list of edge index tuples.
    # Storing 10.8M tuples in a list is memory heavy.
    # Let's store them in a flat numpy array.
    q3_edge_indices = np.empty((total_q3, 12), dtype=np.int32)
    
    current_row = 0
    idx = 0
    for subset in itertools.combinations(range(n), 8):
        # Map subset vertices to local indices 0..7
        # subset is a tuple of 8 vertices
        # We need to map the pattern edges (which use local indices 0..7) to actual vertices
        # Wait, the pattern `p` is a permutation of 0..7.
        # The pattern edges are defined on 0..7.
        # So for a pattern `pat` (tuple of 12 pairs of local indices),
        # and a subset `S` (tuple of 8 actual vertices),
        # the actual edges are (S[u], S[v]) for each (u,v) in pat.
        
        for pat in patterns:
            # pat is a tuple of 12 (u,v) where u,v in 0..7
            # We need to get the global edge index for each
            row_indices = []
            for u, v in pat:
                actual_u = subset[u]
                actual_v = subset[v]
                if actual_u > actual_v:
                    actual_u, actual_v = actual_v, actual_u
                e_idx = edge_to_idx[(actual_u, actual_v)]
                row_indices.append(e_idx)
                
                # Add to sparse matrix data
                row_data.append(current_row)
                col_data.append(e_idx)
                val_data.append(1)
            
            # Store for greedy
            q3_edge_indices[idx, :] = row_indices
            idx += 1
            
            current_row += 1
            
            # Time check
            if time.time() - start_time > 40:
                # Abort ILP construction, fall back to greedy with what we have?
                # No, we need the full list for greedy to be valid.
                # If we abort, we can't use ILP.
                # We can still use greedy if we generate the full list.
                # But if generation is too slow, we are stuck.
                # Let's assume it finishes in < 60s.
                pass

    # Build Sparse Matrix
    # A is (total_q3, m)
    # b is 11 * ones
    A = sparse.csr_matrix((val_data, (row_data, col_data)), shape=(total_q3, m))
    b = np.ones(total_q3) * 11
    
    # Objective: Maximize sum of x
    c = np.ones(m)
    
    # Integrality
    integrality = np.ones(m)
    
    # Bounds
    bounds = Bounds(lb=np.zeros(m), ub=np.ones(m))
    
    # Solve
    # Time limit for solver: 30s
    # Total time limit: 120s
    # Generation took ~40s.
    # Solver: 30s.
    # Total: 70s.
    
    try:
        res = milp(
            c=c,
            constraints=LinearConstraint(A, -np.inf, b),
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": 30, "mip_rel_gap": 0.0}
        )
        
        if res.status == 0 or res.status == 1: # Optimal or best feasible
            x = res.x
            # Extract edges
            selected_edges = []
            for i in range(m):
                if x[i] > 0.5:
                    selected_edges.append(edges[i])
            return selected_edges
        else:
            # Solver failed or no feasible solution found within time
            # Fall back to greedy
            pass
    except Exception:
        # Fall back to greedy
        pass

    # Fallback: Greedy Removal
    # We have q3_edge_indices (10.8M, 12)
    # Start with all edges active
    active = np.ones(m, dtype=bool)
    
    # Iterate through all q3 subgraphs
    # If all 12 edges are active, remove a random one
    # To make it faster, we can shuffle the subgraphs?
    # Or just iterate.
    
    # Optimization: 
    # Instead of checking all 10.8M every time, we can just iterate once.
    # If we remove an edge, it might invalidate future subgraphs.
    # It won't invalidate past ones (we don't re-check).
    # So a single pass ensures that for every subgraph in the list, 
    # either it was not fully active when we checked it, 
    # or we removed an edge from it.
    # Thus, at the end, no subgraph in the list is fully active.
    # Since the list contains ALL subgraphs, the graph is Q3-free.
    
    # Random seed for reproducibility?
    random.seed(42)
    
    for i in range(total_q3):
        e_indices = q3_edge_indices[i]
        # Check if all are active
        # np.all(active[e_indices])
        # This is 12 checks.
        # 10.8M * 12 = 129M checks.
        # In NumPy, this is fast.
        
        if np.all(active[e_indices]):
            # Remove a random edge
            # Pick index in 0..11
            j = random.randint(0, 11)
            active[e_indices[j]] = False
            
    # Construct result
    selected_edges = []
    for i in range(m):
        if active[i]:
            selected_edges.append(edges[i])
            
    return selected_edges

