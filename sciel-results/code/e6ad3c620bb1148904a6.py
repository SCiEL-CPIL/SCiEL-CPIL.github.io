import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Build edge lookup and identify structure
    # Map (u, v) to index in edges list
    edge_to_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i
        
    # Identify chord indices (non-tree edges)
    # Tree: Matching (i, i+m) for all i, and Top cycle edges (i, i+1) for i=0..m-2
    # Chords: Top cycle closing (m-1, 0) and all Bottom cycle edges (m+i, m+i+1)
    
    chord_indices = []
    tree_edge_set = set()
    
    for i, (u, v) in enumerate(edges):
        # Normalize edge to (min, max) for easy checking
        u_norm, v_norm = (u, v) if u < v else (v, u)
        
        is_matching = abs(u - v) == m
        if is_matching:
            tree_edge_set.add(i)
        else:
            # It's a cycle edge
            # Check if it's in the top cycle
            if u_norm < m:
                # Top cycle edge
                # Check if it's the closing edge (m-1, 0)
                if (u_norm == 0 and v_norm == m-1):
                    chord_indices.append(i)
                else:
                    tree_edge_set.add(i)
            else:
                # Bottom cycle edge
                chord_indices.append(i)
                
    num_chords = len(chord_indices)
    
    # Precompute base adjacency matrix (0/1)
    A_base = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        A_base[u, v] = 1.0
        A_base[v, u] = 1.0
        
    best_radius = float('inf')
    best_signs = None
    
    # Helper function to check if lift is connected (i.e., signed graph has a negative cycle)
    # If all cycles have positive product, the lift is disconnected.
    # This is equivalent to checking if the system of linear equations over GF(2) is consistent:
    # c_u + c_v = 0 if sign(u,v) == +1
    # c_u + c_v = 1 if sign(u,v) == -1
    # If consistent, lift is disconnected. If inconsistent, lift is connected.
    
    def is_lift_connected(signs):
        # Use BFS to assign parities
        parity = [-1] * n
        parity[0] = 0
        queue = [0]
        
        while queue:
            u = queue.pop(0)
            for v in range(n):
                if A_base[u, v] == 1.0:
                    # Get sign of edge (u, v)
                    idx = edge_to_idx[(u, v)]
                    s = signs[idx]
                    # Expected parity of v
                    # If s == 1, c_v == c_u
                    # If s == -1, c_v == 1 - c_u
                    expected_parity = parity[u] if s == 1 else 1 - parity[u]
                    
                    if parity[v] == -1:
                        parity[v] = expected_parity
                        queue.append(v)
                    else:
                        if parity[v] != expected_parity:
                            # Inconsistency found -> Negative cycle exists -> Connected
                            return True
        
        # No inconsistency found -> All cycles positive -> Disconnected
        return False
        
    # Iterate over all chord assignments
    for mask in range(1 << num_chords):
        if time.time() > deadline:
            break
            
        signs = np.ones(num_edges, dtype=np.int8)
        
        for j, idx in enumerate(chord_indices):
            if mask & (1 << j):
                signs[idx] = -1
            else:
                signs[idx] = 1
                
        # Check validity
        if not is_lift_connected(signs):
            continue
            
        # Compute spectral radius
        # Construct signed adjacency matrix
        A_signed = np.zeros((n, n), dtype=np.float64)
        for u, v in edges:
            idx = edge_to_idx[(u, v)]
            val = float(signs[idx])
            A_signed[u, v] = val
            A_signed[v, u] = val
            
        # Spectral radius is max absolute eigenvalue
        # Since A_signed is symmetric, eigenvalues are real
        eigvals = np.linalg.eigvalsh(A_signed)
        radius = np.max(np.abs(eigvals))
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            
    if best_signs is None:
        # Fallback: All-ones is disconnected. We need a valid one.
        # Find any valid configuration quickly.
        # Flip one chord to -1. This creates a negative cycle.
        signs = np.ones(num_edges, dtype=np.int8)
        if num_chords > 0:
            signs[chord_indices[0]] = -1
        else:
            # Should not happen for m>=4
            signs[0] = -1
            
        # Verify and return
        if is_lift_connected(signs):
            return signs
        else:
            # Try flipping another
            for i in range(num_edges):
                trial = signs.copy()
                trial[i] = -trial[i]
                if is_lift_connected(trial):
                    return trial
            return signs # Last resort
            
    return best_signs
