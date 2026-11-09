import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n = len(unvisited_nodes)
    if n == 0:
        return -1
    if n == 1:
        return int(unvisited_nodes[0])
    
    U = unvisited_nodes
    d_current = distance_matrix[current_node, U]
    D_U = distance_matrix[U, U]
    d_dest = distance_matrix[U, destination_node]
    
    if n == 2:
        j0, j1 = U[0], U[1]
        cost0 = d_current[0] + D_U[0, 1] + d_dest[1]
        cost1 = d_current[1] + D_U[1, 0] + d_dest[0]
        return int(j0 if cost0 < cost1 else j1)
        
    # M[k, m] = D_U[k, m] + d_dest[m]
    M = D_U + d_dest[np.newaxis, :]
    
    # Precompute min1, min2 values and indices for each row of M
    # min1_val[k], min1_idx[k]
    # min2_val[k], min2_idx[k]
    
    min1_val = np.full(n, np.inf)
    min1_idx = np.full(n, -1, dtype=int)
    min2_val = np.full(n, np.inf)
    min2_idx = np.full(n, -1, dtype=int)
    
    for k in range(n):
        row = M[k]
        # Find min and second min
        # np.argpartition is good for this
        # indices = np.argpartition(row, 2)
        # i1, i2 = indices[0], indices[1]
        # This is O(n)
        
        # Simpler for small n:
        # sort? O(n log n)
        # Just do two passes? O(n)
        
        min1_val[k] = np.min(row)
        min1_idx[k] = np.argmin(row)
        
        # Mask min1_idx and find min again
        row_copy = row.copy()
        row_copy[min1_idx[k]] = np.inf
        min2_val[k] = np.min(row_copy)
        min2_idx[k] = np.argmin(row_copy)
        
    scores = np.full(n, np.inf)
    
    for j in range(n):
        min_k_cost = np.inf
        for k in range(n):
            if k == j:
                continue
            
            # We need min of M[k, :] excluding j and k
            # Check min1
            if min1_idx[k] != j and min1_idx[k] != k:
                m_cost = min1_val[k]
            elif min2_idx[k] != j and min2_idx[k] != k:
                m_cost = min2_val[k]
            else:
                # Both min1 and min2 are excluded. Need 3rd min.
                # This only happens if n >= 3.
                # For n=3, if min1, min2 are j, k, the 3rd is the only one left.
                # For n>3, we need to find the 3rd min.
                # Since this is rare (only if j and k are the two smallest in row k),
                # we can just do the copy/min here.
                row_copy = M[k].copy()
                row_copy[j] = np.inf
                row_copy[k] = np.inf
                m_cost = np.min(row_copy)
                
            total_k_cost = D_U[j, k] + m_cost
            if total_k_cost < min_k_cost:
                min_k_cost = total_k_cost
                
        scores[j] = d_current[j] + min_k_cost
        
    best_idx = np.argmin(scores)
    return int(U[best_idx])
