<<<<<<< SEARCH
def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of multiplying a 3x5 matrix by a 5x6 matrix.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    # First, try to find the best block partition
    best_rank, best_partition = _find_best_block_partition(n, m, p, deadline)
    
    if best_partition is not None:
        n_sizes, m_sizes, p_sizes = best_partition
    else:
        # Default partition
        n_sizes, m_sizes, p_sizes = [1, 2], [2, 2, 1], [2, 2, 2]
        best_rank = 84

    # Build the decomposition
    U, V, W, rank = build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes)
    
    # Verify
    if verify_decomposition(U, V, W, n, m, p):
        # Try to reduce rank if we have time
        if time.time() < deadline - 15:
            U_reduced, V_reduced, W_reduced = try_reduce_rank(U, V, W, n, m, p)
            if U_reduced.shape[1] < U.shape[1]:
                if verify_decomposition(U_reduced, V_reduced, W_reduced, n, m, p):
                    return {"U": U_reduced.tolist(), "V": V_reduced.tolist(), "W": W_reduced.tolist()}
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: Schoolbook
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb), dtype=np.float64)
    V_fb = np.zeros((m * p, rank_fb), dtype=np.float64)
    W_fb = np.zeros((p * n, rank_fb), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
=======
def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of multiplying a 3x5 matrix by a 5x6 matrix.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    # First, try to find the best block partition
    best_rank, best_partition = _find_best_block_partition(n, m, p, deadline)
    
    if best_partition is not None:
        n_sizes, m_sizes, p_sizes = best_partition
    else:
        # Default partition
        n_sizes, m_sizes, p_sizes = [2, 1], [2, 2, 1], [2, 2, 2]
        best_rank = 84

    # Build the decomposition
    U, V, W, rank = build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes)
    
    # Verify
    if verify_decomposition(U, V, W, n, m, p):
        # Try to reduce rank if we have time
        if time.time() < deadline - 30:
            U_reduced, V_reduced, W_reduced = try_reduce_rank(U, V, W, n, m, p, deadline=deadline)
            if U_reduced.shape[1] < U.shape[1]:
                if verify_decomposition(U_reduced, V_reduced, W_reduced, n, m, p):
                    return {"U": U_reduced.tolist(), "V": V_reduced.tolist(), "W": W_reduced.tolist()}
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: Schoolbook
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb), dtype=np.float64)
    V_fb = np.zeros((m * p, rank_fb), dtype=np.float64)
    W_fb = np.zeros((p * n, rank_fb), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
>>>>>>> REPLACE
