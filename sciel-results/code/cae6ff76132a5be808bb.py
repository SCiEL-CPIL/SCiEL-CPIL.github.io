<<<<<<< SEARCH
    rank = len(terms)
    
    if rank == 0:
        add_simple_term(0, 0, 0, 1.0)
        rank = 1
        
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r, (u_sp, v_sp, w_sp) in enumerate(terms):
        for idx, val in u_sp:
            U[idx, r] = val
        for idx, val in v_sp:
            V[idx, r] = val
        for idx, val in w_sp:
            W[idx, r] = val
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
=======
    rank = len(terms)
    
    if rank == 0:
        add_simple_term(0, 0, 0, 1.0)
        rank = 1
        
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r, (u_sp, v_sp, w_sp) in enumerate(terms):
        for idx, val in u_sp:
            U[idx, r] = val
        for idx, val in v_sp:
            V[idx, r] = val
        for idx, val in w_sp:
            W[idx, r] = val
    
    # Try to reduce rank by finding linear dependencies
    # The tensor decomposition is valid if sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T
    # If columns of U, V, W are linearly dependent, we can reduce rank
    
    # Check if the factor matrices have linearly dependent columns
    # Use SVD to find the numerical rank of the combined system
    
    # First, check if we can find a transformation that reduces rank
    # The key insight: if U, V, W share a common null space structure,
    # we can project to a lower-dimensional space
    
    # Try QR decomposition with column pivoting on U
    # to identify independent columns
    try:
        # Stack the three matrices to check for joint dependencies
        # Actually, we need to check if there's a non-trivial linear relation
        # among the rank-1 tensors
        
        # A simpler approach: check if any columns are linearly dependent
        # by computing the rank of U, V, W separately
        
        U_rank = np.linalg.matrix_rank(U, tol=1e-10)
        V_rank = np.linalg.matrix_rank(V, tol=1e-10)
        W_rank = np.linalg.matrix_rank(W, tol=1e-10)
        
        # If any matrix has rank < number of columns, we can potentially reduce
        # But we need to maintain the tensor identity
        
        # Try to find the minimal rank using a greedy approach
        # Start with full rank, try to remove terms one by one
        
        best_U, best_V, best_W = U, V, W
        best_rank = rank
        
        # Try removing each term and checking if the tensor is still preserved
        # This is expensive, so only try if rank is small enough
        
        if rank <= 100:
            # Compute the target tensor T
            T = np.zeros((n*m, m*p, p*n))
            for r in range(rank):
                T += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
            
            # Try removing terms
            removed = set()
            for r in range(rank):
                if time.time() - _START > _DEADLINE - 15:
                    break
                if r in removed:
                    continue
                
                # Try removing term r
                new_rank = len([x for x in range(rank) if x not in removed]) - 1
                if new_rank >= best_rank:
                    continue
                
                # Check if tensor is preserved without term r
                T_without_r = T - np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
                
                # Reconstruct from remaining terms
                T_reconstructed = np.zeros((n*m, m*p, p*n))
                for s in range(rank):
                    if s == r or s in removed:
                        continue
                    T_reconstructed += np.einsum('i,j,k->ijk', U[:,s], V[:,s], W[:,s])
                
                if np.allclose(T_without_r, T_reconstructed, atol=1e-10):
                    removed.add(r)
                    best_rank = len([x for x in range(rank) if x not in removed])
            
            if best_rank < rank:
                # Reconstruct with only non-removed terms
                keep_indices = [r for r in range(rank) if r not in removed]
                U = U[:, keep_indices]
                V = V[:, keep_indices]
                W = W[:, keep_indices]
                rank = best_rank
    except Exception:
        # If anything goes wrong, keep the original
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
>>>>>>> REPLACE
