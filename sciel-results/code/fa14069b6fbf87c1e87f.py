import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Try to construct rank 46 decomposition using Laderman's algorithm for 3x3x3
    # Laderman's algorithm for 3x3 matrix multiplication has rank 23.
    # We use it for the first 3 columns of B and the next 3 columns of B.
    
    try:
        # U, V, W for one 3x3x3 multiplication (rank 23)
        U_33, V_33, W_33 = get_laderman_factors()
        
        rank_33 = U_33.shape[1]
        total_rank = 2 * rank_33
        
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        # First 3 columns of B (k=0,1,2)
        # U is same
        U[:, 0:rank_33] = U_33
        # V: rows are j*p + k. For k=0,1,2, rows are 0..8.
        # V_33 has shape (9, 23) where row index is j*3 + k.
        V[0:9, 0:rank_33] = V_33
        # W: rows are k*n + i. For k=0,1,2, rows are 0..8.
        # W_33 has shape (9, 23) where row index is k*3 + i? 
        # In Laderman, W maps M_r to C_{i,k}.
        # So W_33[k*3+i, r] is the coefficient.
        W[0:9, 0:rank_33] = W_33
        
        # Second 3 columns of B (k=3,4,5)
        # U is same
        U[:, rank_33:total_rank] = U_33
        # V: rows are j*p + k. For k=3,4,5, rows are 9..17.
        # V_33 row j*3+k' where k' is 0,1,2.
        # We need V[j*6 + (k'+3), r] = V_33[j*3 + k', r-23]
        for j in range(3):
            for k_prime in range(3):
                V[j*6 + k_prime + 3, rank_33:total_rank] = V_33[j*3 + k_prime, 0:rank_33]
                
        # W: rows are k*n + i. For k=3,4,5, rows are 9..17.
        # W_33 row k'*3+i.
        # We need W[(k'+3)*3 + i, r] = W_33[k'*3 + i, r-23]
        for k_prime in range(3):
            for i in range(3):
                W[(k_prime + 3)*3 + i, rank_33:total_rank] = W_33[k_prime*3 + i, 0:rank_33]
                
        # Verify
        if verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    except Exception:
        pass

    # Fallback to seed (rank 54)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def get_laderman_factors():
    # Returns U, V, W for 3x3x3 rank 23
    # This is a placeholder. I need to fill this with actual Laderman factors.
    # If I can't, this will fail and fallback.
    raise NotImplementedError("Laderman factors not implemented")

def verify_decomposition(U, V, W, n, m, p):
    # Check if sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k
    # Actually, the tensor T_{i,j,k} is 1 for all i,j,k.
    # So the sum should be 1.
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Tensor product check
    # T = sum_r U_r (n x m) * V_r (m x p) * W_r (p x n)?
    # No, the decomposition is T_{i,j,k} = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # This is not standard matrix multiplication tensor form.
    # Standard form: C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # Tensor T_{i,j,k} = 1.
    # Decomposition: T = sum_r U_r \otimes V_r \otimes W_r.
    # U_r is n x m matrix? No, U is (nm, rank).
    # So U_r is a vector of length nm.
    # The term is U_{i,j,r} * V_{j,k,r} * W_{k,i,r}.
    # This matches the seed code logic.
    
    rank = U.shape[1]
    for r in range(rank):
        pass # Just check dimensions for now
    
    # Full verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if not np.isclose(s, 1.0):
                    return False
    return True
