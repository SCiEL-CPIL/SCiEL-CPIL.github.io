import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p  # 42

    # Build schoolbook decomposition as fallback
    U_school = np.zeros((nm, schoolbook_rank))
    V_school = np.zeros((mp, schoolbook_rank))
    W_school = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    def to_half_int_list(matrix):
        """Convert numpy matrix to list of lists with half-integer entries"""
        return [[float(matrix[i, j]) for j in range(matrix.shape[1])] for i in range(matrix.shape[0])]

    def verify_exact(U, V, W, rank, n, m, p):
        """Verify exact decomposition using integer arithmetic (multiply by 2 to clear halves)."""
        # Multiply by 2 to work in integers
        U2 = np.round(U * 2).astype(np.int64)
        V2 = np.round(V * 2).astype(np.int64)
        W2 = np.round(W * 2).astype(np.int64)
        
        # Check: for all i, j, k: sum_r U2[i*m+j, r] * V2[j*p+k, r] * W2[k*n+i, r] should be 8
        # (because each original factor is 1 in schoolbook, and 1*1*1=1, but we scaled by 2 so 2*2*2=8)
        # Actually, the condition is: for the matmul tensor, the sum should equal 1 for valid (i,j,k)
        # Let me re-derive: The matmul tensor T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
        # Wait no. The matmul tensor for (n x m) * (m x p) -> (n x p):
        # C[i,k] = sum_j A[i,j] B[j,k]
        # The rank-r decomposition says: C[i,k] = sum_r U[i,j1,r] V[j2,k,r] W[k,i,r]... 
        # 
        # Let me use the empirical check: for random A, B, compute C via decomposition and compare.
        rng = np.random.RandomState(12345)
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_true = A @ B
        
        C_comp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                total = 0.0
                for j in range(m):
                    for r in range(rank):
                        total += A[i, j] * B[j, k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                C_comp[i, k] = total
        
        if not np.allclose(C_comp, C_true, atol=1e-9):
            return False
        return True

    def peel_rank1(T, n, m, p):
        """
        Peel off a rank-1 component from tensor T of shape (nm, mp, pn).
        Uses HOSVD-like approach: find best rank-1 approximation.
        Returns (u, v, w, alpha, T_residual) where T ≈ alpha * u ⊗ v ⊗ w + T_residual
        """
        # Reshape T to (nm, mp, pn)
        nm, mp, pn = n * m, m * p, p * n
        T3 = T.reshape(nm, mp, pn)
        
        # Compute HOSVD
        # Mode-1 unfolding: (nm, mp*pn)
        T1 = T3.reshape(nm, mp * pn)
        U1, S1, Vt1 = np.linalg.svd(T1, full_matrices=False)
        
        # Mode-2 unfolding: (mp, nm*pn)
        T2 = T3.transpose(1, 0, 2).reshape(mp, nm * pn)
        U2, S2, Vt2 = np.linalg.svd(T2, full_matrices=False)
        
        # Mode-3 unfolding: (pn, nm*mp)
        T3u = T3.transpose(2, 0, 1).reshape(pn, nm * mp)
        U3, S3, Vt3 = np.linalg.svd(T3u, full_matrices=False)
        
        # Take first left singular vectors
        u = U1[:, 0]
        v = U2[:, 0]
        w = U3[:, 0]
        
        # Compute alpha = trilinear form
        alpha = np.einsum('i,j,k,ijk->', u, v, w, T3)
        
        # Compute residual
        T_residual = T3 - alpha * np.einsum('i,j,k->ijk', u, v, w)
        
        return u, v, w, alpha, T_residual.reshape(nm * mp * pn)

    # Greedy peeling: start from full tensor, peel rank-1 components
    # The matmul tensor in (nm, mp, pn) space has T[i*m+j, j*p+k, k*n+i] = 1 for valid indices
    # Actually, let me construct the tensor properly.
    # The matmul tensor: for each (i, j, k), the entry at position (i*m+j, j*p+k, k*n+i) is 1
    # But we need a dense tensor of shape (nm, mp, pn)
    
    T_dense = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_dense[i * m + j, j * p + k, k * n + i] = 1.0
    
    T_flat = T_dense.reshape(nm * mp * pn)
    
    best_rank = None
    best_factors = None
    
    # Try greedy peeling from rank 42 down to 20
    for target_rank in range(42, 19, -1):
        T_remaining = T_flat.copy()
        U_cols = []
        V_cols = []
        W_cols = []
        
        for r in range(target_rank):
            u, v, w, alpha, T_remaining = peel_rank1(T_remaining, n, m, p)
            U_cols.append(alpha * u)
            V_cols.append(v)
            W_cols.append(w)
        
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        
        # Project to nearest half-integer
        U_half = np.round(U * 2) / 2.0
        V_half = np.round(V * 2) / 2.0
        W_half = np.round(W * 2) / 2.0
        
        if verify_exact(U_half, V_half, W_half, target_rank, n, m, p):
            best_rank = target_rank
            best_factors = (U_half, V_half, W_half)
            break
    
    if best_factors is not None:
        U, V, W = best_factors
        return {"U": to_half_int_list(U), "V": to_half_int_list(V), "W": to_half_int_list(W)}
    
    # Fallback to schoolbook
    return {"U": to_half_int_list(U_school), "V": to_half_int_list(V_school), "W": to_half_int_list(W_school)}
