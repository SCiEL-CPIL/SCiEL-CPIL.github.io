"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy: 
    1. Build rank-77 via Kronecker: Strassen(2x2x2) ⊗ hybrid(2x2x3)
    2. Attempt rank reduction using linear algebra on the coefficient tensor
    3. Return best valid decomposition found within time budget
    """
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 110.0
    
    # ---- Strassen decomposition for 2x2x2 ----
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # U rows: i*m+j, V rows: j*p+k, W rows: k*n+i
    U1 = np.zeros((4, 7), dtype=int)
    V1 = np.zeros((4, 7), dtype=int)
    W1 = np.zeros((4, 7), dtype=int)
    
    # M1 = (a00+a11)(b00+b11)
    U1[0,0] = 1; U1[3,0] = 1
    V1[0,0] = 1; V1[3,0] = 1
    W1[0,0] = 1; W1[3,0] = 1
    
    # M2 = (a10+a11)b00
    U1[2,1] = 1; U1[3,1] = 1
    V1[0,1] = 1
    W1[1,1] = 1; W1[3,1] = -1
    
    # M3 = a00(b01-b11)
    U1[0,2] = 1
    V1[1,2] = 1; V1[3,2] = -1
    W1[2,2] = 1; W1[3,2] = 1
    
    # M4 = a11(b10-b00)
    U1[3,3] = 1
    V1[2,3] = 1; V1[0,3] = -1
    W1[0,3] = 1; W1[1,3] = 1
    
    # M5 = (a00+a01)b11
    U1[0,4] = 1; U1[1,4] = 1
    V1[3,4] = 1
    W1[0,4] = -1; W1[2,4] = 1
    
    # M6 = (a10-a00)(b00+b01)
    U1[2,5] = 1; U1[0,5] = -1
    V1[0,5] = 1; V1[1,5] = 1
    W1[3,5] = 1
    
    # M7 = (a01-a11)(b10+b11)
    U1[1,6] = 1; U1[3,6] = -1
    V1[2,6] = 1; V1[3,6] = 1
    W1[0,6] = 1
    
    # ---- 2x2x3 decomposition (rank 11) ----
    # A is 2x2, B is 2x3, C is 2x3
    # U2 rows: i*2+j (4 rows), V2 rows: j*3+k (6 rows), W2 rows: k*2+i (6 rows)
    U2 = np.zeros((4, 11), dtype=int)
    V2 = np.zeros((6, 11), dtype=int)
    W2 = np.zeros((6, 11), dtype=int)
    
    # Strassen part for columns 0-1 (terms 0-6)
    for c in range(7):
        U2[0,c] = U1[0,c]
        U2[1,c] = U1[1,c]
        U2[2,c] = U1[2,c]
        U2[3,c] = U1[3,c]
        
        # V1 rows j*2+k -> V2 rows j*3+k (skip column 2 of B)
        V2[0,c] = V1[0,c]  # j=0,k=0
        V2[1,c] = V1[1,c]  # j=0,k=1
        V2[3,c] = V1[2,c]  # j=1,k=0
        V2[4,c] = V1[3,c]  # j=1,k=1
        V2[2,c] = 0        # j=0,k=2
        V2[5,c] = 0        # j=1,k=2
        
        # W1 rows k*2+i -> W2 rows k*2+i (only k=0,1)
        W2[0,c] = W1[0,c]
        W2[1,c] = W1[1,c]
        W2[2,c] = W1[2,c]
        W2[3,c] = W1[3,c]
        W2[4,c] = 0
        W2[5,c] = 0
    
    # Trivial part for column 2 (terms 7-10)
    U2[0,7] = 1; V2[2,7] = 1; W2[4,7] = 1
    U2[1,8] = 1; V2[5,8] = 1; W2[4,8] = 1
    U2[2,9] = 1; V2[2,9] = 1; W2[5,9] = 1
    U2[3,10] = 1; V2[5,10] = 1; W2[5,10] = 1
    
    # ---- Kronecker product to get full 4x4x6 decomposition ----
    rank_total = 7 * 11  # 77
    U = np.zeros((16, rank_total), dtype=int)
    V = np.zeros((24, rank_total), dtype=int)
    W = np.zeros((24, rank_total), dtype=int)
    
    # A[i,j]: i=i1*2+i2, j=j1*2+j2 -> U row = i*4+j
    # B[j,k]: j=j1*2+j2, k=k1*3+k2 -> V row = j*6+k
    # C[i,k]: i=i1*2+i2, k=k1*3+k2 -> W row = k*4+i
    
    for c1 in range(7):
        for c2 in range(11):
            c = c1 * 11 + c2
            
            # U: i1*2+i2, j1*2+j2 -> row i1*8+i2*4+j1*2+j2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u1 = U1[i1*2+j1, c1]
                            u2 = U2[i2*2+j2, c2]
                            if u1 != 0 and u2 != 0:
                                U[i1*8 + i2*4 + j1*2 + j2, c] = u1 * u2
            
            # V: j1*2+j2, k1*3+k2 -> row j1*12+j2*6+k1*3+k2
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            v1 = V1[j1*2+k1, c1]
                            v2 = V2[j2*3+k2, c2]
                            if v1 != 0 and v2 != 0:
                                V[j1*12 + j2*6 + k1*3 + k2, c] = v1 * v2
            
            # W: k1*3+k2, i1*2+i2 -> row k1*12+k2*4+i1*2+i2
            for k1 in range(2):
                for k2 in range(3):
                    for i1 in range(2):
                        for i2 in range(2):
                            w1 = W1[k1*2+i1, c1]
                            w2 = W2[k2*2+i2, c2]
                            if w1 != 0 and w2 != 0:
                                W[k1*12 + k2*4 + i1*2 + i2, c] = w1 * w2
    
    def _verify(U_mat, V_mat, W_mat, n, m, p, seed=42):
        np.random.seed(seed)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for r in range(U_mat.shape[1]):
            alpha_r = np.sum(A_test * U_mat[:, r].reshape(n, m))
            beta_r = np.sum(B_test * V_mat[:, r].reshape(m, p))
            C_computed += alpha_r * beta_r * W_mat[:, r].reshape(p, n).T
        
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_s = np.zeros((n * m, rank_school), dtype=int)
        V_s = np.zeros((m * p, rank_school), dtype=int)
        W_s = np.zeros((p * n, rank_school), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, col] = 1
                    V_s[j * p + k, col] = 1
                    W_s[k * n + i, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # ---- Attempt rank reduction via linear algebra ----
    if time.time() < deadline:
        # Build the coefficient tensor as a 3D array T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        # We want to find if some columns are linearly dependent in a way that allows removal.
        
        # Approach: treat each column r as a tensor slice. If the set of columns 
        # is linearly dependent when viewed as vectors in the space of (U_col, V_col, W_col)
        # triplets that produce zero contribution, we can remove one.
        
        # More practical: use the fact that the multiplication tensor T has a known structure.
        # Try a greedy approach with SVD-based rank estimation.
        
        U_float = U.astype(float)
        V_float = V.astype(float)
        W_float = W.astype(float)
        
        # Build matrix M where each row corresponds to a (i,j,k) entry and each column to r
        # M[i,j,k, r] = U[i,j,r] * V[j,k,r] * W[k,i,r]
        # This is 4*4*6 = 96 rows, 77 columns
        num_entries = n * m * p  # 96
        M = np.zeros((num_entries, rank_total))
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M[idx, :] = U_float[i*m+j, :] * V_float[j*p+k, :] * W_float[k*n+i, :]
                    idx += 1
        
        # The multiplication tensor should have M * x = t_target for any valid decomposition
        # But we want to find if we can reduce rank by finding linear dependencies.
        
        # Compute SVD to estimate numerical rank
        try:
            s = np.linalg.svd(M, compute_uv=False)
            # If there are zeros in singular values, those dimensions can be removed
            tol = 1e-10
            num_nonzero = np.sum(s > tol)
            if num_nonzero < rank_total:
                # We might be able to reduce, but SVD of M doesn't directly give us the reduction
                pass
        except np.linalg.LinAlgError:
            pass
        
        # Greedy column removal with verification
        num_cols = U_float.shape[1]
        keep_mask = np.ones(num_cols, dtype=bool)
        
        for r in range(num_cols):
            if time.time() > deadline - 15:
                break
            if not keep_mask[r]:
                continue
            
            keep_mask[r] = False
            mask = keep_mask.copy()
            
            U_trial = U_float[:, mask]
            V_trial = V_float[:, mask]
            W_trial = W_float[:, mask]
            
            if _verify(U_trial, V_trial, W_trial, n, m, p):
                U_float = U_trial
                V_float = V_trial
                W_float = W_trial
            else:
                keep_mask[r] = True
        
        # Final verification
        if _verify(U_float, V_float, W_float, n, m, p):
            U = U_float
            V = V_float
            W = W_float
        
        # Convert back to int if possible
        if np.allclose(U, np.round(U), atol=1e-10):
            U = np.round(U).astype(int)
        if np.allclose(V, np.round(V), atol=1e-10):
            V = np.round(V).astype(int)
        if np.allclose(W, np.round(W), atol=1e-10):
            W = np.round(W).astype(int)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
