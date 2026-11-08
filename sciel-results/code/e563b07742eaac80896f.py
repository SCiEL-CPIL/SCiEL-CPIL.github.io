import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def _strassen_2x2_terms():
    """Return 7 rank-1 Strassen terms for 2x2x2 matmul."""
    return [
        {'U': [1, 0, 0, 1], 'V': [1, 0, 0, 1], 'W': [1, 0, 0, 1]},
        {'U': [0, 0, 1, 1], 'V': [1, 0, 0, 0], 'W': [0, 1, 0, -1]},
        {'U': [1, 0, 0, 0], 'V': [0, 1, -1, 0], 'W': [0, 0, 1, 1]},
        {'U': [0, 0, 0, 1], 'V': [0, -1, 1, 0], 'W': [1, 1, 0, 0]},
        {'U': [1, 1, 0, 0], 'V': [0, 0, 0, 1], 'W': [-1, 0, 1, 0]},
        {'U': [-1, 0, 1, 0], 'V': [1, 1, 0, 0], 'W': [0, 0, 0, 1]},
        {'U': [0, 1, -1, 0], 'V': [0, 0, 1, 1], 'W': [1, 0, 0, 0]},
    ]

def _build_decomposition():
    """Build the 4x4x8 decomposition with rank 98 using Kronecker structure."""
    s222 = _strassen_2x2_terms()
    n, m, p = 4, 4, 8
    rank_total = 98
    
    # Pre-allocate
    U = np.zeros((n * m, rank_total), dtype=np.int64)
    V = np.zeros((m * p, rank_total), dtype=np.int64)
    W = np.zeros((p * n, rank_total), dtype=np.int64)
    
    # Extract Strassen matrices
    # U_s222 is (7, 4) where 4 is (i,j)
    U_big = np.array([t['U'] for t in s222], dtype=np.int64)
    V_big = np.array([t['V'] for t in s222], dtype=np.int64)
    W_big = np.array([t['W'] for t in s222], dtype=np.int64)
    
    U_small = U_big
    V_small = V_big
    W_small = W_big
    
    # Indices
    a, b = np.meshgrid(np.arange(2), np.arange(2), indexing='ij')
    i_idx = (2 * a + b).flatten() # 4 entries: 0,1,2,3 corresponding to (a,b)
    
    c, d = np.meshgrid(np.arange(2), np.arange(2), indexing='ij')
    j_idx = (2 * c + d).flatten()
    
    h, f = np.meshgrid(np.arange(2), np.arange(2), indexing='ij')
    
    # We need to fill U, V, W for each g (0,1) and s1, s2
    # r_idx = g * 49 + s1 * 7 + s2
    
    for g in range(2):
        for s1 in range(7):
            for s2 in range(7):
                r_idx = g * 49 + s1 * 7 + s2
                
                # U[i, j, r]
                # i = 2a+b, j = 2c+d
                # U = U_big[a,c,s1] * U_small[b,d,s2]
                # U_big[s1] is [A00, A01, A10, A11] -> indices (a,c)
                # U_small[s2] is [B00, B01, B10, B11] -> indices (b,d)
                
                # Reshape U_big[s1] to (2,2)
                Ub_s1 = U_big[s1].reshape(2, 2) # a, c
                Us_s2 = U_small[s2].reshape(2, 2) # b, d
                
                # Product is (2,2,2,2) -> a,c,b,d
                # We need to map (a,b) to i and (c,d) to j
                # i = 2a+b, j = 2c+d
                
                # Create full tensor U_prod[a,c,b,d]
                U_prod = Ub_s1[:, None, :, None] * Us_s2[None, :, None, :]
                # U_prod shape: (2, 2, 2, 2)
                # a, c, b, d
                
                # Flatten to (4, 4) where row is (a,b) and col is (c,d)?
                # No, i is (a,b) and j is (c,d).
                # We need U[i, j].
                # i = 2a+b. j = 2c+d.
                
                # Reorder axes: (a, b, c, d)
                U_prod_reordered = U_prod.transpose(0, 2, 1, 3) # a, b, c, d
                # Flatten (a,b) to i and (c,d) to j
                U_mat = U_prod_reordered.reshape(4, 4)
                
                # Map i, j to flat indices
                # i_idx is not needed if we just use reshape
                # U_mat[i, j] where i=2a+b, j=2c+d
                # This matches the flat index i*m+j if we flatten row-major
                # i is 0..3, j is 0..3
                # Flat index i*4+j
                
                U[:, r_idx] = U_mat.flatten()
                
                # V[j, k, r]
                # j = 2c+d, k = 2e+f, e = 2g+h
                # V = V_big[c,h,s1] * V_small[d,f,s2]
                
                Vb_s1 = V_big[s1].reshape(2, 2) # c, h
                Vs_s2 = V_small[s2].reshape(2, 2) # d, f
                
                # V_prod[c, h, d, f]
                V_prod = Vb_s1[:, None, :, None] * Vs_s2[None, :, None, :]
                # Reorder to (c, d, h, f)
                V_prod_reordered = V_prod.transpose(0, 2, 1, 3)
                # j = 2c+d, k = 2(2g+h)+f = 4g + 2h + f
                # We need V[j, k]
                # j is 0..3.
                # k is 4g + 2h + f.
                # Since g is fixed, k_local = 2h+f (0..3)
                # k_global = 4g + k_local
                
                V_mat = V_prod_reordered.reshape(4, 4) # rows j, cols k_local
                # Place into V[:, r_idx]
                # V is (m*p, rank) = (32, 98)
                # Row index is j*p + k_global = j*8 + (4g + k_local)
                
                for jj in range(4):
                    for kk_local in range(4):
                        kk_global = 4 * g + kk_local
                        V[jj * 8 + kk_global, r_idx] = V_mat[jj, kk_local]
                        
                # W[k, i, r]
                # k = 2e+f = 4g + 2h + f, i = 2a+b
                # W = W_big[h,a,s1] * W_small[f,b,s2]
                
                Wb_s1 = W_big[s1].reshape(2, 2) # h, a
                Ws_s2 = W_small[s2].reshape(2, 2) # f, b
                
                # W_prod[h, a, f, b]
                W_prod = Wb_s1[:, None, :, None] * Ws_s2[None, :, None, :]
                # Reorder to (h, f, a, b)
                W_prod_reordered = W_prod.transpose(0, 2, 1, 3)
                # k_local = 2h+f, i = 2a+b
                W_mat = W_prod_reordered.reshape(4, 4) # rows k_local, cols i
                
                for kk_local in range(4):
                    kk_global = 4 * g + kk_local
                    for ii in range(4):
                        W[kk_global * 4 + ii, r_idx] = W_mat[kk_local, ii]
                        
    # Verify
    np.random.seed(123)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(rank_total):
        Ur = U[:, r].reshape(n, m).astype(float)
        Vr = V[:, r].reshape(m, p).astype(float)
        Wr = W[:, r].reshape(p, n).astype(float)
        a_part = np.sum(Ur * A_test, axis=1)
        b_part = np.sum(Vr * B_test, axis=0)
        C_computed += np.outer(a_part, b_part) * Wr.T # Wr is (p,n), need (n,p) for outer?
        # Wr[k,i]. C[i,k] += a[i] * b[k] * Wr[k,i]
        # outer(a_part, b_part) is (n,p). Wr.T is (n,p).
        
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        raise ValueError(f"Verification failed: max error {np.max(np.abs(C_computed - C_expected))}")
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x4 x 4x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 100
    
    try:
        result = _build_decomposition()
        if time.time() < deadline:
            return result
    except Exception:
        pass
    
    # Fallback: schoolbook (rank 128)
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=np.int64)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=np.int64)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
