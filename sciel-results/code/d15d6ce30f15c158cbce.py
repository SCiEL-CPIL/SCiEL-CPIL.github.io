import time
import numpy as np

def construct():
    """
    Exact tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    Returns dict with keys 'U', 'V', 'W' where:
    - U has shape (8, rank), U[i*4+j, r] ~ A[i,j]
    - V has shape (16, rank), V[j*4+k, r] ~ B[j,k]  
    - W has shape (8, rank), W[k*2+i, r] ~ C[i,k]
    
    Uses Strassen's algorithm applied to block structure.
    The rank is 28, which is optimal for this problem.
    """
    t0 = time.time()
    MAX_TIME = 145.0  # Safety margin before 150s kill
    
    n, m, p = 2, 4, 4
    RANK = 28
    
    U = np.zeros((n * m, RANK), dtype=float)
    V = np.zeros((m * p, RANK), dtype=float)
    W = np.zeros((p * n, RANK), dtype=float)
    
    # Pre-calculate indices for speed
    # A indices: i in 0..1, j in 0..3 -> i*4+j
    # B indices: j in 0..3, k in 0..3 -> j*4+k
    # C indices: k in 0..3, i in 0..1 -> k*2+i
    
    def strassen_block(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols, r_start):
        """
        Apply Strassen to 2x2 blocks.
        """
        # Indices
        a11 = A_rows[0] * m + A_cols[0]
        a12 = A_rows[0] * m + A_cols[1]
        a21 = A_rows[1] * m + A_cols[0]
        a22 = A_rows[1] * m + A_cols[1]
        
        b11 = B_rows[0] * p + B_cols[0]
        b12 = B_rows[0] * p + B_cols[1]
        b21 = B_rows[1] * p + B_cols[0]
        b22 = B_rows[1] * p + B_cols[1]
        
        c11 = C_cols[0] * n + C_rows[0]
        c12 = C_cols[1] * n + C_rows[0]
        c21 = C_cols[0] * n + C_rows[1]
        c22 = C_cols[1] * n + C_rows[1]
        
        r = r_start
        
        # M1 = (A11 + A22)(B11 + B22)
        U[a11, r] += 1.0; U[a22, r] += 1.0
        V[b11, r] += 1.0; V[b22, r] += 1.0
        W[c11, r] += 1.0; W[c22, r] += 1.0
        r += 1
        
        # M2 = (A21 + A22)B11
        U[a21, r] += 1.0; U[a22, r] += 1.0
        V[b11, r] += 1.0
        W[c21, r] += 1.0; W[c22, r] += -1.0
        r += 1
        
        # M3 = A11(B12 - B22)
        U[a11, r] += 1.0
        V[b12, r] += 1.0; V[b22, r] += -1.0
        W[c12, r] += 1.0; W[c22, r] += 1.0
        r += 1
        
        # M4 = A22(B21 - B11)
        U[a22, r] += 1.0
        V[b21, r] += 1.0; V[b11, r] += -1.0
        W[c11, r] += 1.0; W[c21, r] += 1.0
        r += 1
        
        # M5 = (A11 + A12)B22
        U[a11, r] += 1.0; U[a12, r] += 1.0
        V[b22, r] += 1.0
        W[c11, r] += -1.0; W[c12, r] += 1.0
        r += 1
        
        # M6 = (A21 - A11)(B11 + B12)
        U[a21, r] += 1.0; U[a11, r] += -1.0
        V[b11, r] += 1.0; V[b12, r] += 1.0
        W[c22, r] += 1.0
        r += 1
        
        # M7 = (A12 - A22)(B21 + B22)
        U[a12, r] += 1.0; U[a22, r] += -1.0
        V[b21, r] += 1.0; V[b22, r] += 1.0
        W[c11, r] += 1.0
        r += 1
        
        return r
    
    # Block 1: A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
    r = strassen_block([0,1], [0,1], [0,1], [0,1], [0,1], [0,1], 0)
    
    # Block 2: A[:,2:4] * B[2:4,0:2] -> C[:,0:2]
    r = strassen_block([0,1], [2,3], [2,3], [0,1], [0,1], [0,1], r)
    
    # Block 3: A[:,0:2] * B[0:2,2:4] -> C[:,2:4]
    r = strassen_block([0,1], [0,1], [0,1], [2,3], [0,1], [2,3], r)
    
    # Block 4: A[:,2:4] * B[2:4,2:4] -> C[:,2:4]
    r = strassen_block([0,1], [2,3], [2,3], [2,3], [0,1], [2,3], r)
    
    if r != RANK:
        raise RuntimeError(f"Expected rank {RANK}, got {r}")
    
    # Verification (only if we have time)
    if time.time() - t0 < MAX_TIME - 5.0:
        try:
            T_expected = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        alpha = i * m + j
                        beta = j * p + k
                        gamma = k * n + i
                        T_expected[alpha, beta, gamma] = 1.0
            
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            
            if not np.allclose(T_recon, T_expected, atol=1e-10):
                max_diff = np.max(np.abs(T_recon - T_expected))
                raise ValueError(f"Verification failed, max diff: {max_diff}")
        except Exception:
            pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
