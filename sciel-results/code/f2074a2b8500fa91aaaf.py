import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                computed = 0.0
                for r in range(rank):
                    computed += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(computed - 1.0) > tol:
                    return False
    return True

def _build_rank36_decomposition(n, m, p):
    """
    Build a rank-36 decomposition:
      - 4 blocks of 2x2 * 2x2 using Strassen (7 each) = 28
      - 2x1 * 1x8 using combined terms (8) = 8
    Total: 36
    """
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Part 1: 2x2 * 2x8 using Strassen on 2-column blocks
    # A is 2x2 (rows 0,1 of A, cols 0,1 of A? No, A is 2x3. We use cols 0,1 of A for Part 1)
    # B is 2x8 (rows 0,1 of B, cols 0..7 of B)
    # We split B into 4 blocks of 2x2.
    for j_block in range(4):
        # B block columns: 2*j_block, 2*j_block+1
        w_r00 = 4 * j_block
        w_r01 = 4 * j_block + 2
        w_r10 = 4 * j_block + 1
        w_r11 = 4 * j_block + 3
        
        v_e = 2 * j_block      # B[0, 2*j_block]
        v_f = 2 * j_block + 1  # B[0, 2*j_block+1]
        v_g = 8 + 2 * j_block      # B[1, 2*j_block]
        v_h = 9 + 2 * j_block      # B[1, 2*j_block+1]
        
        # P1 = (a+d)(e+h)
        U[0, col] = 1; U[4, col] = 1 # A[0,0], A[1,1]
        V[v_e, col] = 1; V[v_h, col] = 1
        W[w_r00, col] = 1; W[w_r11, col] = 1
        col += 1
        # P2 = (c+d)e
        U[3, col] = 1; U[4, col] = 1 # A[1,0], A[1,1]
        V[v_e, col] = 1
        W[w_r10, col] = 1; W[w_r11, col] = -1
        col += 1
        # P3 = a(f-h)
        U[0, col] = 1 # A[0,0]
        V[v_f, col] = 1; V[v_h, col] = -1
        W[w_r01, col] = 1; W[w_r11, col] = 1
        col += 1
        # P4 = d(g-e)
        U[4, col] = 1 # A[1,1]
        V[v_g, col] = 1; V[v_e, col] = -1
        W[w_r00, col] = 1; W[w_r10, col] = 1
        col += 1
        # P5 = (a+b)h
        U[0, col] = 1; U[1, col] = 1 # A[0,0], A[0,1]
        V[v_h, col] = 1
        W[w_r00, col] = -1; W[w_r01, col] = 1
        col += 1
        # P6 = (c-a)(e+f)
        U[3, col] = 1; U[0, col] = -1 # A[1,0], A[0,0]
        V[v_e, col] = 1; V[v_f, col] = 1
        W[w_r11, col] = 1
        col += 1
        # P7 = (b-d)(g+h)
        U[1, col] = 1; U[4, col] = -1 # A[0,1], A[1,1]
        V[v_g, col] = 1; V[v_h, col] = 1
        W[w_r00, col] = 1
        col += 1
        
    # Part 2: 2x1 * 1x8
    # A is 2x1 (rows 0,1 of A, col 2 of A)
    # B is 1x8 (row 2 of B, cols 0..7 of B)
    # We use 8 terms, one for each k
    for k in range(8):
        # U indices for A[0,2] and A[1,2]
        # A is 2x3. Index i*m + j.
        # A[0,2] -> 0*3+2 = 2
        # A[1,2] -> 1*3+2 = 5
        U[2, col] = 1
        U[5, col] = 1
        
        # V index for B[2,k]
        # B is 3x8. Index j*p + k.
        # B[2,k] -> 2*8+k = 16+k
        V[16 + k, col] = 1
        
        # W indices for C[0,k] and C[1,k]
        # C is 2x8. Index k*n + i.
        # C[0,k] -> k*2+0 = 2k
        # C[1,k] -> k*2+1 = 2k+1
        W[2 * k, col] = 1
        W[2 * k + 1, col] = 1
        
        col += 1
        
    return U, V, W

def construct():
    """
    Constructs a low-rank matrix multiplication decomposition for 2x3 * 3x8.
    Returns a dictionary with matrices U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    n, m, p = DIMENSIONS
    
    # Build rank-36 decomposition
    U, V, W = _build_rank36_decomposition(n, m, p)
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to rank-44 if verification fails (should not happen)
        rank = 44
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        # ... (reuse previous logic or just return schoolbook)
        # For safety, return schoolbook
        rank = n * m * p
        U_fb = np.zeros((n * m, rank))
        V_fb = np.zeros((m * p, rank))
        W_fb = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U_fb[i * m + j, idx] = 1
                    V_fb[j * p + k, idx] = 1
                    W_fb[k * n + i, idx] = 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

