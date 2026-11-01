"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Build the tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Reconstruct from decomposition
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.allclose(recon, T, atol=1e-9):
        return False
    return True

def _build_rank44_decomposition(n, m, p):
    """
    Build a rank-44 decomposition for 2x3 x 3x8 matrix multiplication.
    Strategy: Split B into 4 blocks of 3x2.
    For each block, decompose the 2x3x2 multiplication into:
    - 2x2x2 part (j=0,1) using Strassen (rank 7)
    - 2x1x2 part (j=2) using schoolbook (rank 4)
    Total rank per block: 11. Total rank: 4 * 11 = 44.
    """
    n, m, p = 2, 3, 8
    rank = 44
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map these to the global indices.
    # Block s corresponds to columns 2s, 2s+1 of B and C.
    # For a block s:
    # A is 2x3. B_s is 3x2.
    # We split B_s into B_s[:,0] (col 0) and B_s[:,1] (col 1).
    # And A into A[:,0:2] (cols 0,1) and A[:,2] (col 2).
    
    # Indices in U: i*m + j. i in {0,1}, j in {0,1,2}.
    # Indices in V: j*p + k. j in {0,1,2}, k in {0..7}.
    # Indices in W: k*n + i. k in {0..7}, i in {0,1}.
    
    for s in range(4):
        base_rank = s * 11
        # k indices for this block
        k0 = 2 * s
        k1 = 2 * s + 1
        
        # --- Part 1: 2x2x2 Strassen for j=0,1 ---
        # This part computes C_i, {k0, k1} using A_{i, 0:2} and B_{0:2, {k0, k1}}
        
        # M1
        c = base_rank + 0
        # U: A00, A11
        U[0*3+0, c] = 1.0 # A00
        U[1*3+1, c] = 1.0 # A11
        # V: B00, B11 (rows 0,1 of B_s)
        V[0*8+k0, c] = 1.0 # B0, k0
        V[1*8+k1, c] = 1.0 # B1, k1
        # W: C00, C11
        W[k0*2+0, c] = 1.0 # C0, k0
        W[k1*2+1, c] = 1.0 # C1, k1
        
        # M2
        c = base_rank + 1
        # U: A10, A11
        U[1*3+0, c] = 1.0
        U[1*3+1, c] = 1.0
        # V: B00
        V[0*8+k0, c] = 1.0
        # W: C10
        W[k0*2+1, c] = 1.0
        
        # M3
        c = base_rank + 2
        # U: A00
        U[0*3+0, c] = 1.0
        # V: B01, B11
        V[0*8+k1, c] = 1.0
        V[1*8+k1, c] = -1.0
        # W: C01
        W[k1*2+0, c] = 1.0
        
        # M4
        c = base_rank + 3
        # U: A11
        U[1*3+1, c] = 1.0
        # V: B10, B00
        V[1*8+k0, c] = 1.0
        V[0*8+k0, c] = -1.0
        # W: C00, C10
        W[k0*2+0, c] = 1.0
        W[k0*2+1, c] = 1.0
        
        # M5
        c = base_rank + 4
        # U: A00, A01
        U[0*3+0, c] = 1.0
        U[0*3+1, c] = 1.0
        # V: B11
        V[1*8+k1, c] = 1.0
        # W: C00, C01
        W[k0*2+0, c] = -1.0
        W[k1*2+0, c] = 1.0
        
        # M6
        c = base_rank + 5
        # U: A10, A00
        U[1*3+0, c] = 1.0
        U[0*3+0, c] = -1.0
        # V: B00, B01
        V[0*8+k0, c] = 1.0
        V[0*8+k1, c] = 1.0
        # W: C11
        W[k1*2+1, c] = 1.0
        
        # M7
        c = base_rank + 6
        # U: A01, A11
        U[0*3+1, c] = 1.0
        U[1*3+1, c] = -1.0
        # V: B10, B11
        V[1*8+k0, c] = 1.0
        V[1*8+k1, c] = 1.0
        # W: C00
        W[k0*2+0, c] = 1.0

        # --- Part 2: 2x1x2 Schoolbook for j=2 ---
        # This part computes C_i, {k0, k1} using A_{i, 2} and B_{2, {k0, k1}}
        # Terms: A_i2 * B_2k
        
        # Term 1: A02 * B2k0 -> C0k0
        c = base_rank + 7
        U[0*3+2, c] = 1.0
        V[2*8+k0, c] = 1.0
        W[k0*2+0, c] = 1.0
        
        # Term 2: A02 * B2k1 -> C0k1
        c = base_rank + 8
        U[0*3+2, c] = 1.0
        V[2*8+k1, c] = 1.0
        W[k1*2+0, c] = 1.0
        
        # Term 3: A12 * B2k0 -> C1k0
        c = base_rank + 9
        U[1*3+2, c] = 1.0
        V[2*8+k0, c] = 1.0
        W[k0*2+1, c] = 1.0
        
        # Term 4: A12 * B2k1 -> C1k1
        c = base_rank + 10
        U[1*3+2, c] = 1.0
        V[2*8+k1, c] = 1.0
        W[k1*2+1, c] = 1.0

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    """
    Rank-44 decomposition of 2x3*3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    result = _build_rank44_decomposition(n, m, p)
    
    # Verify
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback to schoolbook if verification fails (should not happen)
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
# EVOLVE-BLOCK-END
