import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # P1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # P2 = (A10+A11) B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = 1
    
    # P3 = A00 (B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = -1
    
    # P4 = A11 (B01-B10)
    U[3, 3] = 1
    V[1, 3] = 1; V[2, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # P5 = (A00+A01) B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # P6 = (A10-A00) (B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = -1
    
    # P7 = (A01-A11) (B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _expand_strassen(n, U_base, V_base, W_base):
    """
    Recursively expand the Strassen decomposition from size 2 to size n (power of 2).
    """
    if n == 2:
        return U_base.copy(), V_base.copy(), W_base.copy()
    
    n_half = n // 2
    U_prev, V_prev, W_prev = _expand_strassen(n_half, U_base, V_base, W_base)
    rank_prev = U_prev.shape[1]
    rank_new = rank_prev * 7
    
    U = np.zeros((n*n, rank_new), dtype=float)
    V = np.zeros((n*n, rank_new), dtype=float)
    W = np.zeros((n*n, rank_new), dtype=float)
    
    alpha = [
        np.array([[1,0],[0,1]], dtype=float),
        np.array([[0,0],[1,1]], dtype=float),
        np.array([[1,0],[0,0]], dtype=float),
        np.array([[0,0],[0,1]], dtype=float),
        np.array([[1,1],[0,0]], dtype=float),
        np.array([[-1,0],[1,0]], dtype=float),
        np.array([[0,1],[0,-1]], dtype=float),
    ]
    
    beta = [
        np.array([[1,0],[0,1]], dtype=float),
        np.array([[1,0],[0,0]], dtype=float),
        np.array([[0,0],[1,-1]], dtype=float),
        np.array([[0,1],[-1,0]], dtype=float),
        np.array([[0,0],[0,1]], dtype=float),
        np.array([[1,1],[0,0]], dtype=float),
        np.array([[0,1],[0,1]], dtype=float),
    ]
    
    gamma = [
        np.array([[1,0],[0,1]], dtype=float),
        np.array([[0,0],[1,1]], dtype=float),
        np.array([[0,1],[0,-1]], dtype=float),
        np.array([[1,0],[1,0]], dtype=float),
        np.array([[-1,0],[1,0]], dtype=float),
        np.array([[0,0],[0,-1]], dtype=float),
        np.array([[1,0],[0,0]], dtype=float),
    ]
    
    for r in range(7):
        for I in range(2):
            for J in range(2):
                a_coeff = alpha[r][I, J]
                if a_coeff == 0:
                    continue
                for i in range(n_half):
                    for j in range(n_half):
                        row_idx = (I * n_half + i) * n + (J * n_half + j)
                        sub_row = i * n_half + j
                        for r_prev in range(rank_prev):
                            U[row_idx, r * rank_prev + r_prev] += a_coeff * U_prev[sub_row, r_prev]
        
        for J in range(2):
            for K in range(2):
                b_coeff = beta[r][J, K]
                if b_coeff == 0:
                    continue
                for j in range(n_half):
                    for k in range(n_half):
                        row_idx = (J * n_half + j) * n + (K * n_half + k)
                        sub_row = j * n_half + k
                        for r_prev in range(rank_prev):
                            V[row_idx, r * rank_prev + r_prev] += b_coeff * V_prev[sub_row, r_prev]
        
        for K in range(2):
            for I in range(2):
                c_coeff = gamma[r][K, I]
                if c_coeff == 0:
                    continue
                for k in range(n_half):
                    for i in range(n_half):
                        row_idx = (K * n_half + k) * n + (I * n_half + i)
                        sub_row = k * n_half + i
                        for r_prev in range(rank_prev):
                            W[row_idx, r * rank_prev + r_prev] += c_coeff * W_prev[sub_row, r_prev]
    
    return U, V, W

def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 5x5x5 matrix multiplication.
    Returns a dict with keys 'U', 'V', 'W' containing the factor matrices.
    
    Strategy: Use a 2x2x2 Strassen decomposition (rank 7) for the first 4x4x4 block,
    then add schoolbook terms for the remaining parts involving the 5th row/column.
    
    C[i,k] = sum_j A[i,j] B[j,k]
    
    Split into:
    - i<4, k<4: handled by 4x4x4 Strassen (rank 49)
    - i=4, all k: C[4,k] = sum_j A[4,j] B[j,k] -> 25 schoolbook terms
    - i<4, k=4: C[i,4] = sum_j A[i,j] B[j,4] -> 20 schoolbook terms (j=0..3) + 1 (j=4) = 21? 
      Wait, let me recount.
    
    Actually, let's be more careful. The 4x4x4 Strassen handles:
    - A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4]
    
    Remaining terms:
    1. C[i, k] for i in 0..3, k=4: needs sum_{j=0}^{4} A[i,j] B[j,4]
       - For j=0..3: 4*4 = 16 terms (A[i,0..3] * B[0..3,4])
       - For j=4: 4 terms (A[i,4] * B[4,4])
       Total: 20 terms
    
    2. C[4, k] for k in 0..4: needs sum_{j=0}^{4} A[4,j] B[j,k]
       - For j=0..3: 4*4 = 16 terms (A[4,0..3] * B[0..3,0..3])
       - For j=4: 4 terms (A[4,4] * B[4,0..3])
       Total: 20 terms
    
    3. C[4, 4]: needs sum_{j=0}^{4} A[4,j] B[j,4]
       - For j=0..3: 4 terms (A[4,0..3] * B[0..3,4])
       - For j=4: 1 term (A[4,4] * B[4,4])
       Total: 5 terms
    
    Wait, but some of these overlap with the Strassen part? No, the Strassen part only handles
    i<4, j<4, k<4. So any term with i=4 or k=4 is not in the Strassen part.
    
    Total additional rank = 20 + 20 + 5 = 45
    Total rank = 49 + 45 = 94
    """
    n, m, p = DIMENSIONS
    
    # Build 4x4x4 Strassen decomposition (rank 49)
    U4, V4, W4 = _expand_strassen(4, *_strassen_2x2x2())
    rank_4x4 = U4.shape[1]  # Should be 49
    
    # Total rank = 49 + 20 + 20 + 5 = 94
    rank_total = rank_4x4 + 45
    
    U = np.zeros((25, rank_total), dtype=float)
    V = np.zeros((25, rank_total), dtype=float)
    W = np.zeros((25, rank_total), dtype=float)
    
    # Copy the 4x4x4 part into the first 49 columns
    for i in range(4):
        for j in range(4):
            row_u = i * 5 + j
            for r in range(rank_4x4):
                U[row_u, r] = U4[i * 4 + j, r]
    
    for j in range(4):
        for k in range(4):
            row_v = j * 5 + k
            for r in range(rank_4x4):
                V[row_v, r] = V4[j * 4 + k, r]
    
    for k in range(4):
        for i in range(4):
            row_w = k * 5 + i
            for r in range(rank_4x4):
                W[row_w, r] = W4[k * 4 + i, r]
    
    col_offset = rank_4x4
    
    # Add schoolbook terms for i=4 (row 4 of A): C[4,k] = sum_j A[4,j]*B[j,k]
    # For k=0..4, j=0..4: 25 terms
    for j in range(5):
        for k in range(5):
            U[4 * 5 + j, col_offset] = 1.0
            V[j * 5 + k, col_offset] = 1.0
            W[k * 5 + 4, col_offset] = 1.0
            col_offset += 1
    
    # Add schoolbook terms for k=4 (column 4 of B): C[i,4] = sum_j A[i,j]*B[j,4]
    # For i=0..3, j=0..4: 20 terms
    for i in range(4):
        for j in range(5):
            U[i * 5 + j, col_offset] = 1.0
            V[j * 5 + 4, col_offset] = 1.0
            W[4 * 5 + i, col_offset] = 1.0
            col_offset += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
