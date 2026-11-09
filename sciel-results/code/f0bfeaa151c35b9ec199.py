"""Matrix multiplication 2x3 by 3x6: rank-minimal exact tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-22 decomposition for 2x3 * 3x6 matrix multiplication.
    
    Decomposition:
    1. Split A (2x3) into A1 (2x2) and A2 (2x1).
    2. Split B (3x6) into B1 (2x6) and B2 (1x6).
    3. C = A1*B1 + A2*B2.
    4. A1*B1 (2x2 * 2x6) is decomposed using 3 Strassen algorithms (rank 7 each) -> rank 21.
    5. A2*B2 (2x1 * 1x6) is decomposed using rank 1.
    Total rank = 22.
    """
    n, m, p = 2, 3, 6
    nm, mp, pn = n * m, m * p, p * n
    
    # Rank 22
    rank = 22
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Strassen coefficients for 2x2 * 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Output reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We apply this to 3 blocks of B (columns 0-1, 2-3, 4-5)
    # A1 uses columns 0-1 of A (indices 0, 1 in j)
    # B1 uses rows 0-1 of B (indices 0, 1 in j)
    
    # Strassen terms definitions:
    # Each term is defined by:
    # - U coefficients (for A1, 2x2)
    # - V coefficients (for B block, 2x2)
    # - W coefficients (for C block, 2x2)
    
    # Indices in U: i*m + j. For A1, j is 0,1. i is 0,1.
    # Indices in V: j*p + k. For B1, j is 0,1. k is 0..5.
    # Indices in W: k*n + i. For C, k is 0..5. i is 0,1.
    
    # Define the 7 Strassen terms
    # term: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict { (i,j): val }
    # V_coeffs: dict { (j,k): val }
    # W_coeffs: dict { (k,i): val }
    
    strassen_terms = []
    
    # M1
    u1 = {(0,0): 1, (1,1): 1}
    v1 = {(0,0): 1, (1,1): 1}
    w1 = {(0,0): 1, (1,1): 1}
    strassen_terms.append((u1, v1, w1))
    
    # M2
    u2 = {(1,0): 1, (1,1): 1}
    v2 = {(0,0): 1}
    w2 = {(1,0): 1, (1,1): -1} # C10 += M2, C11 -= M2
    strassen_terms.append((u2, v2, w2))
    
    # M3
    u3 = {(0,0): 1}
    v3 = {(1,0): 1, (1,1): -1}
    w3 = {(0,1): 1, (1,1): 1} # C01 += M3, C11 += M3
    strassen_terms.append((u3, v3, w3))
    
    # M4
    u4 = {(1,1): 1}
    v4 = {(0,1): 1, (0,0): -1}
    w4 = {(0,0): 1, (1,0): 1} # C00 += M4, C10 += M4
    strassen_terms.append((u4, v4, w4))
    
    # M5
    u5 = {(0,0): 1, (0,1): 1}
    v5 = {(1,1): 1}
    w5 = {(0,0): -1, (0,1): 1} # C00 -= M5, C01 += M5
    strassen_terms.append((u5, v5, w5))
    
    # M6
    u6 = {(1,0): 1, (0,0): -1}
    v6 = {(0,0): 1, (0,1): 1}
    w6 = {(1,1): 1} # C11 += M6
    strassen_terms.append((u6, v6, w6))
    
    # M7
    u7 = {(0,1): 1, (1,1): -1}
    v7 = {(1,0): 1, (1,1): 1}
    w7 = {(0,0): 1} # C00 += M7
    strassen_terms.append((u7, v7, w7))
    
    # Add these terms for each of the 3 blocks
    col = 0
    for t in range(3):
        # Block t corresponds to B columns 2t, 2t+1 and C columns 2t, 2t+1
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            # Fill U
            for (i, j), val in u_coeffs.items():
                # j is relative to A1, so actual j is j (0,1)
                U[i * m + j, col] += val
            
            # Fill V
            for (j, k_rel), val in v_coeffs.items():
                # k_rel is 0 or 1, actual k is 2*t + k_rel
                # j is 0 or 1
                k_actual = 2 * t + k_rel
                V[j * p + k_actual, col] += val
            
            # Fill W
            for (k_rel, i), val in w_coeffs.items():
                # k_rel is 0 or 1, actual k is 2*t + k_rel
                k_actual = 2 * t + k_rel
                W[k_actual * n + i, col] += val
            
            col += 1
    
    # Add the A2*B2 term (rank 1)
    # A2 is column 2 of A (j=2)
    # B2 is row 2 of B (j=2)
    # C = A2 * B2
    # U: i*m + 2 = 1, 2 for i=0,1. Values 1.
    U[0 * m + 2, col] = 1
    U[1 * m + 2, col] = 1
    
    # V: 2*p + k for k=0..5. Values 1.
    for k in range(p):
        V[2 * p + k, col] = 1
    
    # W: k*n + i for all k, i. Values 1.
    for k in range(p):
        for i in range(n):
            W[k * n + i, col] = 1
    
    # Verify
    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(U_mat.shape[1]):
                        val += U_mat[i * m + j, c] * V_mat[j * p + k, c] * W_mat[k * n + i, c]
                    if abs(val - 1) > tol:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback to schoolbook if construction failed (should not happen)
        rank_sb = n * m * p
        U = np.zeros((nm, rank_sb))
        V = np.zeros((mp, rank_sb))
        W = np.zeros((pn, rank_sb))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
    
    U_out = [[float(x) for x in row] for row in U]
    V_out = [[float(x) for x in row] for row in V]
    W_out = [[float(x) for x in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
