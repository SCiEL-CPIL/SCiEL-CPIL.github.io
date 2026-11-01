"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n=3, m=3, p=3, tol=1e-9):
    """
    Verify that U, V, W form a valid tensor decomposition for matrix multiplication.
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    """
    try:
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_vec = U_arr[i*m + j]
                    v_vec = V_arr[j*p + k]
                    w_vec = W_arr[k*n + i]
                    s = np.sum(u_vec * v_vec * w_vec)
                    if abs(s - 1.0) > tol:
                        return False
        return True
    except Exception:
        return False

def construct():
    n, m, p = 3, 3, 3
    
    # Rank 26 Construction:
    # We partition the 3x3 multiplication into a 2x2 block (top-left) and the rest.
    # The 2x2 block multiplication has rank 7 (Strassen).
    # The remaining 19 bilinear forms (involving index 3) are independent.
    # Total rank = 7 + 19 = 26.
    
    rank26 = 26
    U26 = [[0] * rank26 for _ in range(n * m)]
    V26 = [[0] * rank26 for _ in range(m * p)]
    W26 = [[0] * rank26 for _ in range(p * n)]
    
    # Indices for 3x3: 0,1,2
    # Top-left 2x2 block: i,j in {0,1}
    # Strassen formulas for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B12)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Map 2x2 indices (0,1) to 3x3 indices (0,1)
    # Strassen col 0..6
    
    # M1: (A00+A11)(B00+B11) -> W00, W11, W01, W10 contributions?
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 - M3 + M6
    
    # Let's populate U, V, W for the 7 Strassen terms.
    # U is (A side), V is (B side), W is (C side coeff)
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    U26[0*3+0][0] = 1; U26[1*3+1][0] = 1
    V26[0*3+0][0] = 1; V26[1*3+1][0] = 1
    W26[0*3+0][0] = 1; W26[1*3+1][0] = 1; W26[0*3+1][0] = -1; W26[1*3+0][0] = -1
    
    # Term 1: M2 = (A10+A11)B00
    U26[1*3+0][1] = 1; U26[1*3+1][1] = 1
    V26[0*3+0][1] = 1
    W26[1*3+0][1] = 1; W26[1*3+1][1] = -1
    
    # Term 2: M3 = A00(B10-B11)
    U26[0*3+0][2] = 1
    V26[1*3+0][2] = 1; V26[1*3+1][2] = -1
    W26[0*3+1][2] = 1; W26[1*3+0][2] = -1
    
    # Term 3: M4 = A11(B00-B01)
    U26[1*3+1][3] = 1
    V26[0*3+0][3] = 1; V26[0*3+1][3] = -1
    W26[0*3+0][3] = -1; W26[1*3+0][3] = 1
    
    # Term 4: M5 = (A00+A01)B11
    U26[0*3+0][4] = 1; U26[0*3+1][4] = 1
    V26[1*3+1][4] = 1
    W26[0*3+0][4] = -1; W26[0*3+1][4] = 1
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    U26[1*3+0][5] = 1; U26[0*3+0][5] = -1
    V26[0*3+0][5] = 1; V26[0*3+1][5] = 1
    W26[1*3+1][5] = 1
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    U26[0*3+1][6] = 1; U26[1*3+1][6] = -1
    V26[1*3+0][6] = 1; V26[1*3+1][6] = 1
    W26[0*3+0][6] = 1
    
    # Now the remaining 19 terms.
    # These are the schoolbook terms for any product involving index 2.
    # C_ij = A_ik B_kj.
    # If i=2 or j=2 or k=2, it's not in the 2x2 block.
    # The 2x2 block covers i,j in {0,1} and k in {0,1}.
    # The "remaining" bilinear forms are:
    # 1. k=2, i,j in {0,1,2} -> 9 terms: A_i2 B_2j
    # 2. k in {0,1}, i in {0,1}, j=2 -> 4 terms: A_ik B_k2
    # 3. k in {0,1}, i=2, j in {0,1} -> 4 terms: A_2k B_kj
    # Total 9 + 4 + 4 = 17?
    # Wait, the 2x2 block calculation used Strassen, which replaced the 8 terms A_ik B_kj for i,j,k in {0,1}.
    # The full schoolbook has 27 terms.
    # The 8 terms for i,j,k in {0,1} are replaced by 7 Strassen terms.
    # The other 19 terms are untouched.
    # So we just add the 19 schoolbook terms for the "rest".
    
    col = 7
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Skip if i,j,k are all in {0,1}
                if i < 2 and j < 2 and k < 2:
                    continue
                U26[i*3+k][col] = 1
                V26[k*3+j][col] = 1
                W26[j*3+i][col] = 1
                col += 1
                
    # col should be 7 + 19 = 26.
    
    if not _verify_decomposition(U26, V26, W26):
        # Fallback to Rank 27 if verification fails
        rank27 = 27
        U27 = [[0] * rank27 for _ in range(n * m)]
        V27 = [[0] * rank27 for _ in range(m * p)]
        W27 = [[0] * rank27 for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U27[i * m + j][col] = 1
                    V27[j * p + k][col] = 1
                    W27[k * n + i][col] = 1
                    col += 1
        return {"U": U27, "V": V27, "W": W27}
        
    return {"U": U26, "V": V26, "W": W26}

# EVOLVE-BLOCK-END
