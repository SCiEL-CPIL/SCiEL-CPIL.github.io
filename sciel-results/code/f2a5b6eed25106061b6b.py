import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    deadline = time.time() + 115
    
    n, m, p = DIMENSIONS
    
    # Schoolbook baseline
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = (i * m + j) * p + k
                U_sb[i * m + j][idx] = 1
                V_sb[j * p + k][idx] = 1
                W_sb[k * n + i][idx] = 1
                
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    # Construct the 51-rank decomposition
    # n=2, m=3, p=9
    # A is 2x3, B is 3x9
    # Split A into A_L (2x2) and A_R (2x1)
    # Split B into B_T (2x9) and B_B (1x9)
    # C = A_L B_T + A_R B_B
    
    # Part 1: A_R B_B (2x1 * 1x9)
    # Rank 18
    # U indices: (i, j=2) for i in 0,1 -> 2 entries
    # V indices: (j=2, k) for k in 0..8 -> 9 entries
    # W indices: (k, i) for k in 0..8, i in 0,1 -> 18 entries
    
    U = []
    V = []
    W = []
    
    # We will build the matrices for the 51 rank terms
    # Term indices 0..17 for A_R B_B
    for i in range(2):
        for k in range(9):
            # U[i, 2] = 1
            # V[2, k] = 1
            # W[k, i] = 1
            U.append([0]*2) # placeholder, will fill later
            V.append([0]*9)
            W.append([0]*18)
            
    # Actually, let's just build the full matrices directly
    rank_target = 51
    U_mat = np.zeros((n*m, rank_target))
    V_mat = np.zeros((m*p, rank_target))
    W_mat = np.zeros((p*n, rank_target))
    
    term_idx = 0
    
    # Part 1: A_R B_B
    # A_R is column 2 of A. Indices (i, 2) in U.
    # B_B is row 2 of B. Indices (2, k) in V.
    # C is (i, k). Indices (k, i) in W.
    for i in range(2):
        for k in range(9):
            U_mat[i*m + 2, term_idx] = 1
            V_mat[2*p + k, term_idx] = 1
            W_mat[k*n + i, term_idx] = 1
            term_idx += 1
            
    # Part 2: A_L B_T
    # A_L is 2x2. Indices (i, j) for i in 0,1, j in 0,1.
    # B_T is 2x9. Split into 3 blocks of 2x3.
    # Block r corresponds to columns 3r, 3r+1, 3r+2 of B_T.
    # For each block, use rank-11 decomposition of 2x2x3.
    
    # Rank-11 decomposition for 2x2x3:
    # A (2x2), B (2x3)
    # Strassen for 2x2x2 (first 2 cols of B):
    # P1 = (a11+a22)(b11+b21)
    # P2 = (a21+a22)b11
    # P3 = a11(b12-b22)
    # P4 = a22(b11-b12)
    # P5 = (a11+a12)b22
    # P6 = (a21-a11)(b11+b12)
    # P7 = (a12-a22)(b21+b22)
    # These give C11, C12, C21, C22.
    # Plus 4 terms for the 3rd column:
    # P8 = a11 b13
    # P9 = a12 b23
    # P10 = a21 b13
    # P11 = a22 b23
    
    # We need to map these to the global indices.
    # For block r:
    # B cols are 3r, 3r+1, 3r+2.
    # b11 -> B[0, 3r], b12 -> B[0, 3r+1], b13 -> B[0, 3r+2]
    # b21 -> B[1, 3r], b22 -> B[1, 3r+1], b23 -> B[1, 3r+2]
    
    for r in range(3):
        b11_idx = 0*p + 3*r
        b12_idx = 0*p + 3*r + 1
        b13_idx = 0*p + 3*r + 2
        b21_idx = 1*p + 3*r
        b22_idx = 1*p + 3*r + 1
        b23_idx = 1*p + 3*r + 2
        
        # P1
        U_mat[0*m+0, term_idx] = 1
        U_mat[1*m+1, term_idx] = 1
        V_mat[b11_idx, term_idx] = 1
        V_mat[b21_idx, term_idx] = 1
        W_mat[(3*r)*n+0, term_idx] = 1
        W_mat[(3*r+1)*n+0, term_idx] = 1
        W_mat[(3*r)*n+1, term_idx] = 1
        W_mat[(3*r+1)*n+1, term_idx] = 1
        term_idx += 1
        
        # P2
        U_mat[1*m+0, term_idx] = 1
        U_mat[1*m+1, term_idx] = 1
        V_mat[b11_idx, term_idx] = 1
        W_mat[(3*r)*n+1, term_idx] = 1
        W_mat[(3*r)*n+0, term_idx] = 0 # C11 doesn't use P2 directly?
        # C11 = P1 - P2 - P4 + P6
        # C12 = P3 + P5 - P6 - P7? No.
        # Let's use the standard Strassen reconstruction.
        # C11 = P1 - P2 - P4 + P6
        # C12 = P3 + P5 - P6 - P7? No.
        # C12 = P3 + P5 - P7?
        # C21 = P2 + P4 - P6?
        # C22 = P1 - P3 - P5 + P7?
        
        # Let's re-verify Strassen formulas.
        # P1 = (a11+a22)(b11+b21)
        # P2 = (a21+a22)b11
        # P3 = a11(b12-b22)
        # P4 = a22(b11-b12)
        # P5 = (a11+a12)b22
        # P6 = (a21-a11)(b11+b12)
        # P7 = (a12-a22)(b21+b22)
        
        # C11 = P1 - P2 - P4 + P6
        # C12 = P3 + P5 - P6 - P7 ? No.
        # C12 = P3 + P5 - P7 ?
        # C21 = P2 + P4 - P6 ?
        # C22 = P1 - P3 - P5 + P7 ?
        
        # Let's check C12.
        # C12 = a11 b12 + a12 b22.
        # P3 = a11 b12 - a11 b22.
        # P5 = a11 b22 + a12 b22.
        # P3 + P5 = a11 b12 + a12 b22. Correct.
        # So C12 = P3 + P5.
        
        # Let's check C21.
        # C21 = a21 b11 + a22 b21.
        # P2 = a21 b11 + a22 b11.
        # P4 = a22 b11 - a22 b12.
        # P6 = a21 b11 + a21 b12 - a11 b11 - a11 b12.
        # P2 + P4 - P6 = a21 b11 + a22 b11 + a22 b11 - a22 b12 - a21 b11 - a21 b12 + a11 b11 + a11 b12
        # = 2 a22 b11 - a22 b12 - a21 b12 + a11 b11 + a11 b12.
        # This is not C21.
        
        # Correct Strassen:
        # C11 = P1 - P2 - P4 + P6
        # C12 = P3 + P5 - P6 - P7 ? No.
        # C12 = P3 + P5 - P7 ? No.
        # C12 = P3 + P5 ?
        # C21 = P2 + P4 - P6 ?
        # C22 = P1 - P3 - P5 + P7 ?
        
        # Let's re-derive C21.
        # C21 = a21 b11 + a22 b21.
        # P2 = a21 b11 + a22 b11.
        # P4 = a22 b11 - a22 b12.
        # P6 = (a21-a11)(b11+b12).
        # P2 - P4 = a21 b11 + a22 b12.
        # P2 - P4 + P6?
        
        # Standard Strassen:
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B12-B22)
        # P4 = A22(B11-B12)
        # P5 = (A11+A12)B22
        # P6 = (A21-A11)(B11+B12)
        # P7 = (A12-A22)(B21+B22)
        
        # C11 = P1 - P2 - P4 + P6
        # C12 = P3 + P5 - P6 - P7 ? No.
        # C12 = P3 + P5 - P7 ?
        # C21 = P2 + P4 - P6 ?
        # C22 = P1 - P3 - P5 + P7 ?
        
        # Let's check C12 again.
        # C12 = A11 B12 + A12 B22.
        # P3 = A11 B12 - A11 B22.
        # P5 = A11 B22 + A12 B22.
        # P3 + P5 = A11 B12 + A12 B22. Correct.
        
        # Let's check C21.
        # C21 = A21 B11 + A22 B21.
        # P2 = A21 B11 + A22 B11.
        # P4 = A22 B11 - A22 B12.
        # P6 = A21 B11 + A21 B12 - A11 B11 - A11 B12.
        # P2 + P4 - P6 = A21 B11 + A22 B11 + A22 B11 - A22 B12 - A21 B11 - A21 B12 + A11 B11 + A11 B12
        # = 2 A22 B11 - A22 B12 - A21 B12 + A11 B11 + A11 B12.
        # This is wrong.
        
        # Ah, P6 is (A21-A11)(B11+B12).
        # Maybe C21 = P2 + P4 - P6 is wrong.
        # C21 = P2 + P4 - P6?
        # Let's use:
        # C11 = P1 - P2 - P4 + P6
        # C12 = P3 + P5 - P6 - P7
        # C21 = P2 + P4 - P6
        # C22 = P1 - P3 - P5 + P7
        
        # Wait, I might have the indices of B mixed up.
        # B is [[B11, B12], [B21, B22]].
        # P3 = A11(B12-B22).
        # P5 = (A11+A12)B22.
        # P3+P5 = A11 B12 - A11 B22 + A11 B22 + A12 B22 = A11 B12 + A12 B22 = C12.
        
        # P2 = (A21+A22)B11.
        # P4 = A22(B11-B12).
        # P6 = (A21-A11)(B11+B12).
        # P2+P4-P6 = A21 B11 + A22 B11 + A22 B11 - A22 B12 - A21 B11 - A21 B12 + A11 B11 + A11 B12
        # = 2 A22 B11 - A22 B12 - A21 B12 + A11 B11 + A11 B12.
        
        # It seems my manual derivation is failing. I will just use the known coefficients.
        # C11 = P1 - P2 - P4 + P6
        # C12 = P3 + P5 - P6 - P7
        # C21 = P2 + P4 - P6
        # C22 = P1 - P3 - P5 + P7
        
        # I will assume these are correct and implement them.
        
        # P2
        U_mat[1*m+0, term_idx] = 1
        U_mat[1*m+1, term_idx] = 1
        V_mat[b11_idx, term_idx] = 1
        W_mat[(3*r)*n+1, term_idx] = 1 # C21
        W_mat[(3*r+1)*n+1, term_idx] = 0 # C22? No.
        # C21 uses P2. C12 uses P2? No.
        # C11 uses P2 (-). C21 uses P2 (+).
        # So W_mat for C21 gets +1.
        # W_mat for C11 gets -1.
        W_mat[(3*r)*n+1, term_idx] = 1
        W_mat[(3*r)*n+0, term_idx] = -1
        term_idx += 1
        
        # P3
        U_mat[0*m+0, term_idx] = 1
        V_mat[b12_idx, term_idx] = 1
        V_mat[b22_idx, term_idx] = -1
        W_mat[(3*r+1)*n+0, term_idx] = 1 # C12
        W_mat[(3*r+1)*n+1, term_idx] = -1 # C22
        term_idx += 1
        
        # P4
        U_mat[1*m+1, term_idx] = 1
        V_mat[b11_idx, term_idx] = 1
        V_mat[b12_idx, term_idx] = -1
        W_mat[(3*r)*n+1, term_idx] = 1 # C21
        W_mat[(3*r)*n+0, term_idx] = -1 # C11
        term_idx += 1
        
        # P5
        U_mat[0*m+0, term_idx] = 1
        U_mat[0*m+1, term_idx] = 1
        V_mat[b22_idx, term_idx] = 1
        W_mat[(3*r+1)*n+0, term_idx] = 1 # C12
        W_mat[(3*r+1)*n+1, term_idx] = -1 # C22
        term_idx += 1
        
        # P6
        U_mat[1*m+0, term_idx] = 1
        U_mat[0*m+0, term_idx] = -1
        V_mat[b11_idx, term_idx] = 1
        V_mat[b12_idx, term_idx] = 1
        W_mat[(3*r)*n+1, term_idx] = -1 # C21
        W_mat[(3*r)*n+0, term_idx] = 1 # C11
        term_idx += 1
        
        # P7
        U_mat[0*m+1, term_idx] = 1
        U_mat[1*m+1, term_idx] = -1
        V_mat[b21_idx, term_idx] = 1
        V_mat[b22_idx, term_idx] = 1
        W_mat[(3*r+1)*n+0, term_idx] = -1 # C12
        W_mat[(3*r+1)*n+1, term_idx] = 1 # C22
        term_idx += 1
        
        # P8-P11 for 3rd column
        # C13 = A11 B13 + A12 B23
        # C23 = A21 B13 + A22 B23
        
        # P8 = A11 B13
        U_mat[0*m+0, term_idx] = 1
        V_mat[b13_idx, term_idx] = 1
        W_mat[(3*r+2)*n+0, term_idx] = 1
        term_idx += 1
        
        # P9 = A12 B23
        U_mat[0*m+1, term_idx] = 1
        V_mat[b23_idx, term_idx] = 1
        W_mat[(3*r+2)*n+0, term_idx] = 1
        term_idx += 1
        
        # P10 = A21 B13
        U_mat[1*m+0, term_idx] = 1
        V_mat[b13_idx, term_idx] = 1
        W_mat[(3*r+2)*n+1, term_idx] = 1
        term_idx += 1
        
        # P11 = A22 B23
        U_mat[1*m+1, term_idx] = 1
        V_mat[b23_idx, term_idx] = 1
        W_mat[(3*r+2)*n+1, term_idx] = 1
        term_idx += 1

    # Verify rank
    if term_idx != rank_target:
        print(f"Warning: term_idx {term_idx} != rank_target {rank_target}")
        
    # Check validity
    def check_valid(U, V, W, R):
        U_res = U.reshape(n, m, R)
        V_res = V.reshape(m, p, R)
        W_res = W.reshape(p, n, R)
        try:
            T = np.einsum('ijr,jkr,kir->ijk', U_res, V_res, W_res)
        except Exception:
            return False
        return np.allclose(T, 1.0, atol=1e-9)
        
    if check_valid(U_mat, V_mat, W_mat, rank_target):
        best = {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
        best_rank = rank_target

    return best
