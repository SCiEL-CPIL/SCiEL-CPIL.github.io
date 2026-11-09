"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 20
    mp = m * p  # 30
    pn = p * n  # 24
    
    deadline = time.time() + 145  # safety margin before 150s kill
    
    best_rank = n * m * p  # start with schoolbook (120)
    
    # First, provide the schoolbook as fallback
    schoolbook_U = np.zeros((nm, best_rank))
    schoolbook_V = np.zeros((mp, best_rank))
    schoolbook_W = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                schoolbook_U[i * m + j, col] = 1
                schoolbook_V[j * p + k, col] = 1
                schoolbook_W[k * n + i, col] = 1
                col += 1
    best_result = {
        "U": schoolbook_U.tolist(),
        "V": schoolbook_V.tolist(),
        "W": schoolbook_W.tolist()
    }
    
    def check_error(U, V, W, tol=1e-9):
        """Check max error of tensor decomposition using vectorized einsum."""
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        max_err = np.max(np.abs(T - 1.0))
        return float(max_err)
    
    # Deterministic construction: use Strassen for 4x4x4 block + schoolbook for remainder
    # This gives rank = 49 + 32 + 24 = 105
    #
    # Decomposition:
    # - Strassen part: A[0:4,0:4] * B[0:4,0:4] -> C[0:4,0:4] (49 terms)
    # - Schoolbook 1: A[0:4,0:4] * B[0:4,4:6] -> C[0:4,4:6] (32 terms)  
    # - Schoolbook 2: A[0:4,4:5] * B[4:5,0:6] -> C[0:4,0:6] (24 terms)
    
    # Strassen 2x2 base
    # For 2x2: 
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Build 4x4x4 Strassen decomposition (49 terms) using tensor product of 2x2 Strassen
    # The 4x4 case can be seen as Kronecker product of two 2x2 Strassen
    # But more directly, we can use the known 49-term decomposition
    
    # Let's build it systematically using the recursive structure
    # 4x4x4 = tensor product of two 2x2x2 Strassen decompositions
    
    # 2x2x2 Strassen: U has shape (4,7), V has shape (4,7), W has shape (4,7)
    # U rows indexed by (i,j) for i,j in {0,1}: row = i*2+j
    # V rows indexed by (j,k) for j,k in {0,1}: row = j*2+k
    # W rows indexed by (k,i) for k,i in {0,1}: row = k*2+i
    
    S_U = np.zeros((4, 7))
    S_V = np.zeros((4, 7))
    S_W = np.zeros((4, 7))
    
    # Term 0: (A00+A11)(B00+B11)
    S_U[0, 0] = 1; S_U[3, 0] = 1  # A00, A11
    S_V[0, 0] = 1; S_V[3, 0] = 1  # B00, B11
    S_W[0, 0] = 1; S_W[3, 0] = -1; S_W[1, 0] = 0; S_W[2, 0] = 0
    # C00 += M1, C11 -= M1 (partially)
    # Actually let me be more careful
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Let me redo this more carefully
    S_U = np.zeros((4, 7))
    S_V = np.zeros((4, 7))
    S_W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    S_U[0, 0] = 1; S_U[3, 0] = 1
    S_V[0, 0] = 1; S_V[3, 0] = 1
    S_W[0, 0] = 1   # C00 += M1
    S_W[3, 0] = 1   # C11 += M1
    
    # M2 = (A10+A11)B00
    S_U[2, 1] = 1; S_U[3, 1] = 1
    S_V[0, 1] = 1
    S_W[2, 1] = 1   # C10 += M2
    S_W[3, 1] = -1  # C11 -= M2
    
    # M3 = A00(B10-B11)
    S_U[0, 2] = 1
    S_V[2, 2] = 1; S_V[3, 2] = -1
    S_W[1, 2] = 1   # C01 += M3
    S_W[3, 2] = 1   # C11 += M3
    
    # M4 = A11(B01-B10)
    S_U[3, 3] = 1
    S_V[1, 3] = 1; S_V[2, 3] = -1
    S_W[0, 3] = 1   # C00 += M4
    S_W[2, 3] = 1   # C10 += M4
    
    # M5 = (A00+A01)B11
    S_U[0, 4] = 1; S_U[1, 4] = 1
    S_V[3, 4] = 1
    S_W[0, 4] = -1  # C00 -= M5
    S_W[1, 4] = 1   # C01 += M5
    
    # M6 = (A10-A00)(B00+B10)
    S_U[2, 5] = 1; S_U[0, 5] = -1
    S_V[0, 5] = 1; S_V[2, 5] = 1
    S_W[3, 5] = 0   # M6 not used in standard Strassen... 
    # Wait, standard Strassen only has 7 terms. Let me check.
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    # So M6 is not used! That's correct, we only need 7 terms but M6 is defined but unused.
    # Actually no, all 7 are needed. Let me recheck.
    # Hmm, actually the standard formula uses all 7. Let me re-derive.
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4  
    # C11 = M1 - M2 + M3 + M7
    # So M6 is indeed not used in the output. But we still count it as a term?
    # No! If M6 doesn't appear in any C, then we don't need it. 
    # Actually I think I made an error. Let me look up the correct Strassen.
    # Correct Strassen:
    # M1 = (A11-A01) * (B01+B11)
    # M2 = (A00+A01) * B11
    # M3 = (A10+A11) * B00
    # M4 = (A11-A10) * (B00+B10)
    # M5 = A00 * (B01-B11)
    # M6 = A11 * (B10-B00)
    # M7 = (A00+A11) * (B00+B11)
    # C00 = M5 + M7 - M2 + M6
    # C01 = M1 + M2
    # C10 = M3 + M4
    # C11 = M1 + M5 - M4 + M7
    # Hmm, this is getting confusing. Let me just use the well-known 7-term formula.
    
    # Let me restart with a clean 2x2 Strassen:
    S_U = np.zeros((4, 7))
    S_V = np.zeros((4, 7))
    S_W = np.zeros((4, 7))
    
    # Using the standard formulation where:
    # C = [[C00, C01], [C10, C11]]
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    #
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # M1 = (A00+A11)(B00+B11)
    S_U[0, 0] = 1; S_U[3, 0] = 1  # A00, A11 (rows 0,3)
    S_V[0, 0] = 1; S_V[3, 0] = 1  # B00, B11 (rows 0,3)
    S_W[0, 0] = 1   # C00 += M1
    S_W[3, 0] = 1   # C11 += M1
    
    # M2 = (A10+A11)B00
    S_U[2, 1] = 1; S_U[3, 1] = 1  # A10, A11 (rows 2,3)
    S_V[0, 1] = 1                 # B00 (row 0)
    S_W[2, 1] = 1   # C10 += M2
    S_W[3, 1] = -1  # C11 -= M2
    
    # M3 = A00(B10-B11)
    S_U[0, 2] = 1                 # A00 (row 0)
    S_V[2, 2] = 1; S_V[3, 2] = -1  # B10, -B11 (rows 2,3)
    S_W[1, 2] = 1   # C01 += M3
    S_W[3, 2] = 1   # C11 += M3
    
    # M4 = A11(B01-B10)
    S_U[3, 3] = 1                 # A11 (row 3)
    S_V[1, 3] = 1; S_V[2, 3] = -1  # B01, -B10 (rows 1,2)
    S_W[0, 3] = 1   # C00 += M4
    S_W[2, 3] = 1   # C10 += M4
    
    # M5 = (A00+A01)B11
    S_U[0, 4] = 1; S_U[1, 4] = 1  # A00, A01 (rows 0,1)
    S_V[3, 4] = 1                 # B11 (row 3)
    S_W[0, 4] = -1  # C00 -= M5
    S_W[1, 4] = 1   # C01 += M5
    
    # M6 = (A10-A00)(B00+B10)
    S_U[2, 5] = 1; S_U[0, 5] = -1  # A10, -A00 (rows 2,0)
    S_V[0, 5] = 1; S_V[2, 5] = 1   # B00, B10 (rows 0,2)
    # M6 doesn't appear in C formula directly... 
    # Wait, I think the standard Strassen doesn't use M6 in the final combination.
    # Let me check: C00 = M1 + M4 - M5 - M7, C01 = M3 + M5, C10 = M2 + M4, C11 = M1 - M2 + M3 + M7
    # So M6 is indeed unused. This means we only need 6 terms? No, that can't be right.
    # Actually, I think I have the wrong formula. The correct Strassen uses 7 multiplications
    # and all 7 appear in the output. Let me use a different reference formulation.
    
    # Alternative: Let me just hardcode a known correct 2x2 Strassen decomposition.
    # From Wikipedia:
    # P1 = (A11−A01)(B01+B11)
    # P2 = (A00+A01)B11
    # P3 = (A10+A11)B00
    # P4 = (A11−A10)(B00+B10)
    # P5 = A00(B01−B11)
    # P6 = A11(B10−B00)
    # P7 = (A00+A11)(B00+B11)
    # C00 = P5 + P7 − P2 + P6
    # C01 = P1 + P2
    # C10 = P3 + P4
    # C11 = P1 + P5 − P4 + P7
    
    S_U = np.zeros((4, 7))
    S_V = np.zeros((4, 7))
    S_W = np.zeros((4, 7))
    
    # P1 = (A11−A01)(B01+B11)
    S_U[3, 0] = 1; S_U[1, 0] = -1  # A11, -A01 (rows 3,1)
    S_V[1, 0] = 1; S_V[3, 0] = 1   # B01, B11 (rows 1,3)
    S_W[1, 0] = 1   # C01 += P1
    S_W[3, 0] = 1   # C11 += P1
    
    # P2 = (A00+A01)B11
    S_U[0, 1] = 1; S_U[1, 1] = 1  # A00, A01 (rows 0,1)
    S_V[3, 1] = 1                 # B11 (row 3)
    S_W[0, 1] = -1  # C00 -= P2
    S_W[1, 1] = 1   # C01 += P2
    
    # P3 = (A10+A11)B00
    S_U[2, 2] = 1; S_U[3, 2] = 1  # A10, A11 (rows 2,3)
    S_V[0, 2] = 1                 # B00 (row 0)
    S_W[2, 2] = 1   # C10 += P3
    
    # P4 = (A11−A10)(B00+B10)
    S_U[3, 3] = 1; S_U[2, 3] = -1  # A11, -A10 (rows 3,2)
    S_V[0, 3] = 1; S_V[2, 3] = 1   # B00, B10 (rows 0,2)
    S_W[2, 3] = 1   # C10 += P4
    S_W[3, 3] = -1  # C11 -= P4
    
    # P5 = A00(B01−B11)
    S_U[0, 4] = 1                 # A00 (row 0)
    S_V[1, 4] = 1; S_V[3, 4] = -1  # B01, -B11 (rows 1,3)
    S_W[0, 4] = 1   # C00 += P5
    S_W[3, 4] = 1   # C11 += P5
    
    # P6 = A11(B10−B00)
    S_U[3, 5] = 1                 # A11 (row 3)
    S_V[2, 5] = 1; S_V[0, 5] = -1  # B10, -B00 (rows 2,0)
    S_W[0, 5] = 1   # C00 += P6
    
    # P7 = (A00+A11)(B00+B11)
    S_U[0, 6] = 1; S_U[3, 6] = 1  # A00, A11 (rows 0,3)
    S_V[0, 6] = 1; S_V[3, 6] = 1  # B00, B11 (rows 0,3)
    S_W[0, 6] = 1   # C00 += P7
    S_W[3, 6] = 1   # C11 += P7
    
    # Verify 2x2 Strassen
    # T[i,j,k] = sum_c S_U[i,j,c] * S_V[j,k,c] * S_W[k,i,c]
    U2 = S_U.reshape(2, 2, 7)
    V2 = S_V.reshape(2, 2, 7)
    W2 = S_W.reshape(2, 2, 7)
    T2 = np.einsum('ijc,jkc,kic->ijk', U2, V2, W2)
    assert np.max(np.abs(T2 - 1.0)) < 1e-10, f"2x2 Strassen failed: {np.max(np.abs(T2 - 1.0))}"
    
    # Now build 4x4x4 using Kronecker product
    # 4x4 matrices can be seen as 2x2 blocks of 2x2 matrices
    # A = [[A11, A12], [A21, A22]] where each Ajk is 2x2
    # B = [[B11, B12], [B21, B22]] where each Bjk is 2x2
    # C = A*B = [[A11*B11+A12*B21, A11*B12+A12*B22], [A21*B11+A22*B21, A21*B12+A22*B22]]
    # 
    # We can compute each 2x2 block of C using Strassen on the block matrices
    # This gives 7*7 = 49 terms
    
    # The 4x4 indices: i,j in 0..3
    # Block index: (i//2, i%2), (j//2, j%2)
    # For the Kronecker product approach:
    # U_4[(i1,i2),(j1,j2), c1,c2] = S_U[i1,j1,c1] * S_U[i2,j2,c2]
    # But we need to be careful with the indexing.
    
    # Actually, for the tensor product, we have:
    # The 4x4x4 tensor T_4[i,j,k] = sum_{c1,c2} U_4[i,j,c1,c2] * V_4[j,k,c1,c2] * W_4[k,i,c1,c2]
    # where U_4[i,j,(c1,c2)] = S_U[i_block, j_block, c1] * S_U[i_local, j_local, c2]
    # and i = i_block*2 + i_local, j = j_block*2 + j_local, etc.
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U4[i * 4 + j, c] = S_U[i1 * 2 + j1, c1] * S_U[i2 * 2 + j2, c2]
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            V4[j * 4 + k, c] = S_V[j1 * 2 + k1, c1] * S_V[j2 * 2 + k2, c2]
            for k1 in range(2):
                for k2 in range(2):
                    for i1 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            W4[k * 4 + i, c] = S_W[k1 * 2 + i1, c1] * S_W[k2 * 2 + i2, c2]
    
    # Verify 4x4x4
    U4_3d = U4.reshape(4, 4, 49)
    V4_3d = V4.reshape(4, 4, 49)
    W4_3d = W4.reshape(4, 4, 49)
    T4 = np.einsum('ijc,jkc,kic->ijk', U4_3d, V4_3d, W4_3d)
    assert np.max(np.abs(T4 - 1.0)) < 1e-10, f"4x4x4 Strassen failed: {np.max(np.abs(T4 - 1.0))}"
    
    # Now construct the full 4x5x5x6 decomposition
    # Part 1: Strassen for A[0:4,0:4] * B[0:4,0:4] -> C[0:4,0:4] (49 terms)
    # Part 2: Schoolbook for A[0:4,0:4] * B[0:4,4:6] -> C[0:4,4:6] (32 terms)
    # Part 3: Schoolbook for A[0:4,4:5] * B[4:5,0:6] -> C[0:4,0:6] (24 terms)
    
    total_rank = 49 + 32 + 24  # = 105
    
    U_full = np.zeros((nm, total_rank))
    V_full = np.zeros((mp, total_rank))
    W_full = np.zeros((pn, total_rank))
    
    col = 0
    
    # Part 1: Strassen 4x4x4
    # U rows: i*5+j for i in 0..3, j in 0..3
    # V rows: j*6+k for j in 0..3, k in 0..3
    # W rows: k*4+i for k in 0..3, i in 0..3
    for c in range(49):
        for i in range(4):
            for j in range(4):
                U_full[i * 5 + j, col + c] = U4[i * 4 + j, c]
        for j in range(4):
            for k in range(4):
                V_full[j * 6 + k, col + c] = V4[j * 4 + k, c]
        for k in range(4):
            for i in range(4):
                W_full[k * 4 + i, col + c] = W4[k * 4 + i, c]
    col += 49
    
    # Part 2: Schoolbook 4x4x2 (A[0:4,0:4] * B[0:4,4:6] -> C[0:4,4:6])
    # U rows: i*5+j for i in 0..3, j in 0..3
    # V rows: j*6+k for j in 0..3, k in 4..5
    # W rows: k*4+i for k in 4..5, i in 0..3
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U_full[i * 5 + j, col] = 1
                V_full[j * 6 + k, col] = 1
                W_full[k * 4 + i, col] = 1
                col += 1
    
    # Part 3: Schoolbook 4x1x6 (A[0:4,4:5] * B[4:5,0:6] -> C[0:4,0:6])
    # U rows: i*5+4 for i in 0..3
    # V rows: 4*6+k for k in 0..5
    # W rows: k*4+i for k in 0..5, i in 0..3
    for i in range(4):
        for k in range(6):
            U_full[i * 5 + 4, col] = 1
            V_full[4 * 6 + k, col] = 1
            W_full[k * 4 + i, col] = 1
            col += 1
    
    assert col == total_rank, f"Column mismatch: {col} vs {total_rank}"
    
    # Verify the full decomposition
    err = check_error(U_full, V_full, W_full)
    if err < 1e-9:
        best_rank = total_rank
        best_result = {
            "U": U_full.tolist(),
            "V": V_full.tolist(),
            "W": W_full.tolist()
        }
        return best_result
    else:
        # Fallback to schoolbook
        print(f"Construction error: {err}")
        return best_result
    
# EVOLVE-BLOCK-END
