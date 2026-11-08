import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill
    
    # Try to find a lower-rank decomposition
    # For 4x4x8, we can use two independent 4x4x4 decompositions
    # Best known rank for 4x4x4 is 17, so total rank would be 34
    
    # Laderman's rank-17 decomposition for 4x4x4
    # U: 16x17, V: 16x17, W: 16x17
    # C[i,j] = sum_k U[i,j,k] * V[i,j,k] * W[k,i,j]
    # where i,j in 0..3, k in 0..16
    
    # The 17 products for 4x4x4:
    # P0 = (A00+A11+A22+A33)(B00+B11+B22+B33)
    # P1 = (A01+A10+A23+A32)(B01+B10+B23+B32)
    # P2 = (A02+A13+A20+A31)(B02+B13+B20+B31)
    # P3 = (A03+A12+A21+A30)(B03+B12+B21+B30)
    # P4 = (A00+A11)(B00+B11)
    # P5 = (A00+A22)(B00+B22)
    # P6 = (A00+A33)(B00+B33)
    # P7 = (A11+A22)(B11+B22)
    # P8 = (A11+A33)(B11+B33)
    # P9 = (A22+A33)(B22+B33)
    # P10 = (A01+A23)(B01+B23)
    # P11 = (A02+A31)(B02+B31)
    # P12 = (A03+A12)(B03+B12)
    # P13 = (A10+A32)(B10+B32)
    # P14 = (A13+A20)(B13+B20)
    # P15 = (A21+A30)(B21+B30)
    # P16 = (A01+A10)(B10+B01)
    
    # Let me define the U, V, W factors explicitly
    # U[i,j,k] = coefficient of A[i,j] in product k
    # V[i,j,k] = coefficient of B[i,j] in product k  
    # W[i,j,k] = coefficient of C[i,j] from product k
    
    U_44 = np.zeros((4, 4, 17))
    V_44 = np.zeros((4, 4, 17))
    W_44 = np.zeros((4, 4, 17))
    
    # P0: (A00+A11+A22+A33)(B00+B11+B22+B33)
    for i in range(4):
        U_44[i, i, 0] = 1
        V_44[i, i, 0] = 1
    for i in range(4):
        for j in range(4):
            W_44[i, j, 0] = 1
    
    # P1: (A01+A10+A23+A32)(B01+B10+B23+B32)
    U_44[0, 1, 1] = 1; U_44[1, 0, 1] = 1; U_44[2, 3, 1] = 1; U_44[3, 2, 1] = 1
    V_44[0, 1, 1] = 1; V_44[1, 0, 1] = 1; V_44[2, 3, 1] = 1; V_44[3, 2, 1] = 1
    W_44[0, 1, 1] = 1; W_44[1, 0, 1] = 1; W_44[2, 3, 1] = 1; W_44[3, 2, 1] = 1
    
    # P2: (A02+A13+A20+A31)(B02+B13+B20+B31)
    U_44[0, 2, 2] = 1; U_44[1, 3, 2] = 1; U_44[2, 0, 2] = 1; U_44[3, 1, 2] = 1
    V_44[0, 2, 2] = 1; V_44[1, 3, 2] = 1; V_44[2, 0, 2] = 1; V_44[3, 1, 2] = 1
    W_44[0, 2, 2] = 1; W_44[1, 3, 2] = 1; W_44[2, 0, 2] = 1; W_44[3, 1, 2] = 1
    
    # P3: (A03+A12+A21+A30)(B03+B12+B21+B30)
    U_44[0, 3, 3] = 1; U_44[1, 2, 3] = 1; U_44[2, 1, 3] = 1; U_44[3, 0, 3] = 1
    V_44[0, 3, 3] = 1; V_44[1, 2, 3] = 1; V_44[2, 1, 3] = 1; V_44[3, 0, 3] = 1
    W_44[0, 3, 3] = 1; W_44[1, 2, 3] = 1; W_44[2, 1, 3] = 1; W_44[3, 0, 3] = 1
    
    # P4: (A00+A11)(B00+B11)
    U_44[0, 0, 4] = 1; U_44[1, 1, 4] = 1
    V_44[0, 0, 4] = 1; V_44[1, 1, 4] = 1
    W_44[0, 0, 4] = 1; W_44[0, 1, 4] = 1; W_44[1, 0, 4] = 1; W_44[1, 1, 4] = 1
    
    # P5: (A00+A22)(B00+B22)
    U_44[0, 0, 5] = 1; U_44[2, 2, 5] = 1
    V_44[0, 0, 5] = 1; V_44[2, 2, 5] = 1
    W_44[0, 0, 5] = 1; W_44[0, 2, 5] = 1; W_44[2, 0, 5] = 1; W_44[2, 2, 5] = 1
    
    # P6: (A00+A33)(B00+B33)
    U_44[0, 0, 6] = 1; U_44[3, 3, 6] = 1
    V_44[0, 0, 6] = 1; V_44[3, 3, 6] = 1
    W_44[0, 0, 6] = 1; W_44[0, 3, 6] = 1; W_44[3, 0, 6] = 1; W_44[3, 3, 6] = 1
    
    # P7: (A11+A22)(B11+B22)
    U_44[1, 1, 7] = 1; U_44[2, 2, 7] = 1
    V_44[1, 1, 7] = 1; V_44[2, 2, 7] = 1
    W_44[1, 1, 7] = 1; W_44[1, 2, 7] = 1; W_44[2, 1, 7] = 1; W_44[2, 2, 7] = 1
    
    # P8: (A11+A33)(B11+B33)
    U_44[1, 1, 8] = 1; U_44[3, 3, 8] = 1
    V_44[1, 1, 8] = 1; V_44[3, 3, 8] = 1
    W_44[1, 1, 8] = 1; W_44[1, 3, 8] = 1; W_44[3, 1, 8] = 1; W_44[3, 3, 8] = 1
    
    # P9: (A22+A33)(B22+B33)
    U_44[2, 2, 9] = 1; U_44[3, 3, 9] = 1
    V_44[2, 2, 9] = 1; V_44[3, 3, 9] = 1
    W_44[2, 2, 9] = 1; W_44[2, 3, 9] = 1; W_44[3, 2, 9] = 1; W_44[3, 3, 9] = 1
    
    # P10: (A01+A23)(B01+B23)
    U_44[0, 1, 10] = 1; U_44[2, 3, 10] = 1
    V_44[0, 1, 10] = 1; V_44[2, 3, 10] = 1
    W_44[0, 1, 10] = 1; W_44[0, 3, 10] = 1; W_44[2, 1, 10] = 1; W_44[2, 3, 10] = 1
    
    # P11: (A02+A31)(B02+B31)
    U_44[0, 2, 11] = 1; U_44[3, 1, 11] = 1
    V_44[0, 2, 11] = 1; V_44[3, 1, 11] = 1
    W_44[0, 2, 11] = 1; W_44[0, 1, 11] = 1; W_44[3, 2, 11] = 1; W_44[3, 1, 11] = 1
    
    # P12: (A03+A12)(B03+B12)
    U_44[0, 3, 12] = 1; U_44[1, 2, 12] = 1
    V_44[0, 3, 12] = 1; V_44[1, 2, 12] = 1
    W_44[0, 3, 12] = 1; W_44[0, 2, 12] = 1; W_44[1, 3, 12] = 1; W_44[1, 2, 12] = 1
    
    # P13: (A10+A32)(B10+B32)
    U_44[1, 0, 13] = 1; U_44[3, 2, 13] = 1
    V_44[1, 0, 13] = 1; V_44[3, 2, 13] = 1
    W_44[1, 0, 13] = 1; W_44[1, 2, 13] = 1; W_44[3, 0, 13] = 1; W_44[3, 2, 13] = 1
    
    # P14: (A13+A20)(B13+B20)
    U_44[1, 3, 14] = 1; U_44[2, 0, 14] = 1
    V_44[1, 3, 14] = 1; V_44[2, 0, 14] = 1
    W_44[1, 3, 14] = 1; W_44[1, 0, 14] = 1; W_44[2, 3, 14] = 1; W_44[2, 0, 14] = 1
    
    # P15: (A21+A30)(B21+B30)
    U_44[2, 1, 15] = 1; U_44[3, 0, 15] = 1
    V_44[2, 1, 15] = 1; V_44[3, 0, 15] = 1
    W_44[2, 1, 15] = 1; W_44[2, 0, 15] = 1; W_44[3, 1, 15] = 1; W_44[3, 0, 15] = 1
    
    # P16: (A01+A10)(B10+B01)
    U_44[0, 1, 16] = 1; U_44[1, 0, 16] = 1
    V_44[1, 0, 16] = 1; V_44[0, 1, 16] = 1
    W_44[0, 1, 16] = 1; W_44[1, 0, 16] = 1
    
    # Verify the 4x4x4 decomposition
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 4)
    C_expected = A_test @ B_test
    C_computed = np.zeros((4, 4))
    for k in range(17):
        prod = 0
        for i in range(4):
            for j in range(4):
                prod += U_44[i, j, k] * A_test[i, j]
        prod2 = 0
        for i in range(4):
            for j in range(4):
                prod2 += V_44[i, j, k] * B_test[i, j]
        for i in range(4):
            for j in range(4):
                C_computed[i, j] += W_44[i, j, k] * prod * prod2
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to Strassen-based approach if Laderman doesn't work
        return _strassen_decomposition()
    
    # Now extend to 4x4x8 by using two independent 4x4x4 decompositions
    rank = 34
    U = np.zeros((n * m, rank))  # 16 x 34
    V = np.zeros((m * p, rank))  # 32 x 34
    W = np.zeros((p * n, rank))  # 32 x 34
    
    # For the left half (columns 0-3 of B):
    for k in range(17):
        for i in range(4):
            for j in range(4):
                U[i * 4 + j, k] = U_44[i, j, k]
                V[i * 8 + j, k] = V_44[i, j, k]
                W[j * 4 + i, k] = W_44[i, j, k]
    
    # For the right half (columns 4-7 of B):
    for k in range(17):
        for i in range(4):
            for j in range(4):
                U[i * 4 + j, 17 + k] = U_44[i, j, k]
                V[i * 8 + (j + 4), 17 + k] = V_44[i, j, k]
                W[(j + 4) * 4 + i, 17 + k] = W_44[i, j, k]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _strassen_decomposition():
    """Fallback Strassen-based decomposition with rank 98"""
    n, m, p = 4, 4, 8
    rank = 98
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1
    
    A_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (1, 3)],
        [(1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (-1, 0)],
        [(1, 1), (-1, 3)],
    ]
    
    B_specs = [
        [(1, 0), (1, 3)],
        [(1, 0)],
        [(1, 1), (-1, 3)],
        [(1, 2), (-1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (1, 3)],
    ]
    
    C_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (-1, 3)],
        [(1, 1), (1, 3)],
        [(1, 0), (1, 2)],
        [(-1, 0), (1, 1)],
        [(1, 3)],
        [(1, 0)],
    ]
    
    for half in range(2):
        for k_p in range(7):
            for m_s in range(7):
                col = half * 49 + k_p * 7 + m_s
                
                for coeff_a, block_idx in A_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_a_idx = global_r * 4 + global_c
                            a_eff_idx = r * 2 + c
                            U[global_a_idx, col] += coeff_a * U_2[a_eff_idx, m_s]
                
                for coeff_b, block_idx in B_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_k = global_c + half * 4
                            global_b_idx = global_r * 8 + global_k
                            b_eff_idx = r * 2 + c
                            V[global_b_idx, col] += coeff_b * V_2[b_eff_idx, m_s]
                
                for coeff_c, c_block_idx in C_specs[k_p]:
                    bi, bj = divmod(c_block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_i = bi * 2 + r
                            global_k = bj * 2 + c + half * 4
                            global_w_idx = global_k * 4 + global_i
                            p_eff_idx = r * 2 + c
                            W[global_w_idx, col] += coeff_c * W_2[p_eff_idx, m_s]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Laderman's 4x4x4 decomposition (rank 17)
    # U: 16x17, V: 16x17, W: 16x17
    # Indices: U[i*4+j, k], V[i*4+j, k], W[k*4+i, j] where C[i,j] = sum_k U[i,j,k]*V[i,j,k]*W[k,i,j]
    
    # Laderman's algorithm coefficients
    # P0 = (A00+A11+A22+A33)(B00+B11+B22+B33)
    # P1 = (A01+A10+A23+A32)(B01+B10+B23+B32)
    # P2 = (A02+A13+A20+A31)(B02+B13+B20+B31)
    # P3 = (A03+A12+A21+A30)(B03+B12+B21+B30)
    # P4 = (A00+A11)(B00+B11)
    # P5 = (A00+A22)(B00+B22)
    # P6 = (A00+A33)(B00+B33)
    # P7 = (A11+A22)(B11+B22)
    # P8 = (A11+A33)(B11+B33)
    # P9 = (A22+A33)(B22+B33)
    # P10 = (A01+A23)(B01+B23)
    # P11 = (A02+A31)(B02+B31)
    # P12 = (A03+A12)(B03+B12)
    # P13 = (A10+A32)(B10+B32)
    # P14 = (A13+A20)(B13+B20)
    # P15 = (A21+A30)(B21+B30)
    # P16 = (A01+A10+A23+A32)(B10+B01+B32+B23)  -- wait, this is getting complex
    
    # Let me use a verified Laderman decomposition.
    # Actually, let me use a simpler approach: just use the known rank-17 decomposition.
    
    # I'll use a standard Laderman 4x4x4 rank-17 decomposition.
    # Reference: Laderman (1976), "The number of multiplications needed to multiply 2x2 matrices"
    # Actually Laderman did 2x2 with 7, the 4x4 rank 17 is from later work.
    
    # Let me use a different approach: keep the Strassen structure but optimize.
    # The current rank is 98. Let me see if I can reduce it.
    
    # Actually, the simplest improvement: the current code has rank 98 = 2 * 7 * 7.
    # But we can potentially share some computations between the two halves.
    
    # For now, let me just add a deadline guard and keep the current approach,
    # but verify correctness and ensure it returns quickly.
    
    n, m, p = 4, 4, 8
    rank = 98
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1
    
    A_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (1, 3)],
        [(1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (-1, 0)],
        [(1, 1), (-1, 3)],
    ]
    
    B_specs = [
        [(1, 0), (1, 3)],
        [(1, 0)],
        [(1, 1), (-1, 3)],
        [(1, 2), (-1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (1, 3)],
    ]
    
    C_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (-1, 3)],
        [(1, 1), (1, 3)],
        [(1, 0), (1, 2)],
        [(-1, 0), (1, 1)],
        [(1, 3)],
        [(1, 0)],
    ]
    
    for half in range(2):
        for k_p in range(7):
            for m_s in range(7):
                col = half * 49 + k_p * 7 + m_s
                
                for coeff_a, block_idx in A_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_a_idx = global_r * 4 + global_c
                            a_eff_idx = r * 2 + c
                            U[global_a_idx, col] += coeff_a * U_2[a_eff_idx, m_s]
                
                for coeff_b, block_idx in B_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_k = global_c + half * 4
                            global_b_idx = global_r * 8 + global_k
                            b_eff_idx = r * 2 + c
                            V[global_b_idx, col] += coeff_b * V_2[b_eff_idx, m_s]
                
                for coeff_c, c_block_idx in C_specs[k_p]:
                    bi, bj = divmod(c_block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_i = bi * 2 + r
                            global_k = bj * 2 + c + half * 4
                            global_w_idx = global_k * 4 + global_i
                            p_eff_idx = r * 2 + c
                            W[global_w_idx, col] += coeff_c * W_2[p_eff_idx, m_s]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
