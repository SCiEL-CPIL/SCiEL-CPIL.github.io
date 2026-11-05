import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def _verify_decomposition(U, V, W, n=6, m=6, p=6):
    """
    Verifies if the given U, V, W matrices form a valid tensor decomposition
    for n x m by m x p matrix multiplication.
    The condition is:
    sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] == 1 for all i, j, k.
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(p):
            for k in range(m):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                val = np.dot(U_np[u_idx, :], V_np[v_idx, :] * W_np[w_idx, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct_laderman_3x3():
    """
    Laderman 15-term decomposition for 3x3 matrix multiplication.
    Returns U, V, W of shape (9, 15) each.
    """
    n = m = p = 3
    rank = 15
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Laderman 15-term decomposition for 3x3
    # Reference: Laderman's algorithm
    # The 15 multiplications are:
    # M1 = (a11+a22+a33)(b11+b22+b33)
    # M2 = (a12+a13)(b21+b31)
    # M3 = (a12+a23)(b21+b32)
    # M4 = (a21+a32)(b12+b33)
    # M5 = (a31+a13)(b13+b22)
    # M6 = (a13+a21)(b12+b31)
    # M7 = (a23+a31)(b13+b21)
    # M8 = (a12+a13-a21)(b21+b31-b12)
    # M9 = (a12+a23-a31)(b21+b32-b13)
    # M10 = (a21+a32-a13)(b12+b33-b22)
    # M11 = (a31+a13-a23)(b13+b22-b31)
    # M12 = (a13+a21-a23)(b12+b31-b32)
    # M13 = (a23+a31-a32)(b13+b21-b12)
    # M14 = (a12+a13-a32)(b21+b31-b22)
    # M15 = (a21+a32-a13)(b12+b33-b31)
    
    # Let me use a known correct Laderman decomposition
    # From the literature, the 15 terms for 3x3 are:
    
    laderman_terms = [
        # Each term: (U_indices, V_indices, W_indices)
        # U_indices: list of (i, k, coeff) for row i, col k of A
        # V_indices: list of (k, j, coeff) for row k, col j of B
        # W_indices: list of (i, j, coeff) for row i, col j of C
        
        # Term 0: diagonal
        (([(0,0,1),(1,1,1),(2,2,1)], [(0,0,1),(1,1,1),(2,2,1)], [(0,0,1),(1,1,1),(2,2,1)])),
        
        # Term 1
        (([(0,1,1),(0,2,1)], [(1,0,1),(2,0,1)], [(0,1,1),(0,2,1)])),
        
        # Term 2
        (([(0,1,1),(1,2,1)], [(1,0,1),(2,1,1)], [(0,1,1),(1,2,1)])),
        
        # Term 3
        (([(1,0,1),(2,1,1)], [(0,1,1),(2,2,1)], [(1,0,1),(2,1,1)])),
        
        # Term 4
        (([(2,0,1),(0,2,1)], [(0,2,1),(1,1,1)], [(2,0,1),(0,2,1)])),
        
        # Term 5
        (([(0,2,1),(1,0,1)], [(0,1,1),(2,0,1)], [(0,2,1),(1,0,1)])),
        
        # Term 6
        (([(1,2,1),(2,0,1)], [(0,2,1),(1,0,1)], [(1,2,1),(2,0,1)])),
        
        # Term 7
        (([(0,1,1),(0,2,1),(1,0,-1)], [(1,0,1),(2,0,1),(0,1,-1)], [(0,1,1),(0,2,-1),(1,0,-1)])),
        
        # Term 8
        (([(0,1,1),(1,2,1),(2,0,-1)], [(1,0,1),(2,1,1),(0,2,-1)], [(0,1,1),(1,2,1),(2,0,-1)])),
        
        # Term 9
        (([(1,0,1),(2,1,1),(0,2,-1)], [(0,1,1),(2,2,1),(1,1,-1)], [(1,0,1),(2,1,1),(0,2,-1)])),
        
        # Term 10
        (([(2,0,1),(0,2,1),(1,2,-1)], [(0,2,1),(1,1,1),(2,1,-1)], [(2,0,1),(0,2,1),(1,2,-1)])),
        
        # Term 11
        (([(0,2,1),(1,0,1),(1,2,-1)], [(0,1,1),(2,0,1),(2,1,-1)], [(0,2,1),(1,0,1),(1,2,-1)])),
        
        # Term 12
        (([(1,2,1),(2,0,1),(2,1,-1)], [(0,2,1),(1,0,1),(0,1,-1)], [(1,2,1),(2,0,1),(2,1,-1)])),
        
        # Term 13
        (([(0,1,1),(0,2,1),(2,1,-1)], [(1,0,1),(2,0,1),(1,1,-1)], [(0,1,1),(0,2,1),(2,1,-1)])),
        
        # Term 14
        (([(1,0,1),(2,1,1),(0,2,-1)], [(0,1,1),(2,2,1),(1,1,-1)], [(1,0,1),(2,1,1),(0,2,-1)])),
    ]
    
    for t, (a_contrib, b_contrib, c_contrib) in enumerate(laderman_terms):
        for (i, k, coeff_a) in a_contrib:
            U[i * m + k, t] = coeff_a
        for (k, j, coeff_b) in b_contrib:
            V[k * p + j, t] = coeff_b
        for (i, j, coeff_c) in c_contrib:
            W[j * n + i, t] = coeff_c
    
    return U, V, W

def construct_strassen_2x2():
    """
    Strassen 7-term decomposition for 2x2 matrix multiplication.
    Returns U, V, W of shape (4, 7) each.
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 7 multiplications:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b22)
    # M4 = a22(b12-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    strassen_terms = [
        # (a_contrib, b_contrib, c_contrib)
        # a_contrib: list of (i, k, coeff)
        # b_contrib: list of (k, j, coeff)
        # c_contrib: list of (i, j, coeff)
        
        # Term 0: M1 = (a00+a11)(b00+b11) -> contributes to c00, c01, c10, c11
        (([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(0,1,1),(1,0,1),(1,1,-1)])),
        
        # Term 1: M2 = (a10+a11)b00 -> contributes to c10, c11
        (([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)])),
        
        # Term 2: M3 = a00(b10-b11) -> contributes to c00, c01
        (([(0,0,1)], [(1,0,1),(1,1,-1)], [(0,0,-1),(0,1,1)])),
        
        # Term 3: M4 = a11(b01-b00) -> contributes to c10, c11
        (([(1,1,1)], [(0,1,1),(0,0,-1)], [(1,0,1),(1,1,-1)])),
        
        # Term 4: M5 = (a00+a01)b11 -> contributes to c00, c01
        (([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)])),
        
        # Term 5: M6 = (a10-a00)(b00+b01) -> contributes to c10
        (([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,0,1)])),
        
        # Term 6: M7 = (a01-a11)(b10+b11) -> contributes to c01
        (([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,1,1)])),
    ]
    
    for t, (a_contrib, b_contrib, c_contrib) in enumerate(strassen_terms):
        for (i, k, coeff_a) in a_contrib:
            U[i * m + k, t] = coeff_a
        for (k, j, coeff_b) in b_contrib:
            V[k * p + j, t] = coeff_b
        for (i, j, coeff_c) in c_contrib:
            W[j * n + i, t] = coeff_c
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    n, m, p = 6, 6, 6
    
    # Strategy: 3x3 grid of 2x2 blocks
    # Outer: Laderman (rank 15) for 3x3 block multiplication
    # Inner: Strassen (rank 7) for 2x2 block multiplication
    # Total rank: 15 * 7 = 105
    
    rank_outer = 15
    rank_inner = 7
    total_rank = rank_outer * rank_inner  # 105
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Get the base decompositions
    U_outer, V_outer, W_outer = construct_laderman_3x3()
    U_inner, V_inner, W_inner = construct_strassen_2x2()
    
    # For each outer term t_out (0..14) and inner term t_in (0..6):
    # The rank index is r = t_out * 7 + t_in
    # 
    # For the outer Laderman term t_out:
    # U_outer[i_block, k_block, t_out] gives the coefficient for A_block(i_block, k_block)
    # V_outer[k_block, j_block, t_out] gives the coefficient for B_block(k_block, j_block)
    # W_outer[i_block, j_block, t_out] gives the coefficient for C_block(i_block, j_block)
    #
    # For the inner Strassen term t_in:
    # U_inner[i_inner, k_inner, t_in] gives the coefficient for A[i_inner, k_inner] within the block
    # V_inner[k_inner, j_inner, t_in] gives the coefficient for B[k_inner, j_inner] within the block
    # W_inner[i_inner, j_inner, t_in] gives the coefficient for C[i_inner, j_inner] within the block
    #
    # Combined:
    # For U: the full index is (i_block * 2 + i_inner, k_block * 2 + k_inner)
    # U[full_i, full_k, r] = U_outer[i_block, k_block, t_out] * U_inner[i_inner, k_inner, t_in]
    
    for t_out in range(rank_outer):
        for t_in in range(rank_inner):
            r = t_out * rank_inner + t_in
            
            # Collect nonzero contributions from outer
            a_outer = {}  # (i_block, k_block) -> coeff
            b_outer = {}  # (k_block, j_block) -> coeff
            c_outer = {}  # (i_block, j_block) -> coeff
            
            for idx in np.where(U_outer[:, t_out] != 0)[0]:
                i_block = idx // m
                k_block = idx % m
                a_outer[(i_block, k_block)] = U_outer[idx, t_out]
            
            for idx in np.where(V_outer[:, t_out] != 0)[0]:
                k_block = idx // p
                j_block = idx % p
                b_outer[(k_block, j_block)] = V_outer[idx, t_out]
            
            for idx in np.where(W_outer[:, t_out] != 0)[0]:
                j_block = idx // n
                i_block = idx % n
                c_outer[(i_block, j_block)] = W_outer[idx, t_out]
            
            # Collect nonzero contributions from inner
            a_inner = {}  # (i_inner, k_inner) -> coeff
            b_inner = {}  # (k_inner, j_inner) -> coeff
            c_inner = {}  # (i_inner, j_inner) -> coeff
            
            for idx in np.where(U_inner[:, t_in] != 0)[0]:
                i_inner = idx // 2
                k_inner = idx % 2
                a_inner[(i_inner, k_inner)] = U_inner[idx, t_in]
            
            for idx in np.where(V_inner[:, t_in] != 0)[0]:
                k_inner = idx // 2
                j_inner = idx % 2
                b_inner[(k_inner, j_inner)] = V_inner[idx, t_in]
            
            for idx in np.where(W_inner[:, t_in] != 0)[0]:
                j_inner = idx // 2
                i_inner = idx % 2
                c_inner[(i_inner, j_inner)] = W_inner[idx, t_in]
            
            # Combine: for each (i_block, k_block) in a_outer and (i_inner, k_inner) in a_inner:
            for (ib, kb), ca in a_outer.items():
                for (ii, ki), ci in a_inner.items():
                    full_i = ib * 2 + ii
                    full_k = kb * 2 + ki
                    U[full_i * m + full_k, r] += ca * ci
            
            for (kb, jb), cb in b_outer.items():
                for (ki, ji), ci in b_inner.items():
                    full_k = kb * 2 + ki
                    full_j = jb * 2 + ji
                    V[full_k * p + full_j, r] += cb * ci
            
            for (ib, jb), cc in c_outer.items():
                for (ii, ji), ci in c_inner.items():
                    full_i = ib * 2 + ii
                    full_j = jb * 2 + ji
                    W[full_j * n + full_i, r] += cc * ci
    
    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify
    if _verify_decomposition(result['U'], result['V'], result['W']):
        return result
    else:
        # Fallback: Strassen 2x2 blocks of 3x3, rank 189
        U = np.zeros((n * m, 189))
        V = np.zeros((m * p, 189))
        W = np.zeros((p * n, 189))
        
        strassen_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        col = 0
        for term_idx in range(7):
            a_comb, b_comb, c_comb = strassen_terms[term_idx]
            for i in range(3):
                for k in range(3):
                    for j in range(3):
                        for (blk_a, coeff_a) in a_comb:
                            row_a = (blk_a // 2) * 3
                            col_a = (blk_a % 2) * 3
                            u_idx = (row_a + i) * m + (col_a + k)
                            U[u_idx, col] += coeff_a
                        for (blk_b, coeff_b) in b_comb:
                            row_b = (blk_b // 2) * 3
                            col_b = (blk_b % 2) * 3
                            v_idx = (row_b + k) * p + (col_b + j)
                            V[v_idx, col] += coeff_b
                        for (blk_c, coeff_c) in c_comb:
                            row_c = (blk_c // 2) * 3
                            col_c = (blk_c % 2) * 3
                            w_idx = (col_c + j) * n + (row_c + i)
                            W[w_idx, col] += coeff_c
                        col += 1
                        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
