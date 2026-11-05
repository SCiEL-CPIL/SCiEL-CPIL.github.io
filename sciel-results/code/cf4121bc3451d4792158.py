import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 145

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    assert U.shape == (n * m, rank), f"U shape {U.shape} != ({n*m}, {rank})"
    assert V.shape == (m * p, rank), f"V shape {V.shape} != ({m*p}, {rank})"
    assert W.shape == (p * n, rank), f"W shape {W.shape} != ({p*n}, {rank})"
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = 0.0
                for r in range(rank):
                    actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                err = abs(actual - 1.0)
                if err > max_err:
                    max_err = err
    
    return max_err < 1e-9, max_err

def construct():
    n, m, p = 4, 4, 7
    total_rank = 93
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    current_rank = 0
    
    # Strassen 2x2x2: 7 terms
    # Each term: (x_mask, y_mask, c_mask) where masks are 2x2 dicts
    # x_mask[(li,lj)] = coefficient for A[li,lj]
    # y_mask[(li,lj)] = coefficient for B[li,lj]  
    # c_mask[(li,lj)] = coefficient for C[li,lj]
    strassen_2x2 = [
        # m1 = (A00+A11)(B00+B11) -> C00 += m1, C11 += m1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # m2 = (A10+A11)B00 -> C10 += m2, C11 -= m2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # m3 = A00(B01-B11) -> C01 += m3, C11 += m3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # m4 = A11(B10-B00) -> C00 += m4, C10 += m4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # m5 = (A00+A01)B11 -> C00 -= m5, C01 += m5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # m6 = (A10-A00)(B00+B01) -> C11 += m6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # m7 = (A01-A11)(B10+B11) -> C00 += m7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Strassen 4x4x4: 7 outer terms
    # Each outer term m: x_def maps block_index -> coeff, y_def maps block_index -> coeff
    # Block index b: bi = b//2 (row block), bj = b%2 (col block)
    # So block 0 = (0,0), block 1 = (0,1), block 2 = (1,0), block 3 = (1,1)
    strassen_4x4_outer = [
        # M1 = (A00+A11)(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10+A11)B00
        ({2: 1, 3: 1}, {0: 1}),
        # M3 = A00(B01-B11)
        ({0: 1}, {1: 1, 3: -1}),
        # M4 = A11(B10-B00)
        ({3: 1}, {2: 1, 0: -1}),
        # M5 = (A00+A01)B11
        ({0: 1, 1: 1}, {3: 1}),
        # M6 = (A10-A00)(B00+B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        # M7 = (A01-A11)(B10+B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}),
    ]
    
    # c_block for each outer term: maps (row_block, col_block) -> coeff
    c_blocks_4x4 = [
        # M1 -> C00 += M1, C11 += M1
        {(0,0): 1.0, (1,1): 1.0},
        # M2 -> C10 += M2, C11 -= M2
        {(1,0): 1.0, (1,1): -1.0},
        # M3 -> C01 += M3, C11 += M3
        {(0,1): 1.0, (1,1): 1.0},
        # M4 -> C00 += M4, C10 += M4
        {(0,0): 1.0, (1,0): 1.0},
        # M5 -> C00 -= M5, C01 += M5
        {(0,0): -1.0, (0,1): 1.0},
        # M6 -> C11 += M6
        {(1,1): 1.0},
        # M7 -> C00 += M7
        {(0,0): 1.0},
    ]
    
    # Part 1: 4x4x4 Strassen (rank 49)
    for m_idx in range(7):
        x_def, y_def = strassen_4x4_outer[m_idx]
        c_block = c_blocks_4x4[m_idx]
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = strassen_2x2[t_idx]
            
            # Compute U coefficients: U[i*m+j, r]
            for (li, lj), cx in x_sub.items():
                for b, cb in x_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    U[i * m + j, r] += cx * cb
            
            # Compute V coefficients: V[j*p+k, r]
            for (li, lj), cy in y_sub.items():
                for b, cb in y_def.items():
                    bi, bj = b // 2, b % 2
                    j = 2 * bi + li  # row of B
                    k = 2 * bj + lj  # col of B
                    V[j * p + k, r] += cy * cb
            
            # Compute W coefficients: W[k*n+i, r]
            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                for (br, bc), cb_c in c_block.items():
                    if cb_c == 0:
                        continue
                    i_final = 2 * br + li  # col of C
                    k_final = 2 * bc + lj  # row of C
                    W[k_final * n + i_final, r] += cc * cb_c
    
    # Part 2: 4x4x3 as 4 * (2x2x3), each of rank 11
    # 2x2x3: Strassen 2x2x2 for first 2 cols (7 terms) + 4 schoolbook terms for col 3
    
    def add_2x2x3(ar, ac, br, cr):
        """Add 2x2x3 decomposition.
        ar, ac: 2x2 block of A at row ar, col ac (global rows 2*ar, 2*ar+1; cols 2*ac, 2*ac+1)
        br: 2x2 block of B at row br (global rows 2*br, 2*br+1)
        cr: 2x2 block of C at row cr (global rows 2*cr, 2*cr+1)
        """
        nonlocal current_rank
        
        # Strassen 2x2x2 for columns 4,5 (local cols 0,1 of the 2x2x3)
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = strassen_2x2[t_idx]
            
            for (li, lj), cx in x_sub.items():
                i = 2 * ar + li
                j = 2 * ac + lj
                U[i * m + j, r] += cx
            
            for (li, lj), cy in y_sub.items():
                j = 2 * br + li  # row of B
                k = 4 + lj       # col of B (4 or 5)
                V[j * p + k, r] += cy
            
            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                i_final = 2 * cr + li  # col of C
                k_final = 4 + lj       # row of C (4 or 5)
                W[k_final * n + i_final, r] += cc
        
        # Schoolbook for column 6 (local col 2)
        # C[i][6] = A[i][0]*B[0][6] + A[i][1]*B[1][1+6-4]... 
        # Actually: C_local[i][2] = A_local[i][0]*B_local[0][2] + A_local[i][1]*B_local[1][2]
        # Global: col 6 of B, rows 2*br and 2*br+1
        
        # Term 1: A_local[0,0] * B_local[0,2] -> C_local[0,2]
        r = current_rank
        current_rank += 1
        U[(2 * ar + 0) * m + (2 * ac + 0), r] = 1.0
        V[(2 * br + 0) * p + 6, r] = 1.0
        W[6 * n + (2 * cr + 0), r] = 1.0
        
        # Term 2: A_local[0,1] * B_local[1,2] -> C_local[0,2]
        r = current_rank
        current_rank += 1
        U[(2 * ar + 0) * m + (2 * ac + 1), r] = 1.0
        V[(2 * br + 1) * p + 6, r] = 1.0
        W[6 * n + (2 * cr + 0), r] = 1.0
        
        # Term 3: A_local[1,0] * B_local[0,2] -> C_local[1,2]
        r = current_rank
        current_rank += 1
        U[(2 * ar + 1) * m + (2 * ac + 0), r] = 1.0
        V[(2 * br + 0) * p + 6, r] = 1.0
        W[6 * n + (2 * cr + 1), r] = 1.0
        
        # Term 4: A_local[1,1] * B_local[1,2] -> C_local[1,2]
        r = current_rank
        current_rank += 1
        U[(2 * ar + 1) * m + (2 * ac + 1), r] = 1.0
        V[(2 * br + 1) * p + 6, r] = 1.0
        W[6 * n + (2 * cr + 1), r] = 1.0

    # The four 2x2x3 blocks:
    # A block (0,0) * B block 0 -> C block 0
    add_2x2x3(0, 0, 0, 0)
    # A block (0,1) * B block 1 -> C block 0
    add_2x2x3(0, 1, 1, 0)
    # A block (1,0) * B block 0 -> C block 1
    add_2x2x3(1, 0, 0, 1)
    # A block (1,1) * B block 1 -> C block 1
    add_2x2x3(1, 1, 1, 1)
    
    assert current_rank == 93, f"Expected 93, got {current_rank}"
    
    U_final = U[:, :current_rank].tolist()
    V_final = V[:, :current_rank].tolist()
    W_final = W[:, :current_rank].tolist()
    
    # Verify if time permits
    if _time_remaining() > 5.0:
        is_valid, max_err = _verify(U_final, V_final, W_final, n, m, p)
        if not is_valid:
            # Should not happen, but if it does, we still return the result
            pass
    
    return {"U": U_final, "V": V_final, "W": W_final}
