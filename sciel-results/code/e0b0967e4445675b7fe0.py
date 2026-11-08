import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = 4, 4, 8
    
    # Build the rank-98 recursive Strassen decomposition
    # Then try to reduce rank by finding linear dependencies
    
    # Strassen 2x2*2x2: 7 terms, each is (u, v, w) in R^4
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # 4x4*4x4 Strassen: 7 M-N pairs, each decomposed with 7 sub-terms = 49 rank
    # M0=A00+A11, M1=A10+A11, M2=A00, M3=A11, M4=A00+A01, M5=A10-A00, M6=A01-A11
    # N0=B00+B11, N1=B00, N2=B01-B11, N3=B10-B00, N4=B11, N5=B00+B01, N6=B10+B11
    # C00=P0+P3-P4+P6, C01=P2+P4, C10=P1+P3, C11=P0-P1+P2+P5
    
    # For 4x4*4x8: two 4x4*4x4 multiplications, total rank 98
    
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # Block index mapping for 2x2 blocks within 4x4
    # Block (bi,bj) covers rows bi*2..bi*2+1, cols bj*2..bj*2+1
    # A block (bi,bj): A[bi*2:(bi+1)*2, bj*2:(bj+1)*2]
    # Flatten: A[bi*2:(bi+1)*2, bj*2:(bj+1)*2] -> indices (bi*2+r)*4 + (bj*2+c) for r,c in 0,1
    
    # M specs: for each of 7 M terms, which A blocks with what coefficients
    # M0 = A00 + A11
    # M1 = A10 + A11
    # M2 = A00
    # M3 = A11
    # M4 = A00 + A01
    # M5 = A10 - A00
    # M6 = A01 - A11
    M_specs = [
        [(1, (0,0)), (1, (1,1))],   # M0
        [(1, (1,0)), (1, (1,1))],   # M1
        [(1, (0,0))],                # M2
        [(1, (1,1))],                # M3
        [(1, (0,0)), (1, (0,1))],   # M4
        [(1, (1,0)), (-1, (0,0))],  # M5
        [(1, (0,1)), (-1, (1,1))],  # M6
    ]
    
    # N specs: for each of 7 N terms, which B blocks with what coefficients
    # N0 = B00+B11, N1=B00, N2=B01-B11, N3=B10-B00, N4=B11, N5=B00+B01, N6=B10+B11
    N_specs = [
        [(1, (0,0)), (1, (1,1))],   # N0
        [(1, (0,0))],                # N1
        [(1, (0,1)), (-1, (1,1))],  # N2
        [(1, (1,0)), (-1, (0,0))],  # N3
        [(1, (1,1))],                # N4
        [(1, (0,0)), (1, (0,1))],   # N5
        [(1, (1,0)), (1, (1,1))],   # N6
    ]
    
    # C specs: for each of 7 P terms, which C blocks with what coefficients
    # C00=P0+P3-P4+P6, C01=P2+P4, C10=P1+P3, C11=P0-P1+P2+P5
    C_specs = [
        [(1, (0,0)), (1, (1,1))],   # P0 -> C00(+1), C11(+1)
        [(1, (1,0)), (-1, (1,1))],  # P1 -> C10(+1), C11(-1)
        [(1, (0,1)), (1, (1,1))],   # P2 -> C01(+1), C11(+1)
        [(1, (0,0)), (1, (1,0))],   # P3 -> C00(+1), C10(+1)
        [(-1, (0,0)), (1, (0,1))],  # P4 -> C00(-1), C01(+1)
        [(1, (1,1))],                # P5 -> C11(+1)
        [(1, (0,0))],                # P6 -> C00(+1)
    ]
    
    for b_half in range(2):
        base_col = b_half * 49
        for m_idx in range(7):
            for s_idx in range(7):
                col = base_col + m_idx * 7 + s_idx
                u_sub, v_sub, w_sub = strassen_2x2[s_idx]
                
                # U part: combine M_specs[m_idx] with u_sub
                for coeff_a, (bi, bj) in M_specs[m_idx]:
                    for r in range(2):
                        for c in range(2):
                            a_idx = (bi * 2 + r) * 4 + (bj * 2 + c)
                            u_eff = r * 2 + c
                            U[a_idx, col] += coeff_a * u_sub[u_eff]
                
                # V part: combine N_specs[m_idx] with v_sub, offset by b_half
                for coeff_b, (bi, bj) in N_specs[m_idx]:
                    for r in range(2):
                        for c in range(2):
                            j_idx = bi * 2 + r
                            k_idx = bj * 2 + c + b_half * 4
                            v_idx = j_idx * 8 + k_idx
                            v_eff = r * 2 + c
                            V[v_idx, col] += coeff_b * v_sub[v_eff]
                
                # W part: combine C_specs[m_idx] with w_sub, offset by b_half
                for coeff_c, (bi, bj) in C_specs[m_idx]:
                    for r in range(2):
                        for c in range(2):
                            i_idx = bi * 2 + r
                            k_idx = bj * 2 + c + b_half * 4
                            w_idx = k_idx * 4 + i_idx
                            w_eff = r * 2 + c
                            W[w_idx, col] += coeff_c * w_sub[w_eff]
                
                if time.time() > deadline:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify the decomposition is correct
    # Check: sum_r U[:,r] * V[:,r] * W[:,r] should equal the multiplication tensor
    # For each (i,k,j): the tensor entry is 1 if i in [0,4), k in [0,4), j in [0,8)
    # U index: i*4+k, V index: k*8+j, W index: j*4+i
    
    # Try to reduce rank by finding linear dependencies
    # The 98 columns of U, V, W must satisfy the tensor equation
    # We can try to find a subset of columns that still spans the same tensor
    
    # Approach: Use the fact that the tensor has only 128 nonzero entries
    # Build a system where each nonzero tensor entry gives a constraint
    
    # Actually, let's try a simpler approach: check if any column is a linear combination
    # of others in all three matrices simultaneously
    
    # For rank reduction, we need to find a vector x in R^98 such that:
    # U*x = 0, V*x = 0, W*x = 0 (all three null spaces intersect non-trivially)
    # But that would mean the column is redundant in all three, which is unlikely
    
    # Better: try to find a lower-rank decomposition from scratch using the tensor structure
    
    # The multiplication tensor T[i*4+k, k*8+j, j*4+i] = 1
    # This is a 16 x 32 x 32 tensor with 128 ones
    
    # Known results: rank of 4x4*4x4 is at most 49 (Strassen), conjectured to be 42-44
    # For 4x4*4x8, the rank is at most 98, but could be lower
    
    # Let's try to find a rank-84 or lower decomposition
    # Strategy: use the structure more cleverly
    
    # Alternative: try to merge the two 49-rank blocks
    # The key is that both halves use the same A matrix
    # So there might be shared structure
    
    # Let's try: for each of the 7 M terms and 7 sub-terms, we have two columns
    # (one for left half, one for right half). Can we combine them?
    
    # Column (m_idx, s_idx, half) has:
    # U: same for both halves (depends only on M_specs[m_idx] and u_sub)
    # V: differs (N_specs[m_idx] with different k offsets)
    # W: differs (C_specs[m_idx] with different k offsets)
    
    # Since U is the same for both halves, we might be able to find linear combinations
    # where the U parts cancel or combine
    
    # For a given (m_idx, s_idx), let U_col be the U column (same for both halves)
    # V_left, V_right, W_left, W_right are the V and W columns
    
    # We need: sum over all 98 cols of U_c * V_c * W_c = T
    
    # This is hard to reduce directly. Let's just return the rank-98 solution.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
