import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Strassen 2x2x2 factors
    # U: (2x2) -> (2x4) indices? No, U maps A indices to rank indices.
    # Shape of U: (n*m, rank) = (8, R)
    # Shape of V: (m*p, rank) = (32, R)
    # Shape of W: (p*n, rank) = (16, R)
    
    # Strassen products for 2x2 * 2x2 -> 2x2
    # U_coeffs: {(i,j): coeff} for A (2x2)
    # V_coeffs: {(j,k): coeff} for B (2x2)
    # W_coeffs: {(k,i): coeff} for C (2x2)
    
    strassen_products = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Total rank: 4 output col pairs * 2 input row pairs * 7 Strassen terms
    total_rank = 4 * 2 * 7
    R = total_rank
    
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    col_idx = 0
    for oc in range(4):  # output column pair 0..3 (cols 2*oc, 2*oc+1)
        for ac in range(2):  # input column pair 0..1 (cols 2*ac, 2*ac+1)
            # A sub-block: rows 0-1, cols 2*ac .. 2*ac+1
            # B sub-block: rows 2*ac .. 2*ac+1, cols 2*oc .. 2*oc+1
            a_col_offset = 2 * ac
            b_row_offset = 2 * ac
            b_col_offset = 2 * oc
            
            for u_coeffs, v_coeffs, w_coeffs in strassen_products:
                # Fill U
                for (li, lj), coeff in u_coeffs.items():
                    # Global A index: row li, col a_col_offset + lj
                    U[li * m + (a_col_offset + lj), col_idx] = coeff
                    
                # Fill V
                for (lj, lk), coeff in v_coeffs.items():
                    # Global B index: row b_row_offset + lj, col b_col_offset + lk
                    V[(b_row_offset + lj) * p + (b_col_offset + lk), col_idx] = coeff
                    
                # Fill W
                for (lk, li), coeff in w_coeffs.items():
                    # Global C index: col b_col_offset + lk, row li
                    # Trace convention: W[k*n + i, r] where C[i,k] = sum_r U[i,r] V[r,k] W[k,r] ?
                    # Task says: W[k*n+i, r]
                    # So index is (b_col_offset + lk) * n + li
                    W[(b_col_offset + lk) * n + li, col_idx] = coeff
                    
                col_idx += 1
                
                if time.time() > deadline:
                    # Should not happen
                    pass
                    
    return {"U": U, "V": V, "W": W}
