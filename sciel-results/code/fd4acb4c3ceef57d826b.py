import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 56
    rank = 56
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # Indices 0..6
    # M1..M7
    # X is 2x2, Y is 2x2
    # U part (X) and V part (Y) and W part (Z)
    
    # We define the 7 Strassen terms as (u_vec, v_vec, w_vec)
    # u_vec is 4 entries (flattened X)
    # v_vec is 4 entries (flattened Y)
    # w_vec is 4 entries (flattened Z)
    
    # Strassen terms definition
    # M1: (X11+X22)(Y11+Y22)
    # M2: (X21+X22)Y11
    # M3: X11(Y12-Y22)
    # M4: X22(Y21-Y11)
    # M5: (X11+X12)Y22
    # M6: (X21-X11)(Y11+Y12)
    # M7: (X12-X22)(Y21+Y22)
    
    # Flatten order:
    # X: [X11, X12, X21, X22]
    # Y: [Y11, Y12, Y21, Y22]
    # Z: [Z11, Z12, Z21, Z22]
    
    strassen_terms = [
        # u, v, w
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]), # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 0, 1]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 0]), # M5 (Note: Z11 has -M5, Z22 has 0? No. Z11 = M1+M4-M5+M7. Z22 = M1-M2+M3+M6. So M5 coeff in Z is [-1, 0, 0, 0])
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6 (Z22 has +M6)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # M7 (Z11 has +M7)
    ]
    
    # Wait, let's re-verify Strassen W coefficients.
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    # Corrected W vectors (indices 0=Z11, 1=Z12, 2=Z21, 3=Z22):
    # M1: [1, 0, 0, 1]
    # M2: [0, 0, 1, -1]
    # M3: [0, 1, 0, 1]
    # M4: [1, 0, 1, 0]
    # M5: [-1, 1, 0, 0]
    # M6: [0, 0, 0, 1]
    # M7: [1, 0, 0, 0]
    
    strassen_terms_corrected = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # M7
    ]
    
    # Mapping of indices
    # A is n x m (2 x 4). Flattened row-major: A[i, j] -> index i*m + j.
    # B is m x p (4 x 8). Flattened row-major: B[j, k] -> index j*p + k.
    # C is n x p (2 x 8). Output is W[k*n + i, col] according to trace convention?
    # "output factor follows the official trace convention W[k*n+i,r]"
    # This means W index is (k, i).
    
    # We partition j into j_L (0,1) and j_R (2,3).
    # We partition k into k_1 (0,1), k_2 (2,3), k_3 (4,5), k_4 (6,7).
    
    col = 0
    for k_group in range(4):
        # k_group 0 -> k in {0, 1}
        # k_group 1 -> k in {2, 3}
        # k_group 2 -> k in {4, 5}
        # k_group 3 -> k in {6, 7}
        k_base = k_group * 2
        
        # For each of the 14 Strassen terms (7 for j_L, 7 for j_R)
        for j_part in range(2):
            # j_part 0 -> j in {0, 1}
            # j_part 1 -> j in {2, 3}
            j_base = j_part * 2
            
            for s_idx in range(7):
                u_vec, v_vec, w_vec = strassen_terms_corrected[s_idx]
                
                # Fill U (A part)
                # u_vec corresponds to A[i, j] for i in 0..1, j in j_base..j_base+1
                # A index: i*m + j
                for i in range(2):
                    for jj in range(2):
                        a_idx = i * m + (j_base + jj)
                        u_val = u_vec[i * 2 + jj]
                        U[a_idx][col] = u_val
                        
                # Fill V (B part)
                # v_vec corresponds to B[j, k] for j in j_base..j_base+1, k in k_base..k_base+1
                # B index: j*p + k
                for jj in range(2):
                    for kk in range(2):
                        b_idx = (j_base + jj) * p + (k_base + kk)
                        v_val = v_vec[jj * 2 + kk]
                        V[b_idx][col] = v_val
                        
                # Fill W (C part)
                # w_vec corresponds to C[i, k] for i in 0..1, k in k_base..k_base+1
                # W index: k*n + i
                for i in range(2):
                    for kk in range(2):
                        w_idx = (k_base + kk) * n + i
                        w_val = w_vec[i * 2 + kk]
                        W[w_idx][col] = w_val
                        
                col += 1
                
    return {"U": U, "V": V, "W": W}
