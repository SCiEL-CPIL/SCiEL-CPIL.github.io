import time
import random

DIMENSIONS = [4, 4, 6]

# Strassen 2x2 decomposition: 7 terms
# Each term: (u_coeffs, v_coeffs, w_coeffs) for 2x2 blocks
# u: [A00, A01, A10, A11], v: [B00, B01, B10, B11], w: [C00, C01, C10, C11]
STRASSEN_2X2 = [
    ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1 = (A00+A11)(B00+B11)
    ([0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 1, 0]),   # M2 = (A10+A11)B11
    ([1, 0, 0, 0], [1, 0, -1, 0], [0, 1, 0, 0]),  # M3 = A00(B00-B10)
    ([0, 0, 0, 1], [0, -1, 0, 1], [1, 0, 0, 0]),  # M4 = A11(B11-B01)
    ([1, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]),   # M5 = (A00+A01)B10
    ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6 = (A10-A00)(B00+B01)
    ([0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0]),  # M7 = (A01-A11)(B10+B11)
]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    nU = n * m  # 16
    nV = m * p  # 24
    nW = p * n  # 24
    
    rank = 84
    U = [[0] * rank for _ in range(nU)]
    V = [[0] * rank for _ in range(nV)]
    W = [[0] * rank for _ in range(nW)]
    
    col = 0
    for u, v, w in STRASSEN_2X2:
        for i in range(4):
            if u[i] == 0:
                continue
            a_row_start = 0 if i < 2 else 2
            a_col_start = 0 if i % 2 == 0 else 2
            
            for j in range(4):
                if v[j] == 0:
                    continue
                b_row_start = 0 if j < 2 else 2
                b_col_start = 0 if j % 2 == 0 else 3
                
                for k in range(4):
                    if w[k] == 0:
                        continue
                    c_row_start = 0 if k < 2 else 2
                    c_col_start = 0 if k % 2 == 0 else 3
                    
                    for ar in range(2):
                        for ac in range(2):
                            if ac == 0:
                                # A[ar, 0] * B[0, bc] -> C[ar, bc]
                                pass
                            if ac == 1:
                                # A[ar, 1] * B[1, bc] -> C[ar, bc]
                                pass
                            
                            for bc in range(3):
                                # Only contribute if ac == br, where br is the B row
                                # A element (ar, ac) contributes to C[ar, bc] via B[ac, bc]
                                br = ac
                                b_row = b_row_start + br
                                b_col = b_col_start + bc
                                a_row = a_row_start + ar
                                a_col = a_col_start + ac
                                c_row = c_row_start + ar
                                c_col = c_col_start + bc
                                
                                a_idx = a_row * m + a_col
                                b_idx = b_row * p + b_col
                                w_idx = c_col * n + c_row
                                
                                U[a_idx][col] += u[i]
                                V[b_idx][col] += v[j]
                                W[w_idx][col] += w[k]
        
        col += 1
    
    # Verify we used all columns
    if col != rank:
        # Truncate or pad
        pass
    
    return {"U": U, "V": V, "W": W}
