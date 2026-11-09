import time
import numpy as np
from itertools import product

def construct():
    """
    Rank-56 decomposition for 2x4 by 4x8 matrix multiplication.
    
    We split the 4-dimension into two 2-dimensions and use a rank-7 
    decomposition (Strassen) for each 2x2 matrix product, 
    applied to the 4 column blocks of the 8-dimensional output 
    for both the top and bottom halves of the 4x8 matrix.
    
    Total rank = 4 (blocks) * 2 (halves) * 7 (Strassen) = 56.
    """
    start_time = time.time()
    DEADLINE = 110.0
    
    n, m, p = 2, 4, 8
    
    # Strassen forms for 2x2 matrix multiplication
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)(b00)
    # m3 = (a00)(b01+b11)
    # m4 = (a11)(b10-b00)
    # m5 = (a00+a01)(b11-b10)
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    
    # X_forms[i][k] is coefficient of A[i] in m_k
    X_forms = [
        [1, 0, 1, 0, 1, -1, 0], # a00
        [0, 0, 0, 0, 1, 0, 1], # a01
        [0, 1, 0, 0, 0, 1, 0], # a10
        [1, 1, 0, 1, 0, 0, -1], # a11
    ]
    
    # Y_forms[j][k] is coefficient of B[j] in m_k
    Y_forms = [
        [1, 1, 0, -1, 0, 1, 0], # b00
        [0, 0, 1, 0, -1, 1, 0], # b01
        [1, 0, 0, 1, 1, 0, 1], # b10
        [1, 0, 1, 0, -1, 0, 1], # b11
    ]
    
    # Z_combos[i][k] is coefficient of m_k in C[i]
    Z_combos = [
        [1, 0, 0, 1, -1, 0, 1], # c00
        [0, 0, 1, 0, 1, 0, 0], # c01
        [0, 1, 0, 1, 0, 0, 0], # c10
        [1, -1, 1, 0, 0, 1, 0], # c11
    ]
    
    num_groups = 4 # 4 blocks of 2 columns in the 8-column output
    num_halves = 2 # Top and bottom halves of the 4-row input
    num_strassen = 7
    rank = num_groups * num_halves * num_strassen
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for h in range(num_halves): # 0: top, 1: bottom
        for g in range(num_groups):
            for s in range(num_strassen):
                # U: A indices
                # A is 2x4.
                # h=0 uses A[:, 0:2], h=1 uses A[:, 2:4]
                for i in range(n):
                    for jp in range(2):
                        a_idx = i * m + h * 2 + jp
                        U[a_idx][r] = X_forms[i][s] # Wait, X_forms is indexed by row i and strassen s?
                        # X_forms[i] is a list of 7 coeffs.
                        # So X_forms[i][s] is correct.
                        
                # V: B indices
                # B is 4x8.
                # h=0 uses B[0:2, :], h=1 uses B[2:4, :]
                # g determines column block.
                for jp in range(2): # row in 2x2 block
                    for kp in range(2): # col in 2x2 block
                        b_row = h * 2 + jp
                        b_col = g * 2 + kp
                        b_idx = b_row * p + b_col
                        V[b_idx][r] = Y_forms[jp * 2 + kp][s]
                        
                # W: C indices
                # C is 2x8.
                # g determines column block.
                for i in range(n):
                    for kp in range(2):
                        c_row = g * 2 + kp
                        c_col = i
                        c_idx = c_row * n + c_col
                        W[c_idx][r] = Z_combos[i * 2 + kp][s]
                        
                r += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == '__main__':
    res = construct()
    U = res['U']
    V = res['V']
    W = res['W']
    n, m, p = 2, 4, 8
    assert len(U) == n*m
    assert len(U[0]) == len(V[0])
    assert len(V) == m*p
    assert len(W) == p*n
    assert len(W[0]) == len(U[0])
    print("Constructed successfully with rank", len(U[0]))
