import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Total rank is 68
    rank = 68
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen blocks for A[0:2, 0:2] * B[0:2, k_block]
    # A[0:2, 0:2] is 2x2. B[0:2, :] is 2x8.
    # Split B[0:2, :] into 4 blocks of 2x2.
    # Block l corresponds to k in {2l, 2l+1}
    
    # Strassen multipliers for 2x2 * 2x2
    # x = A_block, y = B_block
    # M1 = (x00 + x11)(y00 + y11)
    # M2 = (x10 + x11)y00
    # M3 = x00(y01 - y11)
    # M4 = x11(y10 - y00)
    # M5 = (x00 + x01)y11
    # M6 = (x10 - x00)(y00 + y01)
    # M7 = (x01 - x11)(y10 + y11)
    
    # z00 = M1 + M4 - M5 + M7
    # z01 = M3 + M5
    # z10 = M2 + M4
    # z11 = M1 - M2 + M3 + M6
    
    # We define the coefficients for U, V, W for each of the 7 terms.
    # U indices: i*m + j. For 2x2 block, i in {0,1}, j in {0,1}.
    # V indices: j*p + k. For 2x2 block, j in {0,1}, k in {k_start, k_start+1}.
    # W indices: k*n + i. For 2x2 block, k in {k_start, k_start+1}, i in {0,1}.
    
    # Term definitions: (coeff_U, coeff_V, coeff_W)
    # Each coeff is a dict mapping (i,j) or (j,k) or (k,i) to value.
    
    def get_U_idx(i, j): return i * m + j
    def get_V_idx(j, k): return j * p + k
    def get_W_idx(k, i): return k * n + i
    
    # For a given Strassen term, we add contributions to U, V, W
    # The term is of the form sum_u U_u * sum_v V_v * sum_w W_w
    # Actually, each rank-1 component is a product of 3 vectors.
    # Here we are building the matrices column by column.
    # Each column c corresponds to one rank-1 term.
    
    # Strassen terms structure:
    # 1. (x00+x11) * (y00+y11)
    #    U: {00:1, 11:1}
    #    V: {00:1, 11:1}
    #    W: {00:1, 01:1, 10:1, 11:1} ? No, W depends on z.
    #    Wait, the decomposition is T = sum C_r U_r \otimes V_r \otimes W_r.
    #    For Strassen, we have 7 terms.
    #    Term 1: M1.
    #    U1: x00=1, x11=1.
    #    V1: y00=1, y11=1.
    #    W1: z00=1, z11=1? No.
    #    Let's trace:
    #    M1 contributes to z00 (+1), z11 (+1).
    #    So W1: z00=1, z11=1.
    
    # Let's list the 7 terms' U, V, W vectors (restricted to the 2x2 block indices).
    # Indices: 00, 01, 10, 11.
    
    # Term 1:
    # U: {00:1, 11:1}
    # V: {00:1, 11:1}
    # W: {00:1, 11:1}  # z00 gets +M1, z11 gets +M1
    
    # Term 2:
    # U: {10:1, 11:1}
    # V: {00:1}
    # W: {10:1} # z10 gets +M2
    
    # Term 3:
    # U: {00:1}
    # V: {01:1, 11:-1}
    # W: {01:1, 11:1} # z01 gets +M3, z11 gets +M3
    
    # Term 4:
    # U: {11:1}
    # V: {10:1, 00:-1}
    # W: {00:1, 10:1} # z00 gets +M4, z10 gets +M4
    
    # Term 5:
    # U: {00:1, 01:1}
    # V: {11:1}
    # W: {00:-1, 01:1} # z00 gets -M5, z01 gets +M5
    
    # Term 6:
    # U: {10:1, 00:-1}
    # V: {00:1, 01:1}
    # W: {11:1} # z11 gets +M6
    
    # Term 7:
    # U: {01:1, 11:-1}
    # V: {10:1, 11:1}
    # W: {00:1} # z00 gets +M7
    
    # Check z00: M1 + M4 - M5 + M7. Correct.
    # Check z01: M3 + M5. Correct.
    # Check z10: M2 + M4. Correct.
    # Check z11: M1 - M2 + M3 + M6. Correct.
    
    strassen_terms = [
        (
            {(0,0): 1, (1,1): 1},  # U
            {(0,0): 1, (1,1): 1},  # V
            {(0,0): 1, (1,1): 1}   # W
        ),
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1}
        ),
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1}
        ),
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1}
        ),
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1}
        ),
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        )
    ]
    
    for l in range(4):
        k_start = 2 * l
        for term in strassen_terms:
            U_dict, V_dict, W_dict = term
            
            # Fill U
            for (i, j), val in U_dict.items():
                U[get_U_idx(i, j)][col] += val
                
            # Fill V
            for (j, k_rel), val in V_dict.items():
                k_abs = k_start + k_rel
                V[get_V_idx(j, k_abs)][col] += val
                
            # Fill W
            for (k_rel, i), val in W_dict.items():
                k_abs = k_start + k_rel
                W[get_W_idx(k_abs, i)][col] += val
                
            col += 1
            
    # Now handle the remaining schoolbook parts
    # A[0:2, 2] * B[2, :]
    # i in {0,1}, j=2, k in {0..7}
    for i in range(2):
        for k in range(8):
            U[get_U_idx(i, 2)][col] = 1
            V[get_V_idx(2, k)][col] = 1
            W[get_W_idx(k, i)][col] = 1
            col += 1
            
    # A[2, 0:2] * B[0:2, :]
    # i=2, j in {0,1}, k in {0..7}
    for j in range(2):
        for k in range(8):
            U[get_U_idx(2, j)][col] = 1
            V[get_V_idx(j, k)][col] = 1
            W[get_W_idx(k, 2)][col] = 1
            col += 1
            
    # A[2, 2] * B[2, :]
    # i=2, j=2, k in {0..7}
    for k in range(8):
        U[get_U_idx(2, 2)][col] = 1
        V[get_V_idx(2, k)][col] = 1
        W[get_W_idx(k, 2)][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
