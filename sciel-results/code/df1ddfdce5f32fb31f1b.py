"""Program for matrix multiplication tensor decomposition (2x2 by 2x10)."""
import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition
    for multiplying a (2x2) matrix by a (2x10) matrix.
    
    Uses Strassen's algorithm for 2x2 matrix multiplication (rank 7)
    and extends it to 5 blocks of 2 columns each.
    Total rank = 5 * 7 = 35.
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank = 7 * num_blocks
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition factors for 2x2 case
    # U vectors: coeffs for A[00], A[01], A[10], A[11]
    U_2x2 = [
        [1, 0, 0, 1], # M1: A00 + A11
        [0, 0, 1, 1], # M2: A10 + A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00 + A01
        [-1, 0, 1, 0],# M6: A10 - A00
        [0, 1, 0, -1] # M7: A01 - A11
    ]
    
    # V vectors: coeffs for B[00], B[01], B[10], B[11]
    V_2x2 = [
        [1, 0, 0, 1], # M1: B00 + B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01 - B11
        [-1, 0, 1, 0],# M4: B10 - B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00 + B01
        [0, 0, 1, 1]  # M7: B10 + B11
    ]
    
    # W vectors: coeffs for C[00], C[10], C[01], C[11]
    # Note: W index in 2x2 case is k*2+i.
    # k=0,i=0 -> 0 (C00)
    # k=0,i=1 -> 1 (C10)
    # k=1,i=0 -> 2 (C01)
    # k=1,i=1 -> 3 (C11)
    W_2x2 = [
        [1, 0, 0, 1], # M1: C00 + C11
        [0, 1, 0, -1],# M2: C10 - C11
        [0, 0, 1, 1], # M3: C01 + C11
        [1, 1, 0, 0], # M4: C00 + C10
        [-1, 0, 1, 0],# M5: -C00 + C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    
    col = 0
    for t in range(num_blocks):
        for c in range(7):
            # Fill U
            for i in range(4):
                U[i][col] = U_2x2[c][i]
                
            # Fill V
            # V index: j*p + k
            # Block t corresponds to columns 2t and 2t+1 of B
            # j=0, k=2t -> idx 0*10 + 2t = 2t
            # j=0, k=2t+1 -> idx 0*10 + 2t+1 = 2t+1
            # j=1, k=2t -> idx 1*10 + 2t = 10+2t
            # j=1, k=2t+1 -> idx 1*10 + 2t+1 = 11+2t
            
            v_vals = V_2x2[c]
            # v_vals[0] is B00, v_vals[1] is B01, v_vals[2] is B10, v_vals[3] is B11
            
            V[2*t][col] = v_vals[0]
            V[2*t + 1][col] = v_vals[1]
            V[10 + 2*t][col] = v_vals[2]
            V[11 + 2*t][col] = v_vals[3]
            
            # Fill W
            # W index: k*n + i
            # Block t corresponds to output columns 2t and 2t+1
            # k=2t, i=0 -> idx 2t*2 + 0 = 4t
            # k=2t, i=1 -> idx 2t*2 + 1 = 4t+1
            # k=2t+1, i=0 -> idx (2t+1)*2 + 0 = 4t+2
            # k=2t+1, i=1 -> idx (2t+1)*2 + 1 = 4t+3
            
            w_vals = W_2x2[c]
            # w_vals[0] is C00, w_vals[1] is C10, w_vals[2] is C01, w_vals[3] is C11
            
            W[4*t][col] = w_vals[0]
            W[4*t + 1][col] = w_vals[1]
            W[4*t + 2][col] = w_vals[2]
            W[4*t + 3][col] = w_vals[3]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
