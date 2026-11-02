import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    # Start time for deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10 seconds safety margin
    
    # The minimum rank for M_{2,2,8} is 28.
    # This is achieved by applying Strassen's algorithm (rank 7) to each of the 4 
    # independent 2x2 blocks of the multiplication.
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen terms for a 2x2 block multiplication
    # Each term is (u_coeff, v_coeff, w_coeff)
    # u_coeff: {(i, j): coeff} for A[i, j]
    # v_coeff: {(j, k): coeff} for B[j, k]
    # w_coeff: {(i, k): coeff} for C[i, k]
    
    # Standard Strassen decomposition for 2x2x2
    # M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
    # M2 = (A10+A11)B00 -> C10+=M2, C11-=M2
    # M3 = A00(B01-B11) -> C01+=M3, C11+=M3
    # M4 = A11(B10-B00) -> C00+=M4, C10+=M4
    # M5 = (A00+A01)B11 -> C00-=M5, C01+=M5
    # M6 = (A10-A00)(B00+B01) -> C11+=M6
    # M7 = (A01-A11)(B10+B11) -> C00+=M7
    
    strassen_terms = [
        ({(0,0): 1, (1,1): 1}, 
         {(0,0): 1, (1,1): 1}, 
         {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, 
         {(0,0): 1}, 
         {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, 
         {(0,1): 1, (1,1): -1}, 
         {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, 
         {(1,0): 1, (0,0): -1}, 
         {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, 
         {(1,1): 1}, 
         {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, 
         {(0,0): 1, (0,1): 1}, 
         {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, 
         {(1,0): 1, (1,1): 1}, 
         {(0,0): 1}),
    ]
    
    # There are p//2 = 4 blocks of size 2 in the column dimension of B and C.
    num_blocks = p // 2
    
    for q in range(num_blocks):
        k_base = 2 * q
        
        for r_idx, (u_coeff, v_coeff, w_coeff) in enumerate(strassen_terms):
            t = q * 7 + r_idx
            
            # Fill U
            for (i, j), coeff in u_coeff.items():
                U[i * m + j, t] = coeff
            
            # Fill V
            # j is row of B, k_local is col of B within the block
            for (j, k_local), coeff in v_coeff.items():
                k_global = k_base + k_local
                V[j * p + k_global, t] = coeff
            
            # Fill W
            # i is row of C (0..1), k_local is col of C within the block
            # W index convention: k * n + i
            for (i, k_local), coeff in w_coeff.items():
                k_global = k_base + k_local
                W[k_global * n + i, t] = coeff
    
    # Check if time is running out (unlikely for hardcoded solution)
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
