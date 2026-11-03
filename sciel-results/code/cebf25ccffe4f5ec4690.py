import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    # Rank is 42 based on 2 blocks of (3 sub-blocks of Strassen 2x2x2)
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)] # 8 x 42
    V = [[0] * rank for _ in range(m * p)] # 24 x 42
    W = [[0] * rank for _ in range(p * n)] # 12 x 42
    
    # Strassen 2x2x2 decomposition coefficients
    # Order of indices for 2x2 matrix: 00, 01, 10, 11
    # u: A coeffs, v: B coeffs, w: C coeffs
    strassen_steps = [
        # M1 = (A00+A11)(B00+B11) -> C00(+1), C11(+1)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (A10+A11)B00 -> C10(+1), C11(-1)
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3 = A00(B10-B11) -> C01(+1), C11(+1)
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),
        # M4 = A11(B01-B00) -> C00(+1), C10(+1)
        ([0, 0, 0, 1], [-1, 1, 0, 0], [1, 0, 1, 0]),
        # M5 = (A00+A01)B11 -> C00(-1), C01(+1)
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6 = (A10-A00)(B00+B01) -> C11(+1)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7 = (A01-A11)(B10+B11) -> C00(+1)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    col = 0
    # 2 main blocks (splitting inner dimension 4 into 2+2)
    for b in range(2):
        # 3 sub-blocks (splitting output dimension 6 into 3 pairs)
        for q in range(3):
            # 7 Strassen steps
            for u_loc, v_loc, w_loc in strassen_steps:
                # Process U (A matrix)
                # u_loc indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                for idx in range(4):
                    val = u_loc[idx]
                    if val != 0:
                        i_prime = idx // 2
                        j_prime = idx % 2
                        global_i = i_prime
                        global_j = 2 * b + j_prime
                        U[global_i * m + global_j][col] += val
                        
                # Process V (B matrix)
                # v_loc indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                # Note: B is m x p (4x6).
                # Local B is 2x2.
                # j_prime is row in local B (0,1)
                # k_prime is col in local B (0,1)
                for idx in range(4):
                    val = v_loc[idx]
                    if val != 0:
                        j_prime = idx // 2
                        k_prime = idx % 2
                        global_j = 2 * b + j_prime
                        global_k = 2 * q + k_prime
                        V[global_j * p + global_k][col] += val
                        
                # Process W (C matrix)
                # w_loc indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                # w_loc corresponds to C_prime (2x2)
                # i_prime is row in local C (0,1)
                # k_prime is col in local C (0,1)
                for idx in range(4):
                    val = w_loc[idx]
                    if val != 0:
                        i_prime = idx // 2
                        k_prime = idx % 2
                        global_i = i_prime
                        global_k = 2 * q + k_prime
                        # Task convention: W[k*n+i, r]
                        W[global_k * n + global_i][col] += val
                        
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
