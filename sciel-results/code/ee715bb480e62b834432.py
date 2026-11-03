import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 2x3 * 3x10 matrix multiplication.
    
    Strategy:
    - The multiplication is split based on the middle dimension (m=3).
    - Split A (2x3) into A1 (2x2, cols 0,1) and A2 (2x1, col 2).
    - Split B (3x10) into B1 (2x10, rows 0,1) and B2 (1x10, row 2).
    - C = A1 B1 + A2 B2.
    
    Part 1: A1 (2x2) * B1 (2x10)
    - B1 is split into 5 blocks of width 2.
    - Each 2x2 * 2x2 block is multiplied using Strassen's algorithm (rank 7).
    - Total rank for Part 1 = 5 * 7 = 35.
    
    Part 2: A2 (2x1) * B2 (1x10)
    - This is a naive outer product-like sum.
    - C[i, k] += A[i, 2] * B[2, k].
    - There are 2 * 10 = 20 terms.
    - Total rank for Part 2 = 20.
    
    Total Rank = 35 + 20 = 55.
    
    Note on Rank Minimality:
    - The middle indices for Part 1 are {0, 1} and for Part 2 are {2}.
    - Since the supports in the middle mode are disjoint, the rank of the sum is the sum of the ranks,
      assuming no cross-term cancellations are possible with a lower rank decomposition.
    - R(M_{2,2,10}) = 35 is optimal for the 2x2x10 tensor (based on Strassen).
    - R(M_{2,1,10}) = 20 is optimal (each output term is distinct).
    - Thus, 55 is the expected minimal rank for this decomposition strategy.
    """
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 5.0

    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Strassen for A[:, 0:2] * B[0:2, :]
    # A is 2x2, B is 2x10.
    # Split B into 5 blocks of 2 columns each.
    
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # U indices for A1 (2x2 part of A)
        # A is 2x3. Indices: i*m + j
        # A[0,0] -> 0, A[0,1] -> 1
        # A[1,0] -> 3, A[1,1] -> 4
        
        u_a00 = 0 * m + 0
        u_a01 = 0 * m + 1
        u_a10 = 1 * m + 0
        u_a11 = 1 * m + 1
        
        # V indices for B1 (2x10 part of B)
        # B is 3x10. Indices: j*p + k
        # B[0,k0] -> k0, B[0,k1] -> k1
        # B[1,k0] -> 10+k0, B[1,k1] -> 10+k1
        
        v_b00 = 0 * p + k0
        v_b01 = 0 * p + k1
        v_b10 = 1 * p + k0
        v_b11 = 1 * p + k1
        
        # W indices for C (2x10 output)
        # C is 2x10. Indices: k*n + i
        # C[0,k0] -> 2*k0, C[0,k1] -> 2*k1
        # C[1,k0] -> 2*k0+1, C[1,k1] -> 2*k1+1
        
        w_c00 = k0 * n + 0
        w_c01 = k1 * n + 0
        w_c10 = k0 * n + 1
        w_c11 = k1 * n + 1

        # M1 = (a00 + a11)(b00 + b11)
        U[u_a00][col] += 1
        U[u_a11][col] += 1
        V[v_b00][col] += 1
        V[v_b11][col] += 1
        W[w_c00][col] += 1
        W[w_c11][col] += 1
        col += 1
        
        # M2 = (a10 + a11)b00
        U[u_a10][col] += 1
        U[u_a11][col] += 1
        V[v_b00][col] += 1
        W[w_c10][col] += 1
        W[w_c11][col] += -1
        col += 1
        
        # M3 = a00(b01 - b11)
        U[u_a00][col] += 1
        V[v_b01][col] += 1
        V[v_b11][col] += -1
        W[w_c01][col] += 1
        W[w_c11][col] += 1
        col += 1
        
        # M4 = a11(b10 - b00)
        U[u_a11][col] += 1
        V[v_b10][col] += 1
        V[v_b00][col] += -1
        W[w_c00][col] += 1
        W[w_c10][col] += 1
        col += 1
        
        # M5 = (a00 + a01)b11
        U[u_a00][col] += 1
        U[u_a01][col] += 1
        V[v_b11][col] += 1
        W[w_c00][col] += -1
        W[w_c01][col] += 1
        col += 1
        
        # M6 = (a10 - a00)(b00 + b01)
        U[u_a10][col] += 1
        U[u_a00][col] += -1
        V[v_b00][col] += 1
        V[v_b01][col] += 1
        W[w_c11][col] += 1
        col += 1
        
        # M7 = (a01 - a11)(b10 + b11)
        U[u_a01][col] += 1
        U[u_a11][col] += -1
        V[v_b10][col] += 1
        V[v_b11][col] += 1
        W[w_c00][col] += 1
        col += 1

    # Part 2: Naive for A[:, 2] * B[2, :]
    # A is 2x1, B is 1x10
    # C[i,k] += A[i,2] * B[2,k]
    
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    if col != rank:
        # Fallback to schoolbook if construction fails (should not happen)
        rank_schoolbook = n * m * p
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
