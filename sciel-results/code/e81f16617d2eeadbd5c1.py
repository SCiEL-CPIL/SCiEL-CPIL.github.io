"""Optimized exact tensor decomposition for 2x3 by 3x6 matrix multiplication.

Strategy:
- Use block decomposition to reduce rank below schoolbook 36.
- Split the inner dimension m=3 into 1 + 2.
- The 1x1 x 1x6 part (A[:,0] * B[0,:]) has rank 12 (no sharing possible).
- The 2x2 x 2x6 part (A[:,1:3] * B[1:3,:]) can be split into three 2x2 x 2x2 blocks.
- Each 2x2 x 2x2 block can be computed with Strassen's algorithm (rank 7).
- Total rank = 12 + 3*7 = 33.
"""

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    rank = 33
    U = [[0] * rank for _ in range(n * m)]   # 6 x 33
    V = [[0] * rank for _ in range(m * p)]   # 18 x 33
    W = [[0] * rank for _ in range(p * n)]   # 12 x 33
    
    # Part 1: A[:, 0] * B[0, :]
    # i in 0..1, k in 0..5
    # U index: i*3 + 0
    # V index: 0*6 + k
    # W index: k*2 + i
    col = 0
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A[:, 1:3] * B[1:3, :]
    # Split into 3 blocks of k (size 2 each)
    # Block b: k in {2b, 2b+1}
    # U indices for A[:, 1:3]:
    # A[0,1] -> U[1], A[0,2] -> U[2]
    # A[1,1] -> U[4], A[1,2] -> U[5]
    # V indices for B[1:3, k]:
    # B[1, k] -> V[6+k], B[2, k] -> V[12+k]
    # W indices for C[i, k]:
    # C[0, k] -> W[2k], C[1, k] -> W[2k+1]
    
    for b in range(3):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Strassen terms for this block
        # 1. M1 = (u00 + u11) * (v00 + v11)
        U[1][col] += 1
        U[5][col] += 1
        V[6 + k0][col] += 1
        V[12 + k1][col] += 1
        W[2 * k0 + 0][col] += 1
        W[2 * k1 + 0][col] += 1
        W[2 * k0 + 1][col] += 1
        W[2 * k1 + 1][col] += 1
        col += 1
        
        # 2. M2 = (u10 + u11) * v00
        U[4][col] += 1
        U[5][col] += 1
        V[6 + k0][col] += 1
        W[2 * k0 + 1][col] += 1
        W[2 * k1 + 1][col] -= 1
        col += 1
        
        # 3. M3 = u00 * (v10 - v11)
        U[1][col] += 1
        V[12 + k0][col] += 1
        V[12 + k1][col] -= 1
        W[2 * k1 + 0][col] -= 1
        W[2 * k1 + 1][col] += 1
        col += 1
        
        # 4. M4 = u11 * (v01 - v00)
        U[5][col] += 1
        V[6 + k1][col] += 1
        V[6 + k0][col] -= 1
        W[2 * k0 + 0][col] += 1
        W[2 * k0 + 1][col] -= 1
        col += 1
        
        # 5. M5 = (u00 + u01) * v11
        U[1][col] += 1
        U[2][col] += 1
        V[12 + k1][col] += 1
        W[2 * k0 + 0][col] -= 1
        W[2 * k1 + 0][col] += 1
        col += 1
        
        # 6. M6 = (u10 - u00) * (v00 + v01)
        U[4][col] += 1
        U[1][col] -= 1
        V[6 + k0][col] += 1
        V[6 + k1][col] += 1
        W[2 * k1 + 0][col] += 1
        W[2 * k1 + 1][col] -= 1
        col += 1
        
        # 7. M7 = (u01 - u11) * (v10 + v11)
        U[2][col] += 1
        U[5][col] -= 1
        V[12 + k0][col] += 1
        V[12 + k1][col] += 1
        W[2 * k0 + 0][col] += 1
        W[2 * k0 + 1][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
