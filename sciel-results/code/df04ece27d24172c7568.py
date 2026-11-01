"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank 14 decomposition using Strassen's algorithm for two 2x2 blocks
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U vectors for Strassen (same for left and right blocks)
    # A is 2x2: indices 0,1,2,3 correspond to a00, a01, a10, a11
    U_strassen = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    # V vectors for Left Block (cols 0,1 of B)
    # B is 2x4: indices 0,1,2,3,4,5,6,7 correspond to b00,b01,b02,b03,b10,b11,b12,b13
    V_left = [
        [1, 0, 0, 0, 0, 1, 0, 0], # M1: b00 + b11
        [1, 0, 0, 0, 0, 0, 0, 0], # M2: b00
        [0, 1, 0, 0, 0, -1, 0, 0], # M3: b01 - b11
        [-1, 0, 0, 0, 1, 0, 0, 0], # M4: b10 - b00
        [0, 0, 0, 0, 0, 1, 0, 0], # M5: b11
        [1, 1, 0, 0, 0, 0, 0, 0], # M6: b00 + b01
        [0, 0, 0, 0, 1, 1, 0, 0]  # M7: b10 + b11
    ]
    
    # V vectors for Right Block (cols 2,3 of B)
    V_right = [
        [0, 0, 1, 0, 0, 0, 0, 1], # M1: b02 + b13
        [0, 0, 1, 0, 0, 0, 0, 0], # M2: b02
        [0, 0, 0, 1, 0, 0, 0, -1], # M3: b03 - b13
        [0, 0, -1, 0, 0, 0, 1, 0], # M4: b12 - b02
        [0, 0, 0, 0, 0, 0, 0, 1], # M5: b13
        [0, 0, 1, 1, 0, 0, 0, 0], # M6: b02 + b03
        [0, 0, 0, 0, 0, 0, 1, 1]  # M7: b12 + b13
    ]
    
    # W vectors for Left Block (cols 0,1 of C)
    # C is 2x4: indices k*n+i. k=0,1.
    # k=0: i=0->0, i=1->1. k=1: i=0->2, i=1->3.
    W_left = [
        [1, 0, 0, 1, 0, 0, 0, 0], # M1: C00+=1, C11+=1
        [0, 1, 0, -1, 0, 0, 0, 0], # M2: C10+=1, C11-=1
        [0, 0, 1, 1, 0, 0, 0, 0], # M3: C01+=1, C11+=1
        [1, 1, 0, 0, 0, 0, 0, 0], # M4: C00+=1, C10+=1
        [-1, 0, 1, 0, 0, 0, 0, 0], # M5: C00-=1, C01+=1
        [0, 0, 0, 1, 0, 0, 0, 0], # M6: C11+=1
        [1, 0, 0, 0, 0, 0, 0, 0]  # M7: C00+=1
    ]
    
    # W vectors for Right Block (cols 2,3 of C)
    # k=2: i=0->4, i=1->5. k=3: i=0->6, i=1->7.
    W_right = [
        [0, 0, 0, 0, 1, 0, 0, 1], # M1: C02+=1, C13+=1
        [0, 0, 0, 0, 0, 1, 0, -1], # M2: C12+=1, C13-=1
        [0, 0, 0, 0, 0, 0, 1, 1], # M3: C03+=1, C13+=1
        [0, 0, 0, 0, 1, 1, 0, 0], # M4: C02+=1, C12+=1
        [0, 0, 0, 0, -1, 0, 1, 0], # M5: C02-=1, C03+=1
        [0, 0, 0, 0, 0, 0, 0, 1], # M6: C13+=1
        [0, 0, 0, 0, 1, 0, 0, 0]  # M7: C02+=1
    ]
    
    # Fill U, V, W
    for r in range(7):
        # Left block
        for i in range(n * m):
            U[i][r] = U_strassen[r][i]
        for i in range(m * p):
            V[i][r] = V_left[r][i]
        for i in range(p * n):
            W[i][r] = W_left[r][i]
            
        # Right block
        for i in range(n * m):
            U[i][r + 7] = U_strassen[r][i]
        for i in range(m * p):
            V[i][r + 7] = V_right[r][i]
        for i in range(p * n):
            W[i][r + 7] = W_right[r][i]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
