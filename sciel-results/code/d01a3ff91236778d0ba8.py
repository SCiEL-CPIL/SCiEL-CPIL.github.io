"""Matrix multiplication 2,3,8 rank-minimal decomposition."""
DIMENSIONS = [2, 3, 8]

import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    # Partition p=8 into 4 groups of 2: cols (0,1),(2,3),(4,5),(6,7)
    # For each group, solve (2,3,2) matrix multiplication
    # Rank of (2,3,2) is 8 (no known reduction)
    # Total rank = 4 * 8 = 32
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]  # 6 x 32
    V = [[0] * rank for _ in range(m * p)]  # 24 x 32
    W = [[0] * rank for _ in range(p * n)]  # 16 x 32
    
    # For each of the 4 column-pairs (k0, k1) where k0 = 2*g, k1 = 2*g+1
    for g in range(4):
        k0 = 2 * g
        k1 = 2 * g + 1
        # The (2,3,2) subproblem: A (2x3) * B_g (3x2) = C_g (2x2)
        # C_g[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
        # C_g[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
        # C_g[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
        # C_g[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
        # 
        # Using Strassen-like decomposition for (2,3,2):
        # We can use 8 rank-1 terms (schoolbook for this subproblem)
        # Or we can try to reduce. Actually rank(2,3,2) = 8 is tight.
        # So we use 8 terms per group.
        
        base = g * 8
        # For (2,3,2), the schoolbook decomposition uses 8 terms:
        # (i,j,k) in {0,1} x {0,1,2} x {0,1}
        for i in range(2):
            for j in range(3):
                for k_local in range(2):
                    col = base + i * 3 + j * 2 + k_local
                    # U index: i*m + j = i*3 + j
                    U[i * m + j][col] = 1
                    # V index: j*p + k where k = k0 + k_local
                    V[j * p + (k0 + k_local)][col] = 1
                    # W index: k*n + i where k = k0 + k_local
                    W[(k0 + k_local) * n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
