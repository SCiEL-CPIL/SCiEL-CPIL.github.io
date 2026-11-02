"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose the 2x3x8 matrix multiplication tensor into rank-1 terms.
    
    Strategy:
    - Split n=2 into two independent n=1 blocks (direct sum).
    - For each n=1 block:
        - Split m=3 into m=1 and m=2.
        - m=1, p=8: rank 1 (outer product of vectors of length 8).
        - m=2, p=8: split p=8 into four p=2 blocks.
            - Each p=2 block: split m=2 into m=1+m=1.
                - Each m=1, p=2: rank 1.
                - So each p=2 block has rank 2.
            - Total for m=2, p=8: 4 * 2 = 8.
        - Total for n=1, m=3, p=8: 1 + 8 = 9.
    - Total rank: 2 * 9 = 18.
    """
    n, m, p = DIMENSIONS
    
    # Initialize U, V, W as lists of lists with 18 columns
    rank = 18
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 18
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 18
    W = [[0.0] * rank for _ in range(p * n)]  # 16 x 18
    
    col = 0
    
    for i in range(n):
        # For each i (row of A, also determines column block in C)
        
        # m=1 part (j=0)
        # This is a 1x1x8 tensor: rank 1
        # U[i*m + 0, col] = 1
        # V[0*p + k, col] = 1 for k in 0..7
        # W[k*n + i, col] = 1 for k in 0..7
        U[i * m + 0][col] = 1.0
        for k in range(p):
            V[0 * p + k][col] = 1.0
            W[k * n + i][col] = 1.0
        col += 1
        
        # m=2 part (j=1, j=2)
        # Split p=8 into four p=2 blocks: (k0,k1), (k2,k3), (k4,k5), (k6,k7)
        for block in range(4):
            k0 = block * 2
            k1 = block * 2 + 1
            
            # Split m=2 into m=1 (j=1) and m=1 (j=2)
            
            # m=1, p=2: rank 1
            # U[i*m + 1, col] = 1
            # V[1*p + k0, col] = 1, V[1*p + k1, col] = 1
            # W[k0*n + i, col] = 1, W[k1*n + i, col] = 1
            U[i * m + 1][col] = 1.0
            V[1 * p + k0][col] = 1.0
            V[1 * p + k1][col] = 1.0
            W[k0 * n + i][col] = 1.0
            W[k1 * n + i][col] = 1.0
            col += 1
            
            # m=1, p=2: rank 1
            # U[i*m + 2, col] = 1
            # V[2*p + k0, col] = 1, V[2*p + k1, col] = 1
            # W[k0*n + i, col] = 1, W[k1*n + i, col] = 1
            U[i * m + 2][col] = 1.0
            V[2 * p + k0][col] = 1.0
            V[2 * p + k1][col] = 1.0
            W[k0 * n + i][col] = 1.0
            W[k1 * n + i][col] = 1.0
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
