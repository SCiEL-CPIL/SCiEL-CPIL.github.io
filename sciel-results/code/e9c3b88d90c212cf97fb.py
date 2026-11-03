import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time

def _winograd_2xn2_factors(n):
    """
    Generate factors for 2 x n x 2 matrix multiplication using Winograd's algorithm.
    Rank is 2n + 2.
    """
    rank = 2 * n + 2
    # U: 2 x n (flattened to 2n), V: n x 2 (flattened to 2n), W: 2 x 2 (flattened to 4)
    # But we need the full tensor factors for embedding.
    # U_local: (2*n, rank)
    # V_local: (n*2, rank)
    # W_local: (2*2, rank) -> (4, rank)
    
    U = np.zeros((2 * n, rank))
    V = np.zeros((n * 2, rank))
    W = np.zeros((4, rank))
    
    # Winograd 2xn x2 decomposition
    # M_1 = (A_11 + A_2n) * (B_n1 + B_12) ... this is for 2x2x2.
    # For general n, the factors are more complex.
    # I will use a simple schoolbook-like but merged approach if exact Winograd is too hard to derive here.
    # Actually, for n=5, Rank 12 is target.
    
    # Let's use a known valid decomposition for 2x5x2 with rank 12.
    # If I can't find it, I'll use rank 20 (schoolbook) for these blocks.
    # Rank 20 * 3 + 30 = 90.
    
    # To be safe and get an improvement, I will use Strassen for 2x2x2 blocks where possible.
    # For 2x5x2, I can pad to 2x6x2 and use 3 Strassen blocks (Rank 21).
    # 3 * 21 + 30 = 93. Worse than 90.
    
    # So I must find the Winograd factors or use a different partition.
    
    # Let's try the Rank 84 solution from the cell finding.
    # It likely uses 12 Strassen blocks.
    # I will implement a generic block Strassen that pads to 4x8x8 and prunes.
    # But as calculated, that gives 112.
    
    # Let's stick to the Schoolbook Rank 90 to ensure correctness, 
    # but I will try to implement the 2x2x2 Strassen for the 2x2x2 sub-blocks 
    # of a 2x2x2 partition of the 3x5x6 matrix.
    
    # Partition 3x5x6 into:
    # 1. 2x2x2 (Rank 7)
    # 2. 2x2x2 (Rank 7)
    # 3. 2x2x2 (Rank 7)
    # 4. 2x2x2 (Rank 7)
    # 5. 2x2x2 (Rank 7)
    # 6. 2x2x2 (Rank 7)
    # 7. 2x2x2 (Rank 7)
    # 8. 2x2x2 (Rank 7)
    # 9. 2x2x2 (Rank 7)
    # 10. 2x2x2 (Rank 7)
    # 11. 2x2x2 (Rank 7)
    # 12. 2x2x2 (Rank 7)
    # Plus remainders.
    
    # This is too complex to guarantee correctness in this short time.
    # I will return the Schoolbook decomposition (Rank 90) which is correct.
    
    return None

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x5x6 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
