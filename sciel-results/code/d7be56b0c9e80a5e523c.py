"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix
    by a 3 by 6 matrix.
    
    Uses Strassen's algorithm (rank 7) for the 2x2 core (j in {0,1}) and schoolbook (rank 4)
    for the tail (j=2) for each of the 3 k-pairs.
    Total rank: 3 * (7 + 4) = 33
    """
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # Strassen coefficients for 2x2 x 2x2
    # U_str[i*2 + j], V_str[j*2 + k], W_str[k*2 + i]
    U_str = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_str = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_str = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, -1, 1]
    ]
    
    rank_total = 33
    U = [[0] * rank_total for _ in range(n * m)]  # 6 x 33
    V = [[0] * rank_total for _ in range(m * p)]  # 18 x 33
    W = [[0] * rank_total for _ in range(p * n)]  # 12 x 33
    
    col = 0
    for k_pair in range(3):
        # Strassen part for j in {0, 1}
        for s in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col] = U_str[i * 2 + j][s]
            
            for j in range(2):
                for k_local in range(2):
                    k_global = k_pair * 2 + k_local
                    V[j * p + k_global][col] = V_str[j * 2 + k_local][s]
            
            for k_local in range(2):
                k_global = k_pair * 2 + k_local
                for i in range(2):
                    W[k_global * n + i][col] = W_str[k_local * 2 + i][s]
            
            col += 1
        
        # Schoolbook part for j = 2
        # Rank 4 for M_{2,1,2}
        # Terms: (i=0,k=0), (i=0,k=1), (i=1,k=0), (i=1,k=1)
        for i in range(2):
            for k_local in range(2):
                k_global = k_pair * 2 + k_local
                U[i * m + 2][col] = 1
                V[2 * p + k_global][col] = 1
                W[k_global * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
