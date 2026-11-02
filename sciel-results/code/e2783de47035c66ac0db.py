import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Strategy:
    # Decompose the 2x5x6 tensor into:
    # 1. A 2x1x6 part (j=0? No, the current code uses j=0 for Part 1 and j=1..4 for Part 2? 
    #    Wait, let's re-read the current program's logic carefully.)
    
    # Current Program Logic:
    # Part 1: j = 0. 
    #   Loops i in 0..1, k in 0..5.
    #   Adds 12 terms. U[i*5+0], V[0*6+k], W[k*2+i].
    # Part 2: j in 1..4.
    #   Split into two blocks of 2x2x6? No, "Split into two blocks of 2x2x6" comment is confusing.
    #   It says "Block 1: j in 1, 2" and "Block 2: j in 3, 4".
    #   So it treats j=1,2 as one 2-wide block and j=3,4 as another 2-wide block.
    #   For each block (j_base, j_base+1), it loops s in 0..2 (k pairs: 0,1; 2,3; 4,5).
    #   For each k pair, it applies Strassen 2x2x2.
    #   
    # Total Terms:
    # Part 1: 12 terms.
    # Part 2: 2 blocks (j-pairs) * 3 k-pairs * 7 Strassen terms = 42 terms.
    # Total Rank = 54.
    
    # This logic is sound. I will implement it cleanly.
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j = 0
    # A[i,0] * B[0,k] -> C[i,k]
    for i in range(n):
        for k in range(p):
            # U index: i * m + 0
            U[i * m][col] = 1
            # V index: 0 * p + k
            V[k][col] = 1
            # W index: k * n + i
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: j in 1..4
    # Split into two blocks: j in {1, 2} and j in {3, 4}
    for block_idx in range(2):
        # Local j' 0, 1 map to Global j
        j_base = 1 + block_idx * 2
        j1 = j_base
        j2 = j_base + 1
        
        # For each Strassen instance (3 pairs of columns k)
        for s in range(3):
            # Local k 0, 1 map to Global k
            k_base = s * 2
            k1 = k_base
            k2 = k_base + 1
            
            # Indices for U (A[i, j])
            u00 = 0 * m + j1
            u01 = 0 * m + j2
            u10 = 1 * m + j1
            u11 = 1 * m + j2
            
            # Indices for V (B[j, k])
            v00 = j1 * p + k1
            v01 = j1 * p + k2
            v10 = j2 * p + k1
            v11 = j2 * p + k2
            
            # Indices for W (C[i, k])
            w00 = k1 * n + 0
            w01 = k2 * n + 0
            w10 = k1 * n + 1
            w11 = k2 * n + 1
            
            # Strassen's 7 products for 2x2x2 matrix multiplication
            # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
            # P1 = (a00+a11)(b00+b11)
            # P2 = (a10+a11)b00
            # P3 = a00(b01-b11)
            # P4 = a11(b10-b00)
            # P5 = (a00+a01)b11
            # P6 = (a10-a00)(b00+b01)
            # P7 = (a01-a11)(b10+b11)
            
            # P1
            U[u00][col] += 1
            U[u11][col] += 1
            V[v00][col] += 1
            V[v11][col] += 1
            W[w00][col] += 1
            W[w11][col] += 1
            col += 1

            # P2
            U[u10][col] += 1
            U[u11][col] += 1
            V[v00][col] += 1
            W[w10][col] += 1
            W[w11][col] += -1
            col += 1

            # P3
            U[u00][col] += 1
            V[v01][col] += 1
            V[v11][col] += -1
            W[w01][col] += 1
            W[w11][col] += 1
            col += 1

            # P4
            U[u11][col] += 1
            V[v10][col] += 1
            V[v00][col] += -1
            W[w00][col] += 1
            W[w10][col] += 1
            col += 1

            # P5
            U[u00][col] += 1
            U[u01][col] += 1
            V[v11][col] += 1
            W[w00][col] += -1
            W[w01][col] += 1
            col += 1

            # P6
            U[u10][col] += 1
            U[u00][col] += -1
            V[v00][col] += 1
            V[v01][col] += 1
            W[w11][col] += 1
            col += 1

            # P7
            U[u01][col] += 1
            U[u11][col] += -1
            V[v10][col] += 1
            V[v11][col] += 1
            W[w00][col] += 1
            col += 1
            
    # Verify the decomposition using numpy for speed
    def verify(U_mat, V_mat, W_mat):
        U_np = np.array(U_mat, dtype=np.int64)
        V_np = np.array(V_mat, dtype=np.int64)
        W_np = np.array(W_mat, dtype=np.int64)
        # Reshape for efficient batched computation
        U_r = U_np.reshape(n, m, -1)  # (n, m, rank)
        V_r = V_np.reshape(m, p, -1)  # (m, p, rank)
        W_r = W_np.reshape(p, n, -1)  # (p, n, rank)
        # For each (i,j,k), check sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :]
                    if np.sum(prod) != 1:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback: return a rank-60 trivial decomposition
        rank_fallback = n * m * p
        U_f = [[0] * rank_fallback for _ in range(n * m)]
        V_f = [[0] * rank_fallback for _ in range(m * p)]
        W_f = [[0] * rank_fallback for _ in range(p * n)]
        col_f = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j][col_f] = 1
                    V_f[j * p + k][col_f] = 1
                    W_f[k * n + i][col_f] = 1
                    col_f += 1
        return {"U": U_f, "V": V_f, "W": W_f}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
