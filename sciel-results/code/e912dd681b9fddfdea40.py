"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen decomposition for 2x2x2 blocks
    # 6 blocks, 7 products each -> rank 42
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # alpha[s][i][j]
    # beta[s][j][k]
    # w[s][i][k]
    
    alphas = [
        [[1, 0], [0, 1]],  # M1
        [[0, 0], [1, 1]],  # M2
        [[1, 0], [0, 0]],  # M3
        [[0, 0], [0, 1]],  # M4
        [[1, 1], [0, 0]],  # M5
        [[-1, 0], [1, 0]], # M6
        [[0, 1], [-1, 0]]  # M7
    ]
    
    betas = [
        [[1, 0], [0, 1]],  # M1
        [[1, 0], [0, 0]],  # M2
        [[0, 1], [-1, 0]], # M3  # y01 - y11 -> (0,1)=1, (1,1)=-1
        [[-1, 0], [1, 0]], # M4  # y10 - y00 -> (1,0)=1, (0,0)=-1
        [[0, 0], [0, 1]],  # M5
        [[1, 1], [0, 0]],  # M6
        [[0, 0], [1, 1]]   # M7
    ]
    
    ws = [
        [[1, 0], [0, 1]],  # M1: z00+=1, z11+=1
        [[0, 0], [1, -1]], # M2: z10+=1, z11-=1
        [[0, 1], [0, 0]],  # M3: z01+=1
        [[-1, 0], [0, 1]], # M4: z00-=1, z11+=1
        [[1, 0], [0, 0]],  # M5: z00+=1
        [[0, 0], [1, 1]],  # M6: z10+=1, z11+=1
        [[-1, 1], [0, 0]]  # M7: z00-=1, z01+=1
    ]
    
    # Block definitions
    # (I_start, J_start, K_start)
    # A is 2x4, B is 4x6, C is 2x6
    # A11: I=0, J=0. B11: J=0, K=0. C11: I=0, K=0
    # A11: I=0, J=0. B12: J=0, K=2. C12: I=0, K=2
    # A11: I=0, J=0. B13: J=0, K=4. C13: I=0, K=4
    # A12: I=0, J=2. B21: J=2, K=0. C11: I=0, K=0
    # A12: I=0, J=2. B22: J=2, K=2. C12: I=0, K=2
    # A12: I=0, J=2. B23: J=2, K=4. C13: I=0, K=4
    
    blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (0, 2, 0),
        (0, 2, 2),
        (0, 2, 4)
    ]
    
    for b_idx, (i_start, j_start, k_start) in enumerate(blocks):
        for s in range(7):
            r = b_idx * 7 + s
            # U: n*m = 2*4 = 8
            # Index i*m + j
            for i_loc in range(2):
                for j_loc in range(2):
                    val = alphas[s][i_loc][j_loc]
                    if val != 0:
                        i_glob = i_start + i_loc
                        j_glob = j_start + j_loc
                        U[i_glob * m + j_glob][r] = val
            
            # V: m*p = 4*6 = 24
            # Index j*p + k
            for j_loc in range(2):
                for k_loc in range(2):
                    val = betas[s][j_loc][k_loc]
                    if val != 0:
                        j_glob = j_start + j_loc
                        k_glob = k_start + k_loc
                        V[j_glob * p + k_glob][r] = val
            
            # W: p*n = 6*2 = 12
            # Index k*n + i
            for i_loc in range(2):
                for k_loc in range(2):
                    val = ws[s][i_loc][k_loc]
                    if val != 0:
                        i_glob = i_start + i_loc
                        k_glob = k_start + k_loc
                        W[k_glob * n + i_glob][r] = val
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
