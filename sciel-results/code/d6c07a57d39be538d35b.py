import time

# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    _DEADLINE = 145.0  # 5s safety margin before 150s kill
    """
    Returns the 49-rank BCLR decomposition for 4x4x4 matrix multiplication.
    
    The decomposition satisfies:
    T[i,j,k] = sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
    
    This uses a recursive Strassen-like construction:
    1. View 4x4 matrices as 2x2 blocks of 2x2 matrices.
    2. Apply Strassen's 7-multiplication algorithm at the block level.
    3. Each block multiplication is a 2x2x2 multiplication, which is also decomposed
       using Strassen's 7-multiplication algorithm.
    4. Total rank = 7 (outer) * 7 (inner) = 49.
    
    The factor matrices are precomputed as static tables for O(1) construction.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    # Precomputed factor matrices from the recursive Strassen construction.
    # U: (16, 49) - rows indexed by i*4+j for A[i][j]
    # V: (16, 49) - rows indexed by j*4+k for B[j][k]  
    # W: (16, 49) - rows indexed by k*4+i for C[k][i] (trace convention)
    
    # Strassen coefficients for 2x2 multiplication
    # beta[s][I][J]: coefficient of A[I][J] in s-th product's A-factor
    # gamma[s][J][K]: coefficient of B[J][K] in s-th product's B-factor
    # delta[s][I][K]: coefficient of C[I][K] from s-th product
    
    beta = [
        [[1, 0], [0, 1]],  # s=0: (A00+A11)
        [[0, 0], [1, 1]],  # s=1: (A10+A11)
        [[1, 0], [0, 0]],  # s=2: A00
        [[0, 0], [0, 1]],  # s=3: A11
        [[1, 1], [0, 0]],  # s=4: (A00+A01)
        [[-1, 0], [1, 0]], # s=5: (A10-A00)
        [[0, 1], [0, -1]]  # s=6: (A01-A11)
    ]
    
    gamma = [
        [[1, 0], [0, 1]],  # s=0: (B00+B11)
        [[1, 0], [0, 0]],  # s=1: B00
        [[0, 1], [0, -1]], # s=2: (B01-B11)
        [[-1, 0], [1, 0]], # s=3: (B10-B00)
        [[0, 0], [0, 1]],  # s=4: B11
        [[1, 1], [0, 0]],  # s=5: (B00+B01)
        [[0, 0], [1, 1]]   # s=6: (B10+B11)
    ]
    
    delta = [
        [[1, 0], [0, 1]],  # s=0: C00+=1, C11+=1
        [[0, 0], [1, -1]], # s=1: C10+=1, C11-=1
        [[0, 1], [0, 1]],  # s=2: C01+=1, C11+=1
        [[1, 0], [1, 0]],  # s=3: C00+=1, C10+=1
        [[-1, 1], [0, 0]], # s=4: C00-=1, C01+=1
        [[0, 0], [0, 1]],  # s=5: C11+=1
        [[1, 0], [0, 0]]   # s=6: C00+=1
    ]
    
    # Precompute non-zero entries
    beta_nz = [[(i, j, beta[s][i][j]) for i in range(2) for j in range(2) if beta[s][i][j] != 0] for s in range(7)]
    gamma_nz = [[(j, k, gamma[s][j][k]) for j in range(2) for k in range(2) if gamma[s][j][k] != 0] for s in range(7)]
    delta_nz = [[(i, k, delta[s][i][k]) for i in range(2) for k in range(2) if delta[s][i][k] != 0] for s in range(7)]
    
    # Build U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # Fill U: U[i*4+j, col] = beta[s][I][J] * beta[t][i'][j']
            for I, J, cs in beta_nz[s]:
                for ip, jp, ct in beta_nz[t]:
                    i = I * 2 + ip
                    j = J * 2 + jp
                    U[i * 4 + j][col] = cs * ct
            
            # Fill V: V[j*4+k, col] = gamma[s][J][K] * gamma[t][j'][k']
            for J, K, cs in gamma_nz[s]:
                for jp, kp, ct in gamma_nz[t]:
                    j = J * 2 + jp
                    k = K * 2 + kp
                    V[j * 4 + k][col] = cs * ct
            
            # Fill W: W[k*4+i, col] = delta[s][I][K] * delta[t][i'][k']
            for I, K, cs in delta_nz[s]:
                for ip, kp, ct in delta_nz[t]:
                    i = I * 2 + ip
                    k = K * 2 + kp
                    W[k * 4 + i][col] = cs * ct
    
    if time.time() - _t0 > _DEADLINE:
        raise TimeoutError("Exceeded deadline")
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
