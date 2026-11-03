"""Matrix multiplication 3x4x4 rank-minimal tensor decomposition."""
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # n=3, m=4, p=4
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # (12, 36)
    V = [[0] * rank for _ in range(m * p)]  # (16, 36)
    W = [[0] * rank for _ in range(p * n)]  # (12, 36)
    
    col = 0
    
    def strassen_2x2(i_base, k_base, j_base):
        """Add 7 rank-1 terms for A[i_base:i_base+2, k_base:k_base+2] * B[k_base:k_base+2, j_base:j_base+2] -> C[i_base:i_base+2, j_base:j_base+2]"""
        nonlocal col
        # 7 Strassen terms: (u_A, v_B, w_C) where indices are (00,01,10,11)
        terms = [
            ((1,0,0,1), (1,0,0,1), (1,0,0,-1)),
            ((0,0,1,1), (1,0,0,0), (0,0,1,-1)),
            ((1,0,0,0), (0,0,1,-1), (0,1,0,1)),
            ((0,0,0,1), (0,1,-1,0), (1,0,1,0)),
            ((1,1,0,0), (0,0,0,1), (-1,1,0,0)),
            ((0,0,1,-1),(1,0,1,0), (0,0,0,1)),
            ((0,1,0,-1),(0,1,0,1), (1,1,0,0)),
        ]
        for u, v, w in terms:
            for d in range(4):
                ai = i_base + d // 2
                ak = k_base + d % 2
                bi = k_base + d // 2
                bj = j_base + d % 2
                ci = i_base + d // 2
                cj = j_base + d % 2
                if u[d] != 0:
                    U[ai * m + ak][col] += u[d]
                if v[d] != 0:
                    V[bi * p + bj][col] += v[d]
                if w[d] != 0:
                    W[cj * n + ci][col] += w[d]
            col += 1
        return 7
    
    # A1*B1: A[0:2,:]*B[:,0:2]
    strassen_2x2(0, 0, 0)  # A[0:2,0:2]*B[0:2,0:2]
    strassen_2x2(0, 2, 0)  # A[0:2,2:4]*B[2:4,0:2]
    
    # A1*B2: A[0:2,:]*B[:,2:4]
    strassen_2x2(0, 0, 2)  # A[0:2,0:2]*B[0:2,2:4]
    strassen_2x2(0, 2, 2)  # A[0:2,2:4]*B[2:4,2:4]
    
    # A2*B1: A[2,:]*B[:,0:2]
    for k in range(2):
        for j in range(2):
            U[2 * m + k][col] = 1
            V[k * p + j][col] = 1
            W[j * n + 2][col] = 1
            col += 1
    for k in range(2):
        for j in range(2):
            U[2 * m + 2 + k][col] = 1
            V[(2 + k) * p + j][col] = 1
            W[j * n + 2][col] = 1
            col += 1
    
    # A2*B2: A[2,:]*B[:,2:4]
    for k in range(2):
        for j in range(2):
            U[2 * m + k][col] = 1
            V[k * p + 2 + j][col] = 1
            W[(2 + j) * n + 2][col] = 1
            col += 1
    for k in range(2):
        for j in range(2):
            U[2 * m + 2 + k][col] = 1
            V[(2 + k) * p + 2 + j][col] = 1
            W[(2 + j) * n + 2][col] = 1
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
