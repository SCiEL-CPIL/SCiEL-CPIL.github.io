"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Schoolbook decomposition as baseline (rank 90)
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
    
    # Try to reduce rank using Strassen on 2x2x2 sub-blocks
    # We'll rebuild the decomposition with Strassen where applicable
    
    terms = []  # (u_idx, v_idx, w_idx, u_val, v_val, w_val)
    
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        terms.append((u_idx, v_idx, w_idx, u_val, v_val, w_val))
    
    # Partition indices:
    # A: rows 0,1,2; cols 0,1,2,3,4
    # B: rows 0,1,2,3,4; cols 0,1,2,3,4,5
    # C: rows 0,1,2; cols 0,1,2,3,4,5
    
    # For each j (the shared index), we need A[i,j] * B[j,k] -> C[i,k]
    # Group j values: j in {0,1} can use Strassen if i in {0,1} and k in {0,1} or {2,3} or {4,5}
    # j in {2,3} similarly
    # j = 4 is alone
    
    # j-block 0: j in {0,1}
    #   i-blocks: {0,1}, {2}
    #   k-blocks: {0,1}, {2,3}, {4,5}
    for j0, j1 in [(0, 1)]:
        for i0, i1 in [(0, 1), (2, 2)]:
            for k0, k1 in [(0, 1), (2, 3), (4, 5)]:
                i_count = i1 - i0 + 1
                j_count = j1 - j0 + 1
                k_count = k1 - k0 + 1
                
                if i_count == 2 and j_count == 2 and k_count == 2:
                    # Strassen 2x2x2
                    # M1 = (a11+a22)(b11+b22) -> c11 += M1, c22 += M1
                    for di in range(2):
                        for dj in range(2):
                            add_term((i0+di)*m + (j0+dj), (j0+dj)*p + k0, k0*n + (i0+di), 1, 1, 1)
                            add_term((i0+di)*m + (j0+dj), (j0+dj)*p + (k0+1), (k0+1)*n + (i0+di), 1, 1, 1)
                    # This is still schoolbook. Let me properly implement Strassen.
                    # Actually, let me just use schoolbook for all and accept rank 90.
                    pass
                else:
                    for di in range(i_count):
                        for dj in range(j_count):
                            for dk in range(k_count):
                                i = i0 + di
                                j = j0 + dj
                                k = k0 + dk
                                add_term(i*m + j, j*p + k, k*n + i, 1, 1, 1)
    
    # j-block 1: j in {2,3}
    for j0, j1 in [(2, 3)]:
        for i0, i1 in [(0, 1), (2, 2)]:
            for k0, k1 in [(0, 1), (2, 3), (4, 5)]:
                i_count = i1 - i0 + 1
                j_count = j1 - j0 + 1
                k_count = k1 - k0 + 1
                for di in range(i_count):
                    for dj in range(j_count):
                        for dk in range(k_count):
                            i = i0 + di
                            j = j0 + dj
                            k = k0 + dk
                            add_term(i*m + j, j*p + k, k*n + i, 1, 1, 1)
    
    # j-block 2: j = 4
    for j0, j1 in [(4, 4)]:
        for i0, i1 in [(0, 1), (2, 2)]:
            for k0, k1 in [(0, 1), (2, 3), (4, 5)]:
                i_count = i1 - i0 + 1
                j_count = j1 - j0 + 1
                k_count = k1 - k0 + 1
                for di in range(i_count):
                    for dj in range(j_count):
                        for dk in range(k_count):
                            i = i0 + di
                            j = j0 + dj
                            k = k0 + dk
                            add_term(i*m + j, j*p + k, k*n + i, 1, 1, 1)
    
    # Build final matrices from terms
    if len(terms) < rank:
        new_rank = len(terms)
        U = [[0] * new_rank for _ in range(n * m)]
        V = [[0] * new_rank for _ in range(m * p)]
        W = [[0] * new_rank for _ in range(p * n)]
        for c, (u_idx, v_idx, w_idx, u_val, v_val, w_val) in enumerate(terms):
            U[u_idx][c] = u_val
            V[v_idx][c] = v_val
            W[w_idx][c] = w_val
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
