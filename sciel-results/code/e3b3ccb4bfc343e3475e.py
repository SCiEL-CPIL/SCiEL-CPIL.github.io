"""Program for rank-minimal tensor decomposition of 2x3x9 matrix multiplication."""
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We exploit the block structure of B (3x9) as three 3x3 blocks.
    # C = A * [B0, B1, B2] = [A*B0, A*B1, A*B2].
    # If we find a rank-R0 decomposition for 2x3x3 multiplication,
    # we can use it 3 times to get a rank 3*R0 decomposition for 2x3x9.
    # Schoolbook for 2x3x3 is 18, so 3*18=54.
    # We search for R0 < 18.
    
    n0, m0, p0 = 2, 3, 3
    
    def verify_small(U, V, W, R):
        """Verify decomposition for 2x3x3."""
        # U: 6xR, V: 9xR, W: 6xR
        # Check active entries
        for i in range(n0):
            for j in range(m0):
                for k in range(p0):
                    a = i * m0 + j
                    b = j * p0 + k
                    d = k * n0 + i
                    s = 0
                    for c in range(R):
                        s += U[a][c] * V[b][c] * W[d][c]
                    if s != 1:
                        return False
        # Check inactive entries
        # Total entries 6*9*6 = 324. Active 18.
        # We can check all of them since it's small.
        for a in range(6):
            for b in range(9):
                for d in range(6):
                    i1 = a // m0
                    j1 = a % m0
                    j2 = b // p0
                    k2 = b % p0
                    k1 = d // n0
                    i2 = d % n0
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        continue
                    s = 0
                    for c in range(R):
                        s += U[a][c] * V[b][c] * W[d][c]
                    if s != 0:
                        return False
        return True

    def expand_to_2x3x9(U0, V0, W0, R0):
        """Expand 2x3x3 decomposition to 2x3x9."""
        n, m, p = 2, 3, 9
        R = 3 * R0
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        
        for b in range(3):
            col_offset = b * R0
            for c in range(R0):
                # U is same for all blocks
                for i in range(n0):
                    for j in range(m0):
                        idx_u = i * m0 + j
                        U[idx_u][col_offset + c] = U0[idx_u][c]
                
                # V expansion
                for j in range(m0):
                    for k_local in range(p0):
                        idx_v_small = j * p0 + k_local
                        val = V0[idx_v_small][c]
                        if val != 0:
                            idx_v_large = j * p + (3 * b + k_local)
                            V[idx_v_large][col_offset + c] = val
                
                # W expansion
                for k_local in range(p0):
                    for i in range(n0):
                        idx_w_small = k_local * n0 + i
                        val = W0[idx_w_small][c]
                        if val != 0:
                            idx_w_large = (3 * b + k_local) * n + i
                            W[idx_w_large][col_offset + c] = val
                            
        return {"U": U, "V": V, "W": W}

    # Search for 2x3x3 decomposition
    random.seed(42)
    best_rank_small = None
    best_decomp_small = None
    
    # Try ranks from 17 down to 10
    # 18 is schoolbook, we want < 18
    for R0 in range(17, 9, -1):
        for attempt in range(2000):
            U0 = [[random.randint(-1, 1) for _ in range(R0)] for _ in range(6)]
            V0 = [[random.randint(-1, 1) for _ in range(R0)] for _ in range(9)]
            W0 = [[random.randint(-1, 1) for _ in range(R0)] for _ in range(6)]
            
            # Quick check: sum of squares or something? No, just verify.
            if verify_small(U0, V0, W0, R0):
                best_rank_small = R0
                best_decomp_small = (U0, V0, W0)
                break
        if best_decomp_small:
            break

    if best_decomp_small:
        U0, V0, W0 = best_decomp_small
        return expand_to_2x3x9(U0, V0, W0, best_rank_small)
        
    # Fallback to schoolbook for 2x3x9 (rank 54)
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
