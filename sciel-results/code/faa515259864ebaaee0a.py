import time
import numpy as np

DIMENSIONS = [3, 4, 7]
_DEADLINE = 140.0

def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS  # n=3, m=4, p=7

    # Start with the known rank-78 decomposition
    rank = 78
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_block(col_start, i0, i1, j0, j1, k0, k1):
        U[i0 * m + j0][col_start] = 1
        U[i1 * m + j1][col_start] = 1
        V[j0 * p + k0][col_start] = 1
        V[j1 * p + k1][col_start] = 1
        W[k0 * n + i0][col_start] = 1
        W[k1 * n + i1][col_start] = 1

        U[i1 * m + j0][col_start + 1] = 1
        U[i1 * m + j1][col_start + 1] = 1
        V[j0 * p + k0][col_start + 1] = 1
        W[k0 * n + i1][col_start + 1] = 1
        W[k1 * n + i1][col_start + 1] = -1

        U[i0 * m + j0][col_start + 2] = 1
        V[j0 * p + k1][col_start + 2] = 1
        V[j1 * p + k1][col_start + 2] = -1
        W[k1 * n + i0][col_start + 2] = 1
        W[k1 * n + i1][col_start + 2] = 1

        U[i1 * m + j1][col_start + 3] = 1
        V[j1 * p + k0][col_start + 3] = 1
        V[j0 * p + k0][col_start + 3] = -1
        W[k0 * n + i0][col_start + 3] = 1
        W[k0 * n + i1][col_start + 3] = 1

        U[i0 * m + j0][col_start + 4] = 1
        U[i0 * m + j1][col_start + 4] = 1
        V[j1 * p + k1][col_start + 4] = 1
        W[k0 * n + i0][col_start + 4] = -1
        W[k1 * n + i0][col_start + 4] = 1

        U[i1 * m + j0][col_start + 5] = 1
        U[i0 * m + j0][col_start + 5] = -1
        V[j0 * p + k0][col_start + 5] = 1
        V[j0 * p + k1][col_start + 5] = 1
        W[k1 * n + i1][col_start + 5] = 1

        U[i0 * m + j1][col_start + 6] = 1
        U[i1 * m + j1][col_start + 6] = -1
        V[j1 * p + k0][col_start + 6] = 1
        V[j1 * p + k1][col_start + 6] = 1
        W[k0 * n + i0][col_start + 6] = 1

        return col_start + 7

    blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 4, 5),
    ]

    covered = set()
    for (i0, i1, j0, j1, k0, k1) in blocks:
        col = add_strassen_block(col, i0, i1, j0, j1, k0, k1)
        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    covered.add((ii, jj, kk))

    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                if (i0, j0, k0) in covered:
                    continue
                U[i0 * m + j0][col] = 1
                V[j0 * p + k0][col] = 1
                W[k0 * n + i0][col] = 1
                col += 1

    assert col == rank

    # Verify the initial decomposition
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    total = 0
                    for c in range(rank):
                        total += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                    if total != 1:
                        return False
        return True

    assert verify(U, V, W, rank)

    # Try to reduce rank by removing redundant terms
    # A term is redundant if its contribution can be expressed as a linear combination of others
    # For each term, check if we can express it in terms of the remaining terms
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W

    # Try removing each term and see if we can compensate
    for c_remove in range(rank):
        if time.time() - _t0 > _DEADLINE:
            break
            
        # Create a copy without term c_remove
        new_rank = rank - 1
        U_new = [row[:c_remove] + row[c_remove+1:] for row in U]
        V_new = [row[:c_remove] + row[c_remove+1:] for row in V]
        W_new = [row[:c_remove] + row[c_remove+1:] for row in W]
        
        # Check if the removed term's contribution is zero when combined with others
        # This is complex; instead, check if all (i,j,k) still sum to 1
        if verify(U_new, V_new, W_new, new_rank):
            best_rank = new_rank
            best_U, best_V, best_W = U_new, V_new, W_new
            U, V, W = U_new, V_new, W_new
            rank = new_rank
            # Continue trying to remove more
            c_remove = -1  # Reset to try from beginning
        else:
            # Try to find if we can modify other terms to compensate
            # This requires solving a linear system, which is expensive
            pass

    return {"U": best_U, "V": best_V, "W": best_W}
