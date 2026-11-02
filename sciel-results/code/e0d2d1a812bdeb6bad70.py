"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s limit
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}, rank
    
    def verify(U, V, W):
        """Check if the decomposition is valid."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        # Check all (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        # Check some random inconsistent indices to ensure 0
        # (Full check is O(nmp * rank * nm * mp * pn) which is too slow)
        # But for small dims, we can check all 2025 entries
        nm, mp, pn = n*m, m*p, p*n
        for u in range(nm):
            i_u, j_u = divmod(u, m)
            for v in range(mp):
                j_v, k_v = divmod(v, p)
                for w in range(pn):
                    k_w, i_w = divmod(w, n)
                    expected = 0
                    if j_u == j_v and i_u == i_w and k_v == k_w:
                        expected = 1
                    s = 0
                    for c in range(rank):
                        s += U[u][c] * V[v][c] * W[w][c]
                    if abs(s - expected) > 1e-9:
                        return False
        return True

    # Start with schoolbook
    best_result, best_rank = schoolbook()
    
    # Verify schoolbook just in case
    if not verify(best_result["U"], best_result["V"], best_result["W"]):
        # Fallback to manual construction if verify fails (should not happen)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank

    return best_result
# EVOLVE-BLOCK-END
