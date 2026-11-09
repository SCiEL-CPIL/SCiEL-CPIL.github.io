"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # safety margin
    
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 36
    
    # Schoolbook decomposition
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # U[i*m+j, t], V[j*p+k, t], W[k*n+i, t]
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify function: check that sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c]
    # where T[(i,j),(j',k),(k',i')] = 1 iff i==i', j==j', k==k'
    def verify(U_mat, V_mat, W_mat):
        U_np = np.array(U_mat, dtype=float)
        V_np = np.array(V_mat, dtype=float)
        W_np = np.array(W_mat, dtype=float)
        r = U_np.shape[1]
        # Check all entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U_np[a, :] * V_np[b, :] * W_np[c, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check some off-diagonal entries are zero
        # T[(i,j),(j',k),(k',i')] = 0 unless i==i', j==j', k==k'
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if i == i2 and j == j2 and k == k2:
                                    continue
                                a = i * m + j
                                b = j2 * p + k2
                                c = k2 * n + i2
                                expected = 0.0
                                val = np.sum(U_np[a, :] * V_np[b, :] * W_np[c, :])
                                if abs(val - expected) > 1e-9:
                                    return False
        return True
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_school
    
    # Try to find lower rank via random search
    for r in range(rank_school - 1, max(15, rank_school - 8), -1):
        if time.time() > deadline:
            break
        max_attempts = 20
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            np.random.seed(attempt * 7919 + r)
            U_rand = np.random.randn(n * m, r)
            V_rand = np.random.randn(m * p, r)
            W_rand = np.random.randn(p * n, r)
            
            # Round to half-integers
            U_r = np.round(U_rand * 2) / 2
            V_r = np.round(V_rand * 2) / 2
            W_r = np.round(W_rand * 2) / 2
            
            if verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = r
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
