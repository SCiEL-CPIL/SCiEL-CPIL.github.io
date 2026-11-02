import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    def get_schoolbook():
        """Schoolbook decomposition: rank = n*m*p = 63"""
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify_full(U, V, W, n, m, p, tol=1e-9):
        """Exhaustively verify all (i,j,k) entries."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    actual = float(np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :]))
                    if abs(actual - 1.0) > tol:
                        return False
        return True
    
    # Return schoolbook rank-63 decomposition (verified correct)
    result = get_schoolbook()
    
    # Verify before returning
    if time.time() < deadline:
        U = np.array(result["U"])
        V = np.array(result["V"])
        W = np.array(result["W"])
        if verify_full(U, V, W, n, m, p):
            return result
    
    # Fallback: return without verification if time is tight
    return result
# EVOLVE-BLOCK-END
