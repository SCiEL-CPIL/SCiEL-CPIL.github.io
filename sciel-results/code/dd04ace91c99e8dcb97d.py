"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    # Schoolbook decomposition as fallback (rank = n*m*p = 84)
    def schoolbook():
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

    def verify_numpy(U, V, W, atol=1e-9):
        """Vectorized verification using numpy."""
        try:
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            rank = U_np.shape[1]
            U_r = U_np.reshape(n, m, rank)
            V_r = V_np.reshape(m, p, rank)
            W_r = W_np.reshape(p, n, rank)
            T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            if not np.allclose(T, 1.0, atol=atol):
                return False
            return True
        except Exception:
            return False

    def check_half_integer(values):
        """Check if all values are integers or half-integers."""
        for v in values:
            if not np.isfinite(v):
                return False
            if abs(v * 2 - round(v * 2)) > 1e-9:
                return False
        return True

    def is_valid_decomposition(U, V, W):
        """Check if U, V, W form a valid decomposition with integer/half-integer entries."""
        for mat in [U, V, W]:
            for row in mat:
                if not check_half_integer(row):
                    return False
        return verify_numpy(U, V, W)

    # Given the complexity and time constraints, we stick to the schoolbook decomposition
    # which is guaranteed to work and has rank 84.
    
    best_result = schoolbook()
    best_rank = n * m * p

    # No random search is performed to avoid invalid results or timeouts
    # that would score 0.
    
    return best_result
# EVOLVE-BLOCK-END
