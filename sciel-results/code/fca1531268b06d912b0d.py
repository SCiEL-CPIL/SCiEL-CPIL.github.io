"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Leave 10s safety margin for evaluation
    
    def _verify_decomposition(U, V, W, n, m, p, num_trials=10):
        """Verify that the decomposition is correct."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for _ in range(num_trials):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for r in range(rank):
                # left[i] = sum_j U[i*m+j, r] * A[i, j]
                left = np.zeros(n)
                for i in range(n):
                    left[i] = np.dot(U[i*m:(i+1)*m, r], A[i, :])
                # right[k] = sum_j V[j*p+k, r] * B[j, k]
                right = np.zeros(p)
                for k in range(p):
                    right[k] = np.dot(V[k::p, r], B[:, k])
                # C[i,k] += left[i] * right[k] * W[k*n+i, r]
                for k in range(p):
                    for i in range(n):
                        C_computed[i, k] += left[i] * right[k] * W[k*n+i, r]
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True

    def _build_schoolbook(n, m, p):
        """Build the schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W

    # Fallback to schoolbook
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

def _try_rank(n, m, p, rank):
    """Try to determine if a rank-r decomposition exists."""
    return False

def _solve_decomposition(n, m, p, rank):
    """Solve for the decomposition matrices."""
    return None, None, None
# EVOLVE-BLOCK-END
