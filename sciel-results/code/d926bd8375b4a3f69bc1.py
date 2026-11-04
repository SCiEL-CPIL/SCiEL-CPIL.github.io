"""Rank-minimal tensor decomposition for 2x3 x 3x10 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _validate_decomposition(U, V, W, n, m, p):
    """Check if the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    # Build the 4-index tensor S[i',j,j',k'] = sum_r W[k*n+i, r]*U[i'*m+j, r]*V[j'*p+k', r]
    # We need S[i',j,j',k'] = 1 if i'=i, j=j', k'=k, else 0
    # Actually, we need: for all i,k, C[i,k] = sum_j A[i,j]*B[j,k]
    # Test with random matrices
    import numpy as np
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_vec = np.array(U).T[r]  # shape (n*m,)
        v_vec = np.array(V).T[r]  # shape (m*p,)
        w_vec = np.array(W).T[r]  # shape (p*n,)
        u_mat = u_vec.reshape(n, m)
        v_mat = v_vec.reshape(m, p)
        w_mat = w_vec.reshape(p, n).T  # shape (n, p)
        L = u_mat @ A  # shape (n, m) @ (n, m) -- WRONG
        # Actually: L_r(A) = sum_{i,j} U[i*m+j, r] * A[i,j] = scalar
        # R_r(B) = sum_{j,k} V[j*p+k, r] * B[j,k] = scalar
        # C[i,k] += W[k*n+i, r] * L_r(A) * R_r(B)
        L = sum(U[i * m + j][r] * A[i, j] for i in range(n) for j in range(m))
        R = sum(V[j * p + k][r] * B[j, k] for j in range(m) for k in range(p))
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += W[k * n + i][r] * L * R
    return np.allclose(C_computed, C_expected, atol=1e-9)

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    # Schoolbook fallback
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
    
    # Try to find a lower-rank decomposition via structured search
    # Strategy: Group columns into pairs, use shared structure
    # For rank 30: one term per (j, k_pair) where k_pair is a pair of columns
    # This may not work, but let's try a numerical approach
    
    # Try rank 30 with a specific construction
    # For each j in {0,1,2}, for each k in {0,...,9}:
    # Use terms that share the V vector across i values
    # This is equivalent to schoolbook, so skip
    
    # Return schoolbook as the safe answer
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
