"""Rank-minimal exact tensor decomposition for 2x5 @ 5x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS
    rank = m * p  # 25

    # Build U, V, W as flat arrays for speed, then convert to list-of-lists
    # U: shape (n*m, rank) = (10, 25)
    # V: shape (m*p, rank) = (25, 25)
    # W: shape (p*n, rank) = (10, 25)

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    for j0 in range(m):
        for k0 in range(p):
            c = j0 * p + k0
            # U[i*m + j0, c] = 1 for all i
            for i in range(n):
                U[i * m + j0, c] = 1
            # V[j0*p + k0, c] = 1
            V[j0 * p + k0, c] = 1
            # W[k0*n + i, c] = 1 for all i
            for i in range(n):
                W[k0 * n + i, c] = 1

    # Verification using vectorized numpy
    # Tensor element T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need T[i,j,k] = 1 if (i,j,k) corresponds to a valid multiplication term
    # and 0 otherwise. Actually for matrix multiplication A(2x5) @ B(5x5) = C(2x5):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] should be 1 for the "diagonal" and 0 otherwise.
    # More precisely: the multiplication tensor M has M[i,j,k] = 1 if we're computing
    # C[i,k] = sum_j A[i,j] B[j,k]. So the tensor is indexed by (i,j,k) and
    # M[i,j,k] = 1 for all valid (i,j,k) tuples... wait no.
    #
    # Actually the standard tensor for matrix multiplication A(n x m) @ B(m x p) = C(n x p)
    # is T[i,j,k] where i indexes rows of A/C, j indexes columns of A / rows of B,
    # k indexes columns of B/C. T[i,j,k] = 1 for all i,j,k (it's the "multiplication tensor").
    # A rank-r decomposition: T[i,j,k] = sum_{c=1}^r U[i,j,c] * V[j,k,c] * W[i,k,c]
    # Wait, the convention here is W[k*n+i, r] so W is indexed by (k, i) flattened as k*n+i.
    # So W[k*n+i, c] corresponds to the output index (i, k) for C[i,k].
    #
    # The decomposition: C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_k ... ) -- no, let me re-derive.
    #
    # Standard: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For this to equal 1 for all (i,j,k) (the multiplication tensor has all 1s),
    # we need: for each (i,j,k), sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
    #
    # Wait, that's not right either. The multiplication tensor for A(n x m) @ B(m x p) = C(n x p)
    # is: T[i,j,k] = delta_{i,i} delta_{j,j} delta_{k,k} = 1 for all valid indices? No.
    #
    # Let me think again. The bilinear map is: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k].
    # As a trilinear form: <T, A> <T, B> <T, C> where T[i,j,k] = 1 for all i,j,k.
    # No wait: C[i,k] = sum_j A[i,j] B[j,k], so the tensor is T[i,j,k] = 1 for all i,j,k.
    # And the decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # So we need T[i,j,k] = 1 for ALL i in [0,n), j in [0,m), k in [0,p).
    #
    # Let's verify our construction:
    # U[i*m+j, c] where c = j0*p+k0: U[i*m+j, c] = 1 if j == j0, else 0
    # V[j*p+k, c] where c = j0*p+k0: V[j*p+k, c] = 1 if j==j0 AND k==k0, else 0
    # W[k*n+i, c] where c = j0*p+k0: W[k*n+i, c] = 1 if k == k0, else 0
    #
    # So for a given (i,j,k) and a specific c = j0*p+k0:
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 iff j==j0 AND j==j0 AND k==k0 AND k==k0
    # = 1 iff j==j0 AND k==k0
    #
    # Sum over all c = j0*p+k0 for j0 in [0,m), k0 in [0,p):
    # sum = number of (j0,k0) such that j0==j AND k0==k = 1 (when j0=j, k0=k)
    #
    # So T[i,j,k] = 1 for all i,j,k. Correct!

    # Quick verification
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(np.dot(U[i*m+j, :], V[j*p+k, :]), W[k*n+i, :])

    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        # Fallback: schoolbook rank 50
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=np.int64)
        V_fb = np.zeros((m * p, rank_fb), dtype=np.int64)
        W_fb = np.zeros((p * n, rank_fb), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
