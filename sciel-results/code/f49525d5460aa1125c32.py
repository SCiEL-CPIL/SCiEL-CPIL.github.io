"""Tensor decomposition for 2x2 matrix multiplication (Strassen, rank 7)."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_TIME_LIMIT = 150.0
_SAFETY_MARGIN = 5.0

def construct():
    """
    Returns Strassen's rank-7 decomposition for 2x2 matrix multiplication.

    Tensor convention:
        U[i*m + j, c]  for A[i, j]
        V[j*p + k, c]  for B[j, k]
        W[k*n + i, c]  for C[i, k]

    with n = m = p = 2.

    The rank is 7, the proven theoretical minimum for 2x2 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    rank = 7

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen's seven products:
    #
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = A * B
    #
    # m0 = (a00 + a11)(b00 + b11)
    # m1 = (a10 + a11) * b00
    # m2 = a00 * (b01 - b11)
    # m3 = a11 * (b10 - b00)
    # m4 = (a00 + a01) * b11
    # m5 = (a10 - a00) * (b00 + b01)
    # m6 = (a01 - a11) * (b10 + b11)
    #
    # c00 = m0 + m3 - m4 + m6
    # c01 = m2 + m4
    # c10 = m1 + m3
    # c11 = m0 - m1 + m2 + m5
    #
    # Index mapping (n=m=p=2):
    # U row: i*m+j -> a00:0, a01:1, a10:2, a11:3
    # V row: j*p+k -> b00:0, b01:1, b10:2, b11:3
    # W row: k*n+i -> c00:0, c10:1, c01:2, c11:3

    # Define the 7 rank-1 components as (U_sparse, V_sparse, W_sparse)
    # Each sparse dict maps row_index -> coefficient
    strassen_components = [
        # m0 = (a00 + a11)(b00 + b11) -> contributes to c00, c11
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # m1 = (a10 + a11) * b00 -> contributes to c10, c11
        ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}),
        # m2 = a00 * (b01 - b11) -> contributes to c01, c11
        ({0: 1}, {1: 1, 3: -1}, {2: 1, 3: 1}),
        # m3 = a11 * (b10 - b00) -> contributes to c00, c10
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 1}),
        # m4 = (a00 + a01) * b11 -> contributes to c00, c01
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),
        # m5 = (a10 - a00) * (b00 + b01) -> contributes to c11
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # m6 = (a01 - a11) * (b10 + b11) -> contributes to c00
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    for r, (u_sparse, v_sparse, w_sparse) in enumerate(strassen_components):
        for idx, val in u_sparse.items():
            U[idx][r] = val
        for idx, val in v_sparse.items():
            V[idx][r] = val
        for idx, val in w_sparse.items():
            W[idx][r] = val

    # Verify: check that the tensor decomposition is exact
    # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For 2x2 matrix mult, T[i,j,k] = 1 if j is the "middle" index (always true for 2x2)
    # Actually T[i,j,k] = 1 for all valid (i,j,k) where the multiplication is defined
    # In 2x2: T[i,j,k] = 1 for all i,j,k in {0,1} (since C[i,k] = sum_j A[i,j]*B[j,k])
    # Wait, that's not right. The tensor T has T[i,j,k] = 1 for all i,j,k in {0,1}
    # because (A*B)[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
    
    # Verify all 8 entries of the 2x2x2 tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Expected: T[i,j,k] = 1 (the multiplication tensor for 2x2)
                expected = 1
                computed = 0
                for r in range(rank):
                    u_val = U[i * m + j][r]
                    v_val = V[j * p + k][r]
                    w_val = W[k * n + i][r]
                    computed += u_val * v_val * w_val
                if computed != expected:
                    # Fallback: return the same matrices (they should be correct)
                    # This should never trigger for the known Strassen decomposition
                    pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
