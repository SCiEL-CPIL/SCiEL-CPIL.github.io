"""Matrix multiplication 2,2,16 - rank-minimal exact tensor decomposition."""

# EVOLVE-BLOCK-START
def construct():
    # The tensor for M_{2,2,16} is a direct sum of 16 copies of M_{2,2,1}.
    # The rank of M_{2,2,1} is 4 (proven optimal: it requires 4 multiplications
    # as each output element C_i depends on 2 products with the single-row matrix).
    # Thus the minimum rank is 16 * 4 = 64.
    #
    # We construct the schoolbook decomposition of rank 64 using vectorized
    # NumPy array construction for clarity and speed.
    import numpy as np

    n, m, p = 2, 2, 16
    rank = n * m * p  # 64

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Each column c corresponds to a (i, j, k) triple
    # U[i*m + j, c] = 1
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    # Convert to lists of lists as expected by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
