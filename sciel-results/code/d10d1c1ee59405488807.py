# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    rank = n * m * p  # 72

    # Generate index triples (i, j, k) corresponding to the rank-1 components
    # of the tensor T = sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{i,k}
    # This corresponds to the direct sum of n*m*1 tensors for each k.
    indices = []
    for k in range(p):
        for i in range(n):
            for j in range(m):
                indices.append((i, j, k))
                
    rank = len(indices)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    for col, (i, j, k) in enumerate(indices):
        U[i * m + j, col] = 1
        V[j * p + k, col] = 1
        W[k * n + i, col] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
