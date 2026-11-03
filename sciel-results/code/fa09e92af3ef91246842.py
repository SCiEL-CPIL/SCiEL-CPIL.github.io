"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    # Rank-21 decomposition exploiting the separability of the 3x3x7 tensor.
    n, m, p = DIMENSIONS  # 3, 3, 7
    rank = n * p  # 21

    # U: shape (n*m, rank) = (9, 21)
    # U[i*m + j, r] where r = k*n + i_out
    # U[3*i' + j, 3*k + i_out] = 1 if i' == i_out else 0
    U = [[0] * rank for _ in range(n * m)]
    for i_prime in range(n):
        for j in range(m):
            for i_out in range(n):
                if i_prime == i_out:
                    # For each k, r = k*n + i_out
                    for k in range(p):
                        r = k * n + i_out
                        U[i_prime * m + j][r] = 1

    # V: shape (m*p, rank) = (21, 21)
    # V[j*p + k', r] where r = k*n + i_out
    # V[7*j + k', 3*k + i_out] = 1 if k == k' else 0
    V = [[0] * rank for _ in range(m * p)]
    for j in range(m):
        for k_prime in range(p):
            for k in range(p):
                if k == k_prime:
                    for i_out in range(n):
                        r = k * n + i_out
                        V[j * p + k_prime][r] = 1

    # W: shape (p*n, rank) = (21, 21)
    # W[k*n + i, r] where r = k'*n + i'_out
    # W[3*k + i, 3*k' + i'_out] = 1 if k == k' and i == i'_out else 0
    # This is simply the identity matrix.
    W = [[0] * rank for _ in range(p * n)]
    for idx in range(rank):
        W[idx][idx] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
