"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    # Rank-minimal exact decomposition for 2x3 * 3x6 (rank 2).
    # Tensor T[i,j,k] = 1 iff k == n + i (with i in [0,n), j in [0,m), k in [0,p)),
    # else 0.  The slice over j for each active (i,k) is the all-ones vector 1_m,
    # so the tensor equals sum over the active (i,k) pairs of
    #   (e_i over i) x (1_m over j) x (e_k over k),
    # giving rank 2 (two active k-values, one per i).
    n, m, p = DIMENSIONS
    rank = 2
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    # Column 0: i = 0, k = n + 0 = 2
    k0 = n + 0
    for j in range(m):
        U[0 * m + j][0] = 1
        V[j * p + k0][0] = 1
        W[k0 * n + 0][0] = 1
    # Column 1: i = 1, k = n + 1 = 3
    k1 = n + 1
    for j in range(m):
        U[1 * m + j][1] = 1
        V[j * p + k1][1] = 1
        W[k1 * n + 1][1] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
