"""Program for matrix multiplication tensor decomposition: 2x2 by 2x16.
Uses rank-56 Strassen-based decomposition: 8 independent 2-wide column blocks,
each handled by the 7-multiplication Strassen scheme for 2x2 * 2x2."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    NB = p // 2  # 8 blocks
    R = 7 * NB   # rank 56

    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]

    # Strassen factors for C = A(2x2) * B(2x2)
    # U: entries (i, j, coeff) -> A[i,j]
    # V: entries (j, k, coeff) -> B[j,k]  (k local 0..1)
    # W: entries (k, i, coeff) -> C[i,k]
    S = [
        ([(0,0,1),(1,1,1)],   [(0,0,1),(1,1,1)],   [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)],   [(0,0,1)],           [(0,1,1),(1,1,-1)]),
        ([(0,0,1)],           [(0,1,1),(1,1,-1)],  [(0,0,-1),(1,0,1)]),
        ([(1,1,1)],           [(1,0,1),(0,0,-1)],  [(0,0,1),(0,1,1)]),
        ([(0,0,1),(0,1,1)],   [(1,1,1)],           [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)],  [(0,0,1),(0,1,1)],   [(1,1,1)]),
        ([(0,1,1),(1,1,-1)],  [(1,0,1),(1,1,1)],   [(0,0,1)]),
    ]

    for t in range(NB):
        ks = 2 * t
        for s in range(7):
            c = t * 7 + s
            for i, j, a in S[s][0]:
                U[i * m + j][c] = a
            for j, k, b in S[s][1]:
                V[j * p + ks + k][c] = b
            for k, i, w in S[s][2]:
                W[(ks + k) * n + i][c] = w

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
