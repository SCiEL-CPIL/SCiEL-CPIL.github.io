"""Kronecker/decoupled decomposition: 6x6x6 = 8 x 3x3x3 blocks."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # A 6x6 matrix A (A[i,j]) with i=3*i0+i1, j=3*j0+j1 can be seen as a 2x2 block
    # matrix of 3x3 blocks: A[a,b] where a=2*i0+i1, b=2*j0+j1, and the (a,b) block
    # is A[3*i0+i1, 3*j0+j1] for i1,j1 in 0..2.
    # (A*B)[a,b] = sum_c A[a,c]*B[c,b]  ->  8 independent 3x3 matmuls.
    #
    # We factor each 3x3 matmul via its schoolbook rank-27 decomposition and pack
    # the 8 blocks into a single rank-8*27 = 216 tensor with decoupled factors.
    #
    # Index conventions (official):
    #   U row = i*m + j  (m=6)
    #   V row = j*p + k  (p=6)
    #   W row = k*n + i  (n=6)
    n = m = p = 6
    N = 3          # block size
    B = n // N     # = 2 blocks per axis
    base_rank = N * N * N  # 27
    rank = B * B * B * base_rank  # 8 * 27 = 216

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i0 in range(B):
        for j0 in range(B):
            for k0 in range(B):
                for i1 in range(N):
                    for j1 in range(N):
                        for k1 in range(N):
                            i = i0 * N + i1
                            j = j0 * N + j1
                            k = k0 * N + k1
                            U[i * m + j][col] = 1
                            V[j * p + k][col] = 1
                            W[k * n + i][col] = 1
                            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
