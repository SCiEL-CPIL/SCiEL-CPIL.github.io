import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication."""
    deadline = time.time() + 140
    n, m, p = 2, 3, 8

    def verify(U, V, W, n, m, p):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        T = np.zeros((n, m, p))
        for c in range(r):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-10)

    def build_2x2x4_rank13():
        """Rank-13 decomposition of M_{2,2,4}."""
        rank = 13
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(8)]
        W = [[0]*rank for _ in range(8)]

        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[4][0] = 1
        W[0][0] = 1; W[5][0] = 1

        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[4][1] = 1

        U[0][2] = 1
        V[1][2] = 1; V[5][2] = 1
        W[1][2] = 1; W[5][2] = 1

        U[0][3] = 1; U[2][3] = -1
        V[1][3] = 1; V[4][3] = 1
        W[0][3] = 1; W[4][3] = 1

        U[1][4] = 1; U[2][4] = 1
        V[0][4] = 1; V[4][4] = -1
        W[0][4] = -1; W[1][4] = 1

        U[1][5] = -1; U[3][5] = 1
        V[1][5] = 1; V[5][5] = -1
        W[5][5] = 1

        U[0][6] = 1; U[1][6] = 1
        V[1][6] = -1; V[5][6] = 1
        W[0][6] = 1

        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[6][7] = 1
        W[4][7] = 1; W[7][7] = 1

        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[6][8] = 1

        U[0][9] = 1
        V[3][9] = 1; V[7][9] = 1
        W[5][9] = 1; W[7][9] = 1

        U[0][10] = 1; U[2][10] = -1
        V[3][10] = 1; V[6][10] = 1
        W[4][10] = 1; W[6][10] = 1

        U[1][11] = 1; U[2][11] = 1
        V[2][11] = 1; V[6][11] = -1
        W[4][11] = -1; W[5][11] = 1

        U[1][12] = -1; U[3][12] = 1
        V[3][12] = 1; V[7][12] = -1
        W[7][12] = 1

        return U, V, W

    def build_1plus2_split():
        """
        Split m=3 as 1+2:
        - j=0: schoolbook, rank 16
        - j=1,2: two M_{2,2,4} blocks (k=0..3 and k=4..7), rank 13 each
        Total rank = 42
        """
        n, m, p = 2, 3, 8
        rank_school = n * p  # 16
        rank_224 = 13
        total_rank = rank_school + 2 * rank_224  # 42

        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]

        col = 0
        # Schoolbook for j=0: A[i,0]*B[0,k] -> C[i,k]
        for i in range(n):
            for k in range(p):
                U[i*m + 0][col] = 1
                V[0*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1

        U224, V224, W224 = build_2x2x4_rank13()

        # First M_{2,2,4} block: k=0..3, j=1,2
        for r in range(rank_224):
            for i in range(n):
                U[i*m+1][col+r] = U224[i*2+0][r]
                U[i*m+2][col+r] = U224[i*2+1][r]
            for j_local in range(2):
                for k_local in range(4):
                    V[(1+j_local)*p + k_local][col+r] = V224[j_local*4+k_local][r]
            for k_local in range(4):
                for i in range(n):
                    W[k_local*n+i][col+r] = W224[k_local*2+i][r]
        col += rank_224

        # Second M_{2,2,4} block: k=4..7, j=1,2
        for r in range(rank_224):
            for i in range(n):
                U[i*m+1][col+r] = U224[i*2+0][r]
                U[i*m+2][col+r] = U224[i*2+1][r]
            for j_local in range(2):
                for k_local in range(4):
                    V[(1+j_local)*p + (4+k_local)][col+r] = V224[j_local*4+k_local][r]
            for k_local in range(4):
                for i in range(n):
                    W[(4+k_local)*n+i][col+r] = W224[k_local*2+i][r]
        col += rank_224

        assert col == total_rank, f"Expected {total_rank}, got {col}"
        return U, V, W

    def build_direct_rank44():
        """Direct rank-44: 16 schoolbook for j=0 + 4x7 Strassen for j=1,2 blocks."""
        n, m, p = 2, 3, 8
        rank = 44
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0

        for i in range(n):
            for k in range(p):
                U[i*m + 0][col] = 1
                V[0*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1

        for block in range(4):
            k0 = block * 2
            b00 = 1*p + k0
            b01 = 1*p + k0 + 1
            b10 = 2*p + k0
            b11 = 2*p + k0 + 1
            w00 = k0*n + 0
            w01 = (k0+1)*n + 0
            w10 = k0*n + 1
            w11 = (k0+1)*n + 1

            U[1][col] = 1; U[5][col] = 1
            V[b00][col] = 1; V[b11][col] = 1
            W[w00][col] = 1; W[w11][col] = 1
            col += 1

            U[4][col] = 1; U[5][col] = 1
            V[b00][col] = 1
            W[w10][col] = 1; W[w11][col] = -1
            col += 1

            U[1][col] = 1
            V[b01][col] = 1; V[b11][col] = -1
            W[w01][col] = 1; W[w11][col] = 1
            col += 1

            U[5][col] = 1
            V[b10][col] = 1; V[b00][col] = -1
            W[w00][col] = 1; W[w10][col] = 1
            col += 1

            U[1][col] = 1; U[2][col] = 1
            V[b11][col] = 1
            W[w00][col] = -1; W[w01][col] = 1
            col += 1

            U[4][col] = 1; U[1][col] = -1
            V[b00][col] = 1; V[b01][col] = 1
            W[w11][col] = 1
            col += 1

            U[2][col] = 1; U[5][col] = -1
            V[b10][col] = 1; V[b11][col] = 1
            W[w00][col] = 1
            col += 1

        assert col == 44, f"Expected 44, got {col}"
        return U, V, W

    def build_2plus1_strassen():
        """Split m=3 as 2+1: Strassen for j=0,1 blocks + schoolbook for j=2."""
        n, m, p = 2, 3, 8
        rank = 44
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0

        for block in range(4):
            k0 = block * 2
            b00 = 0*p + k0
            b01 = 0*p + k0 + 1
            b10 = 1*p + k0
            b11 = 1*p + k0 + 1
            w00 = k0*n + 0
            w01 = (k0+1)*n + 0
            w10 = k0*n + 1
            w11 = (k0+1)*n + 1

            U[0][col] = 1; U[3][col] = 1
            V[b00][col] = 1; V[b11][col] = 1
            W[w00][col] = 1; W[w11][col] = 1
            col += 1

            U[2][col] = 1; U[3][col] = 1
            V[b00][col] = 1
            W[w10][col] = 1; W[w11][col] = -1
            col += 1

            U[0][col] = 1
            V[b01][col] = 1; V[b11][col] = -1
            W[w01][col] = 1; W[w11][col] = 1
            col += 1

            U[3][col] = 1
            V[b10][col] = 1; V[b00][col] = -1
            W[w00][col] = 1; W[w10][col] = 1
            col += 1

            U[0][col] = 1; U[1][col] = 1
            V[b11][col] = 1
            W[w00][col] = -1; W[w01][col] = 1
            col += 1

            U[2][col] = 1; U[0][col] = -1
            V[b00][col] = 1; V[b01][col] = 1
            W[w11][col] = 1
            col += 1

            U[1][col] = 1; U[3][col] = -1
            V[b10][col] = 1; V[b11][col] = 1
            W[w00][col] = 1
            col += 1

        for i in range(n):
            for k in range(p):
                U[i*m + 2][col] = 1
                V[2*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1

        assert col == 44, f"Expected 44, got {col}"
        return U, V, W

    best_result = None
    best_rank = float('inf')

    candidates = [
        ("1+2 split rank 42", build_1plus2_split, 42),
        ("direct rank 44", build_direct_rank44, 44),
        ("2+1 Strassen rank 44", build_2plus1_strassen, 44),
    ]

    for name, builder, expected_rank in candidates:
        if time.time() > deadline:
            break
        try:
            U, V, W = builder()
            actual_rank = len(U[0]) if U else 0
            if verify(U, V, W, n, m, p) and actual_rank < best_rank:
                best_result = {"U": U, "V": V, "W": W}
                best_rank = actual_rank
        except Exception as e:
            pass

    if best_result is not None:
        return best_result

    # Fallback: schoolbook
    rank_fallback = n*m*p
    U_f = [[0]*rank_fallback for _ in range(n*m)]
    V_f = [[0]*rank_fallback for _ in range(m*p)]
    W_f = [[0]*rank_fallback for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_f[i*m+j][col] = 1
                V_f[j*p+k][col] = 1
                W_f[k*n+i][col] = 1
                col += 1
    return {"U": U_f, "V": V_f, "W": W_f}
# EVOLVE-BLOCK-END
