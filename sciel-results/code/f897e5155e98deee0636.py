"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = DIMENSIONS  # 3, 5, 5

    # Winograd 3x3x3 tensor decomposition (rank 23)
    # A is 3x3, B is 3x3, C = A*B is 3x3
    # U: 9x23 (rows correspond to a00,a01,a02,a10,a11,a12,a20,a21,a22)
    # V: 9x23 (rows correspond to b00,b01,b02,b10,b11,b12,b20,b21,b22)
    # W: 9x23 (rows correspond to c00,c01,c02,c10,c11,c12,c20,c21,c22)

    # The 23 rank-1 terms of the Winograd 3x3x3 scheme
    # Using the standard Winograd 3x3x3 factorization
    # Preprocessing A:
    # d0 = a00+a01+a02, d1 = a00-a01, d2 = a02-a01, d3 = a10+a11+a12
    # d4 = a10-a11, d5 = a12-a11, d6 = a20+a21+a22, d7 = a20-a21, d8 = a22-a21
    # Preprocessing B:
    # e0 = b00+b10+b20, e1 = b01+b11+b21, e2 = b02+b12+b22
    # e3 = b00-b01, e4 = b02-b01, e5 = b10-b11, e6 = b12-b11, e7 = b20-b21, e8 = b22-b21

    # Multiplications (23 total):
    # z0 = d0*e0, z1 = d1*e1, z2 = d2*e1, z3 = d3*e0, z4 = d4*e1, z5 = d5*e1
    # z6 = d6*e0, z7 = d7*e1, z8 = d8*e1, z9 = d0*e3, z10 = d1*e3, z11 = d2*e3
    # z12 = d3*e3, z13 = d4*e3, z14 = d5*e3, z15 = d6*e3, z16 = d7*e3, z17 = d8*e3
    # z18 = d0*e5, z19 = d1*e5, z20 = d2*e5, z21 = d3*e5, z22 = d4*e5

    # Let me construct the Winograd 3x3x3 factorization properly.
    # The 23 rank-1 tensors are defined by their U, V, W vectors.

    # I'll use a known explicit factorization from the literature.
    # After careful construction, here's a valid rank-23 Winograd 3x3x3:

    # The 23 terms are constructed as follows:
    # Terms 0-2: diagonal terms for first row of A with first column of B
    # Terms 3-5: cross terms using Winograd's additions
    # ... and so on

    # Let me just build a valid decomposition programmatically using the known scheme.

    # Actually, let me use a simpler approach: partition into 3x3x3 and handle
    # the remaining terms with schoolbook.

    # Partition: n=3 (one 3-block), m=5 (one 3-block + one 2-block), p=5 (one 3-block + one 2-block)
    # Blocks: (3,3,3), (3,3,2), (3,2,3), (3,2,2)

    # For 3x3x3: use Winograd (rank 23) vs schoolbook (rank 27) -> save 4
    # For 3x3x2: can we do better than schoolbook (rank 18)? 
    # For 3x2x3: same as above
    # For 3x2x2: schoolbook (rank 12)

    # Total with Winograd 3x3x3: 23 + 18 + 18 + 12 = 71
    # Total with all schoolbook: 27 + 18 + 18 + 12 = 75

    # So Winograd 3x3x3 gives 71, same as the Strassen-based approach.

    # Let me try to construct the Winograd 3x3x3 factorization.
    # I'll use the standard scheme from Winograd (1968).

    # The 23 rank-1 terms:
    # I'll define them explicitly.

    # Term 0: U=[1,0,0,0,0,0,0,0,0], V=[1,0,0,0,0,0,0,0,0], W=[1,0,0,0,0,0,0,0,0]
    # Term 1: U=[0,1,0,0,0,0,0,0,0], V=[0,1,0,0,0,0,0,0,0], W=[0,1,0,0,0,0,0,0,0]
    # ... this is getting too complex to do by hand.

    # Let me use the Strassen-based approach which gives rank 71, and verify it works.

    # Using partition n=[2,1], m=[2,2,1], p=[2,2,1] with 4 Strassen 2x2x2 blocks.
    # Rank = 4*7 + 43 = 71.

    def build_decomposition():
        """Build rank-71 decomposition using 4 Strassen 2x2x2 blocks + schoolbook."""
        rank_total = 71
        U = [[0] * rank_total for _ in range(n * m)]
        V = [[0] * rank_total for _ in range(m * p)]
        W = [[0] * rank_total for _ in range(p * n)]

        col = 0
        # 4 Strassen 2x2x2 blocks: i in {0,1}, j in {0,1} or {2,3}, k in {0,1} or {2,3}
        blocks = [
            ([0, 1], [0, 1], [0, 1]),
            ([0, 1], [0, 1], [2, 3]),
            ([0, 1], [2, 3], [0, 1]),
            ([0, 1], [2, 3], [2, 3]),
        ]

        covered = set()
        for i_idx, j_idx, k_idx in blocks:
            for i in i_idx:
                for j in j_idx:
                    for k in k_idx:
                        covered.add((i, j, k))

            i0, i1 = i_idx
            j0, j1 = j_idx
            k0, k1 = k_idx

            u_rows = [i0 * m + j0, i0 * m + j1, i1 * m + j0, i1 * m + j1]
            v_rows = [j0 * p + k0, j0 * p + k1, j1 * p + k0, j1 * p + k1]
            w_rows = [k0 * n + i0, k1 * n + i0, k0 * n + i1, k1 * n + i1]

            # Strassen 2x2x2 matrices
            U_strassen = [
                [1, 0, 1, 0, 1, -1, 0],
                [0, 0, 0, 0, 1, 0, 1],
                [0, 1, 0, 0, 0, 1, 0],
                [1, 1, 0, 1, 0, 0, -1]
            ]
            V_strassen = [
                [1, 1, 0, -1, 0, 1, 0],
                [0, 0, 1, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 1],
                [1, 0, -1, 0, 1, 0, 1]
            ]
            W_strassen = [
                [1, 0, 0, 1, -1, 0, 1],
                [0, 0, 1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0, 0, 0],
                [1, -1, 1, 0, 0, 1, 0]
            ]

            for c in range(7):
                for s in range(4):
                    U[u_rows[s]][col + c] = U_strassen[s][c]
                    V[v_rows[s]][col + c] = V_strassen[s][c]
                    W[w_rows[s]][col + c] = W_strassen[s][c]
            col += 7

        # Schoolbook for remaining 43 terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in covered:
                        continue
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

        assert col == rank_total, f"Expected rank {rank_total}, got {col}"
        return U, V, W

    U, V, W = build_decomposition()

    # Verify using numpy for speed
    def verify_fast(U_mat, V_mat, W_mat):
        U_np = np.array(U_mat, dtype=float)
        V_np = np.array(V_mat, dtype=float)
        W_np = np.array(W_mat, dtype=float)
        rank = U_np.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True

    if not verify_fast(U, V, W):
        # Fallback to schoolbook
        rank_school = n * m * p
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
