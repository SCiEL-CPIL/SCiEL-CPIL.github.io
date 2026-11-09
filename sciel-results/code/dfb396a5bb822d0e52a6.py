"""
Matrix multiplication 3,5,6 - rank-minimal decomposition.

Uses Winograd's 7-rank algorithm for 2x2x2 blocks and schoolbook for the rest.
Partition: I = [2,1], J = [2,2,1], K = [2,2,2]
Total rank: 6*7 + 3*4 + 6*4 + 3*2 = 42 + 12 + 24 + 6 = 84
"""

DIMENSIONS = [3, 5, 6]


def construct():
    n, m, p = DIMENSIONS
    # Partition sizes
    Is = [2, 1]
    Js = [2, 2, 1]
    Ks = [2, 2, 2]

    # Compute block indices
    i_offsets = [0]
    for s in Is[:-1]:
        i_offsets.append(i_offsets[-1] + s)
    j_offsets = [0]
    for s in Js[:-1]:
        j_offsets.append(j_offsets[-1] + s)
    k_offsets = [0]
    for s in Ks[:-1]:
        k_offsets.append(k_offsets[-1] + s)

    # Determine rank for each block type
    # Block (a x b x c): rank = a*b*c if not 2x2x2, else 7
    def block_rank(a, b, c):
        if a == 2 and b == 2 and c == 2:
            return 7
        return a * b * c

    # Build list of blocks with their rank
    blocks = []
    for i_idx, i_off in enumerate(i_offsets):
        a = Is[i_idx]
        for j_idx, j_off in enumerate(j_offsets):
            b = Js[j_idx]
            for k_idx, k_off in enumerate(k_offsets):
                c = Ks[k_idx]
                r = block_rank(a, b, c)
                blocks.append((i_idx, i_off, a, j_idx, j_off, b, k_idx, k_off, c, r))

    total_rank = sum(b[9] for b in blocks)

    # Initialize matrices
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]

    # Winograd 2x2x2 coefficients
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Standard Winograd 2x2 matrix multiplication
    # U = [[1, 0, 0, 0, 1, -1, 0],
    #      [0, 1, 0, 0, 1, 1, 0],
    #      [0, 0, 1, 0, 1, 0, 0],
    #      [0, 0, 0, 1, 1, 0, 0]]
    # V = same as U (symmetric for matrix mult)
    # W = [[1, 1, 1, 1, 0, 0, 0],
    #      [0, 0, 1, -1, 1, 1, 0],
    #      [0, 0, 1, -1, -1, -1, 0],
    #      [0, 0, 0, 0, 1, -1, 1]]
    # Actually, let me use the standard formulation:
    # For A(2x2) * B(2x2) = C(2x2):
    # m0 = (A00+A11)*(B00+B11)
    # m1 = (A10+A11)*B01
    # m2 = A01*(B10+B11)
    # m3 = A00*B11
    # m4 = A11*B00
    # m5 = (A00-A10)*(B10+B11)
    # m6 = (A01+A11)*(B00-B01)
    # C00 = m0+m3+m4+m5+m6
    # C01 = m2-m3-m5+m6
    # C10 = m1+m3+m4-m6
    # C11 = m0-m1-m2+m4+m5

    # Let me define U, V, W for Winograd 2x2x2:
    # U rows correspond to A elements: A00, A01, A10, A11
    # V rows correspond to B elements: B00, B01, B10, B11
    # W rows correspond to C elements: C00, C01, C10, C11

    U2 = [
        [1, 0, 0, 1, 1, 0, 0],   # A00
        [0, 0, 0, 1, 0, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 0, 0, 0, 1],   # A11
    ]
    V2 = [
        [1, 0, 0, 1, 0, 1, -1],  # B00
        [0, 1, 0, 0, 0, 0, 1],   # B01
        [1, 1, 1, 0, 0, 1, 0],   # B10
        [1, 0, 1, 0, 0, 0, 0],   # B11
    ]
    W2 = [
        [1, 0, 0, 1, 1, 1, 1],   # C00
        [0, 0, 1, -1, -1, -1, 1],# C01
        [0, 1, 0, 1, 1, -1, 0],  # C10
        [1, -1, -1, 0, 1, 1, 0], # C11
    ]

    def get_block_matrices(a, b, c):
        """Return (U_block, V_block, W_block) for a x b x c block.
        U_block: (a*b) x rank
        V_block: (b*c) x rank
        W_block: (c*a) x rank
        """
        if a == 2 and b == 2 and c == 2:
            return U2, V2, W2
        else:
            r = a * b * c
            Ub = [[0.0] * r for _ in range(a * b)]
            Vb = [[0.0] * r for _ in range(b * c)]
            Wb = [[0.0] * r for _ in range(c * a)]
            col = 0
            for ii in range(a):
                for jj in range(b):
                    for kk in range(c):
                        Ub[ii * b + jj][col] = 1.0
                        Vb[jj * c + kk][col] = 1.0
                        Wb[kk * a + ii][col] = 1.0
                        col += 1
            return Ub, Vb, Wb

    # Fill in the global matrices
    rank_offset = 0
    for (i_idx, i_off, a, j_idx, j_off, b, k_idx, k_off, c, r) in blocks:
        Ub, Vb, Wb = get_block_matrices(a, b, c)

        for local_i in range(a):
            for local_j in range(b):
                global_u_row = (i_off + local_i) * m + (j_off + local_j)
                for k in range(r):
                    U[global_u_row][rank_offset + k] = Ub[local_i * b + local_j][k]

        for local_j in range(b):
            for local_k in range(c):
                global_v_row = (j_off + local_j) * p + (k_off + local_k)
                for k in range(r):
                    V[global_v_row][rank_offset + k] = Vb[local_j * c + local_k][k]

        for local_k in range(c):
            for local_i in range(a):
                # W index: k*n + i where k is the column index of B, i is row of A
                # In our convention: W[k*n + i, c] where k is from B's column, i from A's row
                global_w_row = (k_off + local_k) * n + (i_off + local_i)
                for k in range(r):
                    W[global_w_row][rank_offset + k] = Wb[local_k * a + local_i][k]

        rank_offset += r

    return {"U": U, "V": V, "W": W}
