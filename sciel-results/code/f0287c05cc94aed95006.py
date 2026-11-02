"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    # Block-partitioned strategy:
    # Split the 3x8 input B into two 3x4 blocks.
    # For each 3x4 block, split the 3 rows into a 2-row part and a 1-row part.
    # The 2x4 part is further split into two 2x2 blocks (each using Strassen, rank 7).
    # The 1x4 part uses schoolbook (rank 4).
    # Total rank per 3x4 block: 7 + 7 + 4 = 18.
    # Total rank for 3x8: 18 + 18 = 36.

    rank = 36

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_2x2x2(i_off, j_off, k_off):
        nonlocal col
        # U indices: (i_off+li)*m + (j_off+lj) for li in 0,1 and lj in 0,1
        # V indices: (j_off+lj)*p + (k_off+lk) for lj in 0,1 and lk in 0,1
        # W indices: (k_off+lk)*n + (i_off+li) for lk in 0,1 and li in 0,1

        # m1: (a+d)(e+h)
        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1

        # m2: (c+d)e
        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        W[(k_off+0)*n + (i_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += -1
        col += 1

        # m3: a(f-h)
        U[(i_off+0)*m + (j_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += -1
        W[(k_off+1)*n + (i_off+0)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1

        # m4: d(g-e)
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += -1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        W[(k_off+0)*n + (i_off+1)][col] += 1
        col += 1

        # m5: (a+b)h
        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += -1
        W[(k_off+1)*n + (i_off+0)][col] += 1
        col += 1

        # m6: (c-a)(e+f)
        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+0)][col] += -1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1

        # m7: (b-d)(g+h)
        U[(i_off+0)*m + (j_off+1)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += -1
        V[(j_off+1)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        col += 1

    def add_schoolbook_2x1x4(i_off, j_off, k_off):
        nonlocal col
        for i in range(2):
            for k in range(4):
                U[(i_off+i)*m + (j_off+0)][col] = 1
                V[(j_off+0)*p + (k_off+k)][col] = 1
                W[(k_off+k)*n + (i_off+i)][col] = 1
                col += 1

    # Block 1: k in 0..3
    # Strassen for j in 0..1, k in 0..1
    add_strassen_2x2x2(0, 0, 0)
    # Strassen for j in 0..1, k in 2..3
    add_strassen_2x2x2(0, 0, 2)
    # Schoolbook for j=2, k in 0..3
    add_schoolbook_2x1x4(0, 2, 0)

    # Block 2: k in 4..7
    # Strassen for j in 0..1, k in 4..5
    add_strassen_2x2x2(0, 0, 4)
    # Strassen for j in 0..1, k in 6..7
    add_strassen_2x2x2(0, 0, 6)
    # Schoolbook for j=2, k in 4..7
    add_schoolbook_2x1x4(0, 2, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
