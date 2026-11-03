"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s limit

    n, m, p = DIMENSIONS

    # Rank-44 block-partitioned decomposition:
    # Split p=8 into two p=4 blocks.
    # For each p=4 block:
    #   - Split m=3 into m=2 (rows 0,1) and m=1 (row 2)
    #   - m=2, p=4: split p=4 into two p=2 blocks, each Strassen 2x2x2 (rank 7) -> rank 14
    #   - m=1, p=4: schoolbook 2x1x4 -> rank 8
    #   - Total per p=4 block: 22
    # Total rank: 44

    total_rank = 44

    U = [[0] * total_rank for _ in range(n * m)]  # shape (6, 44)
    V = [[0] * total_rank for _ in range(m * p)]  # shape (24, 44)
    W = [[0] * total_rank for _ in range(p * n)]  # shape (16, 44)

    col = 0

    def _add_strassen_block(i_off, j_off, k_off):
        """Add a Strassen 2x2x2 block at given offsets."""
        nonlocal col
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

    def _add_schoolbook_2x1x4(i_off, j_off, k_off):
        """Add schoolbook terms for 2x1x4 block."""
        nonlocal col
        for i in range(2):
            for k in range(4):
                U[(i_off+i)*m + (j_off)][col] = 1
                V[(j_off)*p + (k_off+k)][col] = 1
                W[(k_off+k)*n + (i_off+i)][col] = 1
                col += 1

    # Block 1: k in 0..3 (p offset 0)
    _add_strassen_block(0, 0, 0)   # Strassen for m rows 0-1, p cols 0-1
    _add_strassen_block(0, 0, 2)   # Strassen for m rows 0-1, p cols 2-3
    _add_schoolbook_2x1x4(0, 2, 0) # Schoolbook for m row 2, p cols 0-3

    # Block 2: k in 4..7 (p offset 4)
    _add_strassen_block(0, 0, 4)   # Strassen for m rows 0-1, p cols 4-5
    _add_strassen_block(0, 0, 6)   # Strassen for m rows 0-1, p cols 6-7
    _add_schoolbook_2x1x4(0, 2, 4) # Schoolbook for m row 2, p cols 4-7

    if time.time() - start_time > DEADLINE:
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
