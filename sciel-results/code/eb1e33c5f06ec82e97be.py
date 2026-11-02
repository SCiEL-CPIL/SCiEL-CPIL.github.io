"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-28 exact tensor decomposition for 2x4 by 4x4 matrix multiplication.
    Strategy: Decompose the problem into four 2x2 by 2x2 sub-problems and apply Strassen's
    algorithm to each, accumulating contributions to the output tensor.
    """
    import numpy as np

    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28  # 4 instances * 7 Strassen multipliers each

    # U: shape (n*m, rank) = (8, 28)
    # V: shape (m*p, rank) = (16, 28)
    # W: shape (p*n, rank) = (8, 28)
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    # Define the four 2x2 sub-problems:
    # Instance 1: A rows 0-1, cols 0-1; B rows 0-1, cols 0-1; C rows 0-1, cols 0-1
    # Instance 2: A rows 0-1, cols 0-1; B rows 0-1, cols 2-3; C rows 0-1, cols 2-3
    # Instance 3: A rows 0-1, cols 2-3; B rows 2-3, cols 0-1; C rows 0-1, cols 0-1
    # Instance 4: A rows 0-1, cols 2-3; B rows 2-3, cols 2-3; C rows 0-1, cols 2-3

    instances = [
        # (a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset)
        (0, 0, 0, 0, 0, 0),   # Instance 1
        (0, 0, 0, 2, 0, 2),   # Instance 2
        (0, 2, 2, 0, 0, 0),   # Instance 3
        (0, 2, 2, 2, 0, 2),   # Instance 4
    ]

    for inst_idx, (a_r, a_c, b_r, b_c, c_r, c_c) in enumerate(instances):
        base_col = inst_idx * 7

        # M0: (a00 + a11) * (b00 + b11) -> c00(+1), c11(+1)
        col = base_col
        # U: a00 at (a_r + 0)*m + (a_c + 0) = 0*4+0=0; a11 at (a_r+1)*m+(a_c+1)=1*4+1=5
        U[a_r * m + a_c, col] += 1.0
        U[(a_r + 1) * m + (a_c + 1), col] += 1.0
        # V: b00 at (b_r+0)*p+(b_c+0)=0*4+0=0; b11 at (b_r+1)*p+(b_c+1)=1*4+1=5
        V[b_r * p + b_c, col] += 1.0
        V[(b_r + 1) * p + (b_c + 1), col] += 1.0
        # W: c00 at (c_c+0)*n+(c_r+0)=0*2+0=0; c11 at (c_c+1)*n+(c_r+1)=1*2+1=3
        W[(c_c + 0) * n + (c_r + 0), col] += 1.0
        W[(c_c + 1) * n + (c_r + 1), col] += 1.0

        # M1: (a10 + a11) * b00 -> c10(+1), c11(-1)
        col = base_col + 1
        # U: a10 at (a_r+1)*m+(a_c+0); a11 at (a_r+1)*m+(a_c+1)
        U[(a_r + 1) * m + a_c, col] += 1.0
        U[(a_r + 1) * m + (a_c + 1), col] += 1.0
        # V: b00 at b_r*p+b_c
        V[b_r * p + b_c, col] += 1.0
        # W: c10 at (c_c+0)*n+(c_r+1); c11 at (c_c+1)*n+(c_r+1)
        W[(c_c + 0) * n + (c_r + 1), col] += 1.0
        W[(c_c + 1) * n + (c_r + 1), col] += -1.0

        # M2: a00 * (b01 - b11) -> c01(+1), c11(+1)
        col = base_col + 2
        # U: a00 at a_r*m+a_c
        U[a_r * m + a_c, col] += 1.0
        # V: b01 at b_r*p+(b_c+1); b11 at (b_r+1)*p+(b_c+1)
        V[b_r * p + (b_c + 1), col] += 1.0
        V[(b_r + 1) * p + (b_c + 1), col] += -1.0
        # W: c01 at (c_c+1)*n+(c_r+0); c11 at (c_c+1)*n+(c_r+1)
        W[(c_c + 1) * n + c_r, col] += 1.0
        W[(c_c + 1) * n + (c_r + 1), col] += 1.0

        # M3: a11 * (b10 - b00) -> c00(+1), c10(+1)
        col = base_col + 3
        # U: a11 at (a_r+1)*m+(a_c+1)
        U[(a_r + 1) * m + (a_c + 1), col] += 1.0
        # V: b10 at (b_r+1)*p+b_c; b00 at b_r*p+b_c
        V[(b_r + 1) * p + b_c, col] += 1.0
        V[b_r * p + b_c, col] += -1.0
        # W: c00 at c_c*n+c_r; c10 at c_c*n+(c_r+1)
        W[c_c * n + c_r, col] += 1.0
        W[c_c * n + (c_r + 1), col] += 1.0

        # M4: (a00 + a01) * b11 -> c00(-1), c01(+1)
        col = base_col + 4
        # U: a00 at a_r*m+a_c; a01 at a_r*m+(a_c+1)
        U[a_r * m + a_c, col] += 1.0
        U[a_r * m + (a_c + 1), col] += 1.0
        # V: b11 at (b_r+1)*p+(b_c+1)
        V[(b_r + 1) * p + (b_c + 1), col] += 1.0
        # W: c00 at c_c*n+c_r; c01 at (c_c+1)*n+c_r
        W[c_c * n + c_r, col] += -1.0
        W[(c_c + 1) * n + c_r, col] += 1.0

        # M5: (a10 - a00) * (b00 + b01) -> c11(+1)
        col = base_col + 5
        # U: a10 at (a_r+1)*m+a_c; a00 at a_r*m+a_c
        U[(a_r + 1) * m + a_c, col] += 1.0
        U[a_r * m + a_c, col] += -1.0
        # V: b00 at b_r*p+b_c; b01 at b_r*p+(b_c+1)
        V[b_r * p + b_c, col] += 1.0
        V[b_r * p + (b_c + 1), col] += 1.0
        # W: c11 at (c_c+1)*n+(c_r+1)
        W[(c_c + 1) * n + (c_r + 1), col] += 1.0

        # M6: (a01 - a11) * (b10 + b11) -> c00(+1)
        col = base_col + 6
        # U: a01 at a_r*m+(a_c+1); a11 at (a_r+1)*m+(a_c+1)
        U[a_r * m + (a_c + 1), col] += 1.0
        U[(a_r + 1) * m + (a_c + 1), col] += -1.0
        # V: b10 at (b_r+1)*p+b_c; b11 at (b_r+1)*p+(b_c+1)
        V[(b_r + 1) * p + b_c, col] += 1.0
        V[(b_r + 1) * p + (b_c + 1), col] += 1.0
        # W: c00 at c_c*n+c_r
        W[c_c * n + c_r, col] += 1.0

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
