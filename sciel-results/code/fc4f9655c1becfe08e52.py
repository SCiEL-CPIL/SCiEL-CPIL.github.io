"""Matrix multiplication 4x4x4 via nested Strassen (rank 49)."""
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Outer Strassen products for 4x4 (split into 2x2 blocks)
    outer_products = [
        # M1 = (A10 + A11) * (B00 + B01)
        ([(1, (1, 0)), (1, (1, 1))],
         [(1, (0, 0)), (1, (0, 1))],
         [(1, (0, 0)), (-1, (1, 1))]),
        # M2 = (A00 + A11) * B10
        ([(1, (0, 0)), (1, (1, 1))],
         [(1, (1, 0))],
         [(1, (1, 0)), (-1, (1, 1))]),
        # M3 = A00 * (B01 - B11)
        ([(1, (0, 0))],
         [(1, (0, 1)), (-1, (1, 1))],
         [(1, (0, 1)), (1, (1, 1))]),
        # M4 = A11 * (B10 - B00)
        ([(1, (1, 1))],
         [(1, (1, 0)), (-1, (0, 0))],
         [(1, (0, 0)), (1, (1, 0))]),
        # M5 = (A00 + A01) * B11
        ([(1, (0, 0)), (1, (0, 1))],
         [(1, (1, 1))],
         [(-1, (0, 0))]),
        # M6 = (A10 - A00) * (B00 + B10)
        ([(1, (1, 0)), (-1, (0, 0))],
         [(1, (0, 0)), (1, (1, 0))],
         [(1, (1, 1))]),
        # M7 = (A01 - A11) * (B01 + B11)
        ([(1, (0, 1)), (-1, (1, 1))],
         [(1, (0, 1)), (1, (1, 1))],
         [(1, (0, 0))]),
    ]

    # Inner Strassen products for 2x2 (scalar products)
    inner_products = [
        # S1 = (X10 + X11) * (Y00 + Y01)
        ([(1, (1, 0)), (1, (1, 1))],
         [(1, (0, 0)), (1, (0, 1))],
         [(1, (0, 0)), (-1, (1, 1))]),
        # S2 = (X00 + X11) * Y10
        ([(1, (0, 0)), (1, (1, 1))],
         [(1, (1, 0))],
         [(1, (1, 0)), (-1, (1, 1))]),
        # S3 = X00 * (Y01 - Y11)
        ([(1, (0, 0))],
         [(1, (0, 1)), (-1, (1, 1))],
         [(1, (0, 1)), (1, (1, 1))]),
        # S4 = X11 * (Y10 - Y00)
        ([(1, (1, 1))],
         [(1, (1, 0)), (-1, (0, 0))],
         [(1, (0, 0)), (1, (1, 0))]),
        # S5 = (X00 + X01) * Y11
        ([(1, (0, 0)), (1, (0, 1))],
         [(1, (1, 1))],
         [(-1, (0, 0))]),
        # S6 = (X10 - X00) * (Y00 + Y10)
        ([(1, (1, 0)), (-1, (0, 0))],
         [(1, (0, 0)), (1, (1, 0))],
         [(1, (1, 1))]),
        # S7 = (X01 - X11) * (Y01 + Y11)
        ([(1, (0, 1)), (-1, (1, 1))],
         [(1, (0, 1)), (1, (1, 1))],
         [(1, (0, 0))]),
    ]

    U_rows = []
    V_rows = []
    W_rows = []

    for Ublocks_outer, Vblocks_outer, Cblocks_outer in outer_products:
        if time.time() > deadline:
            break
        for Ublocks_inner, Vblocks_inner, Cblocks_inner in inner_products:
            if time.time() > deadline:
                break
            col_U = [0] * (n * m)
            col_V = [0] * (m * p)
            col_W = [0] * (p * n)

            # Build U factor: A entries
            for si, (ir, ic) in Ublocks_inner:
                for su, (ub_r, ub_c) in Ublocks_outer:
                    gi = 2 * ub_r + ir
                    gj = 2 * ub_c + ic
                    col_U[gi * m + gj] += si * su

            # Build V factor: B entries
            for si, (ir, ic) in Vblocks_inner:
                for sv, (vb_r, vb_c) in Vblocks_outer:
                    gi = 2 * vb_r + ir
                    gj = 2 * vb_c + ic
                    col_V[gi * p + gj] += si * sv

            # Build W factor: C entries (trace convention W[k*n+i])
            for si, (ir, ic) in Cblocks_inner:
                for sw, (wb_r, wb_c) in Cblocks_outer:
                    gi = 2 * wb_r + ir
                    gj = 2 * wb_c + ic
                    col_W[gi * n + gj] += si * sw

            U_rows.append(col_U)
            V_rows.append(col_V)
            W_rows.append(col_W)

    return {"U": U_rows, "V": V_rows, "W": W_rows}
# EVOLVE-BLOCK-END
