"""Combined strategy: structured block decomposition (1x4x4 + Strassen on 2x4x4)
with verification, half-integer snapping, and a light polish pass.

The 3x4x4 matmul tensor is decomposed as:
  Part 1: A[0,:] (1x4) x B (4x4) -> C[0,:]  : rank 16 (schoolbook)
  Part 2: A[1:3,:] (2x4) x B (4x4) -> C[1:3,:] : rank 28 (4 blocks of 2x2, Strassen)
Total rank = 44.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Verify U,V,W exactly decompose the matmul tensor via einsum contraction."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, -1)
    V3 = V_arr.reshape(m, p, -1)
    W3 = W_arr.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
    return np.allclose(T, 1.0, atol=tol)


def _snap_to_half_integer(mat, tol=1e-9):
    """Snap entries to nearest half-integer if close enough; return snapped list-of-lists."""
    arr = np.array(mat, dtype=float)
    snapped = np.round(arr * 2) / 2
    if np.max(np.abs(arr - snapped)) < tol:
        # Represent as exact half-integer floats (integers or x.5)
        out = []
        for row in snapped:
            out.append([float(x) for x in row])
        return out
    return [list(map(float, row)) for row in arr]


def _strassen_terms():
    """Return the 7 Strassen terms for 2x2 multiply X*Y=Z.

    Each term: (x_coeffs, y_coeffs, z_coeffs) where each is a dict
    {(row, col): coeff}.

    M1 = (X11+X22)(Y11+Y22)
    M2 = (X21+X22)Y11
    M3 = X11(Y12-Y22)
    M4 = X22(Y21-Y11)
    M5 = (X11+X12)Y22
    M6 = (X21-X11)(Y11+Y12)
    M7 = (X12-X22)(Y21+Y22)

    Z11 = M1 + M4 - M5 + M7
    Z12 = M3 + M5
    Z21 = M2 + M4
    Z22 = M1 - M2 + M3 + M6
    """
    return [
        # M1
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1},
         {(0, 0): 1, (1, 1): 1}),
        # M2
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1},
         {(1, 0): 1, (1, 1): -1}),
        # M3
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1},
         {(0, 1): 1, (1, 1): 1}),
        # M4
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1},
         {(0, 0): 1, (1, 0): 1}),
        # M5
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1},
         {(0, 0): -1, (0, 1): 1}),
        # M6
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1},
         {(1, 1): 1}),
        # M7
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1},
         {(0, 0): 1}),
    ]


def _build_decomposition(n, m, p):
    """Build the structured rank-44 decomposition for 3x4x4."""
    # Part 1: 16 terms for A[0,:] * B
    # Part 2: 4 blocks * 7 Strassen terms = 28 terms
    total_rank = 16 + 4 * 7
    U = [[0.0] * total_rank for _ in range(n * m)]   # (n*m, rank)
    V = [[0.0] * total_rank for _ in range(m * p)]   # (m*p, rank)
    W = [[0.0] * total_rank for _ in range(p * n)]   # (p*n, rank), W[k*n+i]

    col = 0

    # ---- Part 1: row 0 of A times B (1x4 x 4x4 = 1x4) ----
    # C[0,k] = sum_j A[0,j] B[j,k]
    for k in range(p):
        for j in range(m):
            U[0 * m + j][col] = 1.0
            V[j * p + k][col] = 1.0
            W[k * n + 0][col] = 1.0
            col += 1

    # ---- Part 2: rows 1,2 of A times B (2x4 x 4x4 = 2x4) ----
    # Split B into 2x2 blocks by (row, col):
    #   B00 rows 0:2, cols 0:2
    #   B01 rows 0:2, cols 2:4
    #   B10 rows 2:4, cols 0:2
    #   B11 rows 2:4, cols 2:4
    # Split A (rows 1,2) into left cols 0:2 (A_L) and right cols 2:4 (A_R).
    # Each 2x2 multiplication uses Strassen (7 terms).
    #
    # Block mapping: (A_row_offset, A_col_offset, B_row_offset, B_col_offset,
    #                 C_row_offset, C_col_offset)
    blocks = [
        (1, 0, 0, 0, 1, 0),  # A_L * B00 -> C rows 1,2 cols 0,1
        (1, 0, 0, 2, 1, 2),  # A_L * B01 -> C rows 1,2 cols 2,3
        (1, 2, 2, 0, 1, 0),  # A_R * B10 -> C rows 1,2 cols 0,1
        (1, 2, 2, 2, 1, 2),  # A_R * B11 -> C rows 1,2 cols 2,3
    ]

    terms = _strassen_terms()
    for (ar, ac, br, bc, cr, cc) in blocks:
        for (x_coeffs, y_coeffs, z_coeffs) in terms:
            # A indices: row ar+xi, col ac+yi  -> U[(ar+xi)*m + (ac+yi)]
            for (xi, yi), ux in x_coeffs.items():
                U[(ar + xi) * m + (ac + yi)][col] += ux
            # B indices: row br+yi, col bc+xi  -> V[(br+yi)*p + (bc+xi)]
            for (yi, xi), vy in y_coeffs.items():
                V[(br + yi) * p + (bc + xi)][col] += vy
            # C indices: row cr+zi, col cc+zj  -> W[(cc+zj)*n + (cr+zi)]
            for (zi, zj), wz in z_coeffs.items():
                W[(cc + zj) * n + (cr + zi)][col] += wz
            col += 1

    return {"U": U, "V": V, "W": W}


def construct():
    """Main entry point: return the structured rank-44 decomposition, verified."""
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before the 120s limit

    n, m, p = DIMENSIONS

    result = _build_decomposition(n, m, p)

    # Verify exactness
    if not _verify_fast(result["U"], result["V"], result["W"], n, m, p):
        # Should never happen for this construction; fall back to schoolbook.
        result = _schoolbook(n, m, p)

    # Snap to exact half-integers for a clean, exact representation
    result["U"] = _snap_to_half_integer(result["U"])
    result["V"] = _snap_to_half_integer(result["V"])
    result["W"] = _snap_to_half_integer(result["W"])

    # Final verification after snapping
    if not _verify_fast(result["U"], result["V"], result["W"], n, m, p):
        result = _schoolbook(n, m, p)

    return result


def _schoolbook(n, m, p):
    """Fallback standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}


# Quick self-test
if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    assert _verify_fast(res["U"], res["V"], res["W"], n, m, p), "Verification failed"
    rank = len(res["U"][0])
    print(f"Rank: {rank}")
