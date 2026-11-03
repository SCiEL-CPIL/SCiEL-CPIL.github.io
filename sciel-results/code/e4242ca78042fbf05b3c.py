import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, -1)
    V3 = V_arr.reshape(m, p, -1)
    W3 = W_arr.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
    return np.allclose(T, 1.0, atol=tol)

def _snap_to_half_integer(mat, tol=1e-9):
    arr = np.array(mat, dtype=float)
    snapped = np.round(arr * 2) / 2
    if np.max(np.abs(arr - snapped)) < tol:
        out = []
        for row in snapped:
            out.append([float(x) for x in row])
        return out
    return [list(map(float, row)) for row in arr]

def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2 matrix multiplication.
    
    For X(2x2) * Y(2x2) = Z(2x2):
    M1 = (X00+X11)(Y00+Y11)
    M2 = (X10+X11)(Y00)
    M3 = (X00)(Y01-Y11)
    M4 = (X11)(Y10-Y00)
    M5 = (X00+X01)(Y11)
    M6 = (X10-X00)(Y00+Y01)
    M7 = (X01-X11)(Y10+Y11)
    
    Z00 = M1 + M4 - M5 + M7
    Z01 = M3 + M5
    Z10 = M2 + M4
    Z11 = M1 - M2 + M3 + M6
    """
    terms = [
        # M1: (X00+X11)(Y00+Y11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2: (X10+X11)(Y00)
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3: (X00)(Y01-Y11)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4: (X11)(Y10-Y00)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5: (X00+X01)(Y11)
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6: (X10-X00)(Y00+Y01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7: (X01-X11)(Y10+Y11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    return terms

def _build_decomposition(n, m, p):
    """Build rank-44 decomposition for 3x4 x 4x4 -> 3x4.
    
    Strategy:
    - Row 0 of A: schoolbook 1x4 x 4x4 = 16 terms
    - Rows 1-2 of A: 2x4 x 4x4 using 4 Strassen 2x2 blocks = 28 terms
    Total: 44 terms
    """
    total_rank = 44
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Row 0 of A times B (1x4 x 4x4), schoolbook rank 16
    # C[0, k] = sum_j A[0, j] * B[j, k]
    for k in range(p):
        for j in range(m):
            U[0 * m + j][col] = 1.0
            V[j * p + k][col] = 1.0
            W[k * n + 0][col] = 1.0
            col += 1
    
    # Part 2: Rows 1-2 of A times B (2x4 x 4x4)
    # Split B into four 2x2 blocks:
    # B00 = B[0:2, 0:2], B01 = B[0:2, 2:4]
    # B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
    # A_L = A[1:3, 0:2], A_R = A[1:3, 2:4]
    # Four products: A_L*B00, A_L*B01, A_R*B10, A_R*B11
    # Each is 2x2 x 2x2, use Strassen (7 terms each) = 28 total
    
    strassen_terms = _strassen_2x2_terms()
    
    # Blocks: (A_row_off, A_col_off, B_row_off, B_col_off, C_row_off, C_col_off)
    blocks = [
        (1, 0, 0, 0, 1, 0),  # A_L * B00 -> C[1:3, 0:2]
        (1, 0, 0, 2, 1, 2),  # A_L * B01 -> C[1:3, 2:4]
        (1, 2, 2, 0, 1, 0),  # A_R * B10 -> C[1:3, 0:2]
        (1, 2, 2, 2, 1, 2),  # A_R * B11 -> C[1:3, 2:4]
    ]
    
    for (ar, ac, br, bc, cr, cc) in blocks:
        for (x_coeffs, y_coeffs, z_coeffs) in strassen_terms:
            # U: A indices
            for (xi, yi), ux in x_coeffs.items():
                U[(ar + xi) * m + (ac + yi)][col] += ux
            # V: B indices
            for (yi, xi), vy in y_coeffs.items():
                V[(br + yi) * p + (bc + xi)][col] += vy
            # W: C indices (W[k*n + i] where k is col, i is row)
            for (zi, zj), wz in z_coeffs.items():
                W[(cc + zj) * n + (cr + zi)][col] += wz
            col += 1
    
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
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

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    n, m, p = DIMENSIONS
    
    # Build the rank-44 decomposition
    result = _build_decomposition(n, m, p)
    
    # Verify
    if not _verify_fast(result["U"], result["V"], result["W"], n, m, p):
        result = _schoolbook(n, m, p)
    
    # Snap to half-integers for clean output
    result["U"] = _snap_to_half_integer(result["U"])
    result["V"] = _snap_to_half_integer(result["V"])
    result["W"] = _snap_to_half_integer(result["W"])
    
    # Final verification after snapping
    if not _verify_fast(result["U"], result["V"], result["W"], n, m, p):
        result = _schoolbook(n, m, p)
    
    return result
