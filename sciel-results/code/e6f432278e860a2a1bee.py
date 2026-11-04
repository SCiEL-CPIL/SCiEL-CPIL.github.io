import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    t0 = time.time()
    n, m, p = 2, 4, 5
    R = 36

    # Pre-allocate as numpy int8 arrays for speed, then convert to list-of-lists
    U = np.zeros((n * m, R), dtype=np.int8)
    V = np.zeros((m * p, R), dtype=np.int8)
    W = np.zeros((p * n, R), dtype=np.int8)

    c = 0  # current column index

    # Strassen block: multiplies 2x2 A by 2x2 B to get 2x2 C
    # A rows: ar, ar+1 ; A cols: ac, ac+1
    # B rows: br, br+1 ; B cols: bc, bc+1
    # C: W index = k * n + i, so C row i, col k -> index k*n + i
    #   c_row corresponds to i (0 or 1), c_col corresponds to k (0..4)
    def strassen(ar, ac, br, bc, cr, cc):
        nonlocal c
        # A indices: i*m + j
        a00 = ar * m + ac
        a01 = ar * m + ac + 1
        a10 = (ar + 1) * m + ac
        a11 = (ar + 1) * m + ac + 1

        # B indices: j*p + k
        b00 = br * p + bc
        b01 = br * p + bc + 1
        b10 = (br + 1) * p + bc
        b11 = (br + 1) * p + bc + 1

        # C indices: k*n + i  (W[k*n+i, r])
        c00 = cc * n + cr
        c01 = (cc + 1) * n + cr
        c10 = cc * n + cr + 1
        c11 = (cc + 1) * n + cr + 1

        # M1 = (a00 + a11)(b00 + b11) -> contributes to c00, c11
        U[a00, c] += 1; U[a11, c] += 1
        V[b00, c] += 1; V[b11, c] += 1
        W[c00, c] += 1; W[c11, c] += 1
        c += 1

        # M2 = (a10 + a11) * b00 -> contributes to c10, c11
        U[a10, c] += 1; U[a11, c] += 1
        V[b00, c] += 1
        W[c10, c] += 1; W[c11, c] -= 1
        c += 1

        # M3 = a00 * (b01 - b11) -> contributes to c01, c11
        U[a00, c] += 1
        V[b01, c] += 1; V[b11, c] -= 1
        W[c01, c] += 1; W[c11, c] += 1
        c += 1

        # M4 = a11 * (b10 - b00) -> contributes to c00, c10
        U[a11, c] += 1
        V[b10, c] += 1; V[b00, c] -= 1
        W[c00, c] += 1; W[c10, c] += 1
        c += 1

        # M5 = (a00 + a01) * b11 -> contributes to c00, c01
        U[a00, c] += 1; U[a01, c] += 1
        V[b11, c] += 1
        W[c00, c] -= 1; W[c01, c] += 1
        c += 1

        # M6 = (a10 - a00) * (b00 + b01) -> contributes to c11
        U[a10, c] += 1; U[a00, c] -= 1
        V[b00, c] += 1; V[b01, c] += 1
        W[c11, c] += 1
        c += 1

        # M7 = (a01 - a11) * (b10 + b11) -> contributes to c00
        U[a01, c] += 1; U[a11, c] -= 1
        V[b10, c] += 1; V[b11, c] += 1
        W[c00, c] += 1
        c += 1

    # Schoolbook 2x2 * 2x1 -> 2x1 (4 terms)
    def schoolbook_221(ar, ac, br, bc, cr, cc):
        nonlocal c
        a00 = ar * m + ac
        a01 = ar * m + ac + 1
        a10 = (ar + 1) * m + ac
        a11 = (ar + 1) * m + ac + 1

        b0 = br * p + bc
        b1 = (br + 1) * p + bc

        c0 = cc * n + cr
        c1 = cc * n + cr + 1

        # a00*b0 -> c0
        U[a00, c] += 1; V[b0, c] += 1; W[c0, c] += 1
        c += 1
        # a01*b1 -> c0
        U[a01, c] += 1; V[b1, c] += 1; W[c0, c] += 1
        c += 1
        # a10*b0 -> c1
        U[a10, c] += 1; V[b0, c] += 1; W[c1, c] += 1
        c += 1
        # a11*b1 -> c1
        U[a11, c] += 1; V[b1, c] += 1; W[c1, c] += 1
        c += 1

    # 4 Strassen blocks for B cols 0..3, A split into left (cols 0,1) and right (cols 2,3)
    # C has 5 columns (k=0..4). For k=0..3, we use Strassen 2x2 blocks.
    # A left: rows 0,1 cols 0,1  -> ar=0, ac=0
    # A right: rows 0,1 cols 2,3 -> ar=0, ac=2
    # B block 0: rows 0,1 cols 0,1 -> br=0, bc=0
    # B block 1: rows 0,1 cols 2,3 -> br=0, bc=2

    # C block for k=0,1 (cc=0, cr=0): sum of A_left * B_block0 + A_right * B_block1
    strassen(0, 0, 0, 0, 0, 0)   # A_left * B_block0 -> C[0:2, 0:2]
    strassen(0, 2, 0, 2, 0, 0)   # A_right * B_block1 -> C[0:2, 0:2]

    # C block for k=2,3 (cc=2, cr=0): A_left * B_block1 + A_right * B_block0
    # Wait, need to be careful. C = A * B.
    # A is 2x4, B is 4x5.
    # C[i][k] = sum_j A[i][j] * B[j][k]
    # Split A into A_L (cols 0,1) and A_R (cols 2,3).
    # C = A_L * B[0:2, :] + A_R * B[2:4, :]
    # B[0:2, :] is 2x5, B[2:4, :] is 2x5.
    # For k=0,1: C[0:2, 0:2] = A_L * B[0:2, 0:2] + A_R * B[2:4, 0:2]
    # B[0:2, 0:2]: br=0, bc=0
    # B[2:4, 0:2]: br=2, bc=0
    # For k=2,3: C[0:2, 2:4] = A_L * B[0:2, 2:4] + A_R * B[2:4, 2:4]
    # B[0:2, 2:4]: br=0, bc=2
    # B[2:4, 2:4]: br=2, bc=2
    # For k=4: C[0:2, 4:5] = A_L * B[0:2, 4:5] + A_R * B[2:4, 4:5]
    # B[0:2, 4:5]: br=0, bc=4 (2x1)
    # B[2:4, 4:5]: br=2, bc=4 (2x1)

    # So correct calls:
    c = 0  # reset
    # k=0,1
    strassen(0, 0, 0, 0, 0, 0)   # A_L * B[0:2,0:2]
    strassen(0, 2, 2, 0, 0, 0)   # A_R * B[2:4,0:2]
    # k=2,3
    strassen(0, 0, 0, 2, 0, 2)   # A_L * B[0:2,2:4]
    strassen(0, 2, 2, 2, 0, 2)   # A_R * B[2:4,2:4]
    # k=4
    schoolbook_221(0, 0, 0, 4, 0, 4)  # A_L * B[0:2,4:5]
    schoolbook_221(0, 2, 2, 4, 0, 4)  # A_R * B[2:4,4:5]

    # Verify c == R
    assert c == R, f"Expected rank {R}, got {c}"

    # Convert to list-of-lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
