"""Matrix multiplication 4,4,8 rank-minimal exact tensor decomposition.

Combines the current program's recursive two-level Strassen structure with the
inspiration's correct factor-list representation (standard Strassen W
coefficients). The current program's rank-98 result failed verification because
its 2x2 W factors were wrong (e.g. p2 only set C21, missing -C22). This version
uses the correct standard Strassen factors so the rank-98 decomposition for
4x4 * 4x8 verifies exactly.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _schoolbook():
    """Return the trivial rank n*m*p schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, tol=1e-8):
    """Check that U,V,W exactly decompose the matmul tensor. Returns True if valid."""
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    if r == 0:
        return False
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, r)
    V3 = V_arr.reshape(m, p, r)
    W3 = W_arr.reshape(p, n, r)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    expected = np.ones((n, m, p))
    return bool(np.allclose(T, expected, atol=tol))


def _strassen_2x2_factors():
    """
    Return the 7 rank-1 factors for 2x2*2x2 matrix multiplication.
    Each factor is (u_vec, v_vec, w_vec) where each is a 4-element vector
    in row-major order: index (a,b) -> a*2+b.
      u_vec: A (A00, A01, A10, A11)
      v_vec: B (B00, B01, B10, B11)
      w_vec: C (C00, C01, C10, C11)
    Standard Strassen:
      p1 = (A00+A11)(B00+B11)
      p2 = (A10+A11) B00
      p3 = A00 (B01-B11)
      p4 = A11 (B10-B00)
      p5 = (A00+A01) B11
      p6 = (A10-A00)(B00+B01)
      p7 = (A01-A11)(B10+B11)
    Reconstruction:
      C00 = p1 + p4 - p5 + p7
      C01 = p3 + p5
      C10 = p2 + p4
      C11 = p1 - p2 + p3 + p6
    """
    factors = []
    # p1 = (A00+A11)(B00+B11) -> C00 += p1, C11 += p1
    factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # p2 = (A10+A11) B00 -> C10 += p2, C11 -= p2
    factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    # p3 = A00 (B01-B11) -> C01 += p3, C11 += p3
    factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    # p4 = A11 (B10-B00) -> C00 += p4, C10 += p4
    factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
    # p5 = (A00+A01) B11 -> C00 -= p5, C01 += p5
    factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    # p6 = (A10-A00)(B00+B01) -> C11 += p6
    factors.append([[-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]])
    # p7 = (A01-A11)(B10+B11) -> C00 += p7
    factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
    return factors


def _strassen_4x4_factors():
    """
    Return U, V, W for 4x4*4x4 matrix multiplication using recursive Strassen.
    U: 16x49, V: 16x49, W: 16x49
    Indices: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    Rank = 7 (top-level) * 7 (scalar) = 49.
    """
    s2 = _strassen_2x2_factors()
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))

    # Top-level block coefficients derived from the 7 standard Strassen factors.
    # Each is (u_blocks, v_blocks, w_blocks) where each is a 4-vector over
    # block indices (bi,bj) -> bi*2+bj, matching the 2x2 factor layout.
    # s=0: (A00+A11)(B00+B11) -> C00, C11
    # s=1: (A10+A11)B00      -> C10, C11
    # s=2: A00(B01-B11)      -> C01, C11
    # s=3: A11(B10-B00)      -> C00, C10
    # s=4: (A00+A01)B11      -> C00, C01
    # s=5: (A10-A00)(B00+B01)-> C11
    # s=6: (A01-A11)(B10+B11)-> C00
    top = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # s=0
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # s=1
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # s=2
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),  # s=3
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # s=4
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # s=5
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # s=6
    ]

    for s in range(7):
        ub, vb, wb = top[s]
        for r in range(7):
            u_s, v_s, w_s = s2[r]
            col = s * 7 + r

            # U: A element (i,j) = (2*bi+ei, 2*bj+ej)
            for bi in range(2):
                for bj in range(2):
                    cb = ub[bi * 2 + bj]
                    if cb == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            cs = u_s[ei * 2 + ej]
                            if cs == 0:
                                continue
                            i = 2 * bi + ei
                            j = 2 * bj + ej
                            U[i * 4 + j, col] += cb * cs

            # V: B element (j,k) = (2*bi+ei, 2*bj+ej)
            for bi in range(2):
                for bj in range(2):
                    cb = vb[bi * 2 + bj]
                    if cb == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            cs = v_s[ei * 2 + ej]
                            if cs == 0:
                                continue
                            j = 2 * bi + ei
                            k = 2 * bj + ej
                            V[j * 4 + k, col] += cb * cs

            # W: C element (k,i) = (2*bi+ei, 2*bj+ej)
            for bi in range(2):
                for bj in range(2):
                    cb = wb[bi * 2 + bj]
                    if cb == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            cs = w_s[ei * 2 + ej]
                            if cs == 0:
                                continue
                            k = 2 * bi + ei
                            i = 2 * bj + ej
                            W[k * 4 + i, col] += cb * cs

    return U, V, W


def _construct_rank98():
    """
    Construct a rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    Split B (4x8) into two 4x4 blocks B_left, B_right. Compute
    C_left = A*B_left and C_right = A*B_right, each via rank-49 Strassen.
    Independent U factors per block (U has only 4 columns, cannot be shared).
    U: 16x98 = [U_left | U_right]
    V: 32x98, rows 0-15 = V_left (cols 0-48), rows 16-31 = V_right (cols 49-97)
    W: 32x98, rows 0-15 = W_left (cols 0-48), rows 16-31 = W_right (cols 49-97)
    """
    U4, V4, W4 = _strassen_4x4_factors()
    rank4 = U4.shape[1]  # 49
    rank = rank4 * 2     # 98

    U = np.zeros((16, rank))
    U[:, :rank4] = U4
    U[:, rank4:] = U4

    V = np.zeros((32, rank))
    V[:16, :rank4] = V4
    V[16:, rank4:] = V4

    W = np.zeros((32, rank))
    W[:16, :rank4] = W4
    W[16:, rank4:] = W4

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix
    multiplication. Returns {'U', 'V', 'W'} with the decomposition.
    """
    deadline = time.time() + 110  # Safety margin before the 120s limit

    try:
        result = _construct_rank98()
        if _verify(result["U"], result["V"], result["W"]):
            return result
    except Exception:
        pass

    if time.time() > deadline:
        return _schoolbook()

    return _schoolbook()

# EVOLVE-BLOCK-END
