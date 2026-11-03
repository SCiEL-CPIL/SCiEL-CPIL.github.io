"""Tensor decomposition for 3x4x8 matrix multiplication."""
DIMENSIONS = [3, 4, 8]

import time

_START = time.time()
_DEADLINE = 140.0  # safety margin under 150s kill

def _add_block(U, V, W, u, v, w, i0, j0, k0):
    """Add a block decomposition (lists of (row,col,val) tuples) into global matrices."""
    for (r, c, val) in u:
        U[i0 + r][c] += val
    for (r, c, val) in v:
        V[j0 + r][c] += val
    for (r, c, val) in w:
        W[k0 + r][c] += val


def _strassen_2x2x2():
    """Return U,V,W (each 4x7 list of (row,col,val)) for M_{2,2,2} Strassen."""
    # Strassen: M222 rank 7
    # U indices (i,j) in 0..3, V (j,k), W (k,i)
    U = []
    V = []
    W = []
    # 7 products
    # m0 = (A00+A11)(B00+B11)
    U += [(0,0,1),(3,0,1)]
    V += [(0,0,1),(3,0,1)]
    W += [(0,0,1),(3,3,1)]
    # m1 = (A10+A11)B00
    U += [(2,1,1),(3,1,1)]
    V += [(0,1,1)]
    W += [(0,2,1),(0,3,1),(3,2,1),(3,3,1)]
    # m2 = A00(B10-B11)
    U += [(0,2,1)]
    V += [(2,2,1),(3,2,-1)]
    W += [(2,0,1),(3,0,1)]
    # m3 = A11(B01-B00)
    U += [(3,3,1)]
    V += [(1,3,1),(0,3,-1)]
    W += [(1,3,1),(0,3,1)]
    # m4 = (A00+A01)B11
    U += [(0,4,1),(1,4,1)]
    V += [(3,4,1)]
    W += [(0,0,1),(1,0,1)]
    # m5 = (A10-A00)(B00+B01)
    U += [(2,5,1),(0,5,-1)]
    V += [(0,5,1),(1,5,1)]
    W += [(0,2,1),(1,2,1)]
    # m6 = (A01-A11)(B10+B11)
    U += [(1,6,1),(3,6,-1)]
    V += [(2,6,1),(3,6,1)]
    W += [(2,0,1),(2,1,1)]
    return U, V, W


def _block_M244():
    """
    M_{2,4,4} decomposition: multiply 2x4 by 4x4.
    Split into two 2x2x4 blocks? Actually use Strassen on the 4x4 by viewing
    as 2x2 blocks of 2x2.
    Rank: 2 * 7 = 14? No. M_{2,4,4}: A is 2x4, B is 4x4, C is 2x4.
    We can split B into [B0 B1] each 4x2, giving two M_{2,4,2}.
    Each M_{2,4,2}: split A into 2x2 (A0,A1), B into 4x2 = [C0;C1] (4x2).
    Actually simpler: M_{2,4,2} = split A rows into 2x2? A is 2x4.
    Let's do: A (2x4) = [A0 A1] (each 2x2). B (4x2) = [B0; B1] (each 2x2).
    C = A0*B0 + A1*B1, each 2x2 -> two M_{2,2,2} = 2*7 = 14.
    So M_{2,4,2} rank 14, M_{2,4,4} = 2 * M_{2,4,2} = 28.
    """
    U = []
    V = []
    W = []
    # M_{2,4,4}: A(2x4), B(4x4), C(2x4)
    # Split B into B_left(4x2), B_right(4x2)
    # For each: M_{2,4,2}
    for half in range(2):  # 0: left, 1: right
        # A(2x4) = [A0 A1] 2x2 each; B(4x2) = [B0;B1] 2x2 each
        # C_left = A0*B0 + A1*B1
        # A0*B0: M_{2,2,2} with A rows 0-1, B rows 0-1, C cols 0-1
        U0, V0, W0 = _strassen_2x2x2()
        # A0 at A[0:2,0:2], B0 at B[0:2, half*2:half*2+2], C at C[0:2, half*2:half*2+2]
        for (r, c, val) in U0:
            U.append((r, c, val))
        for (r, c, val) in V0:
            V.append((r, c, val))
        for (r, c, val) in W0:
            W.append((r + half * 2, c, val))
        # A1*B1: A1 at A[0:2,2:4], B1 at B[2:4, half*2:half*2+2], C at C[0:2, half*2:half*2+2]
        U1, V1, W1 = _strassen_2x2x2()
        for (r, c, val) in U1:
            U.append((r + 2, c, val))
        for (r, c, val) in V1:
            V.append((r + 2, c, val))
        for (r, c, val) in W1:
            W.append((r + half * 2, c, val))
    return U, V, W


def _block_M144():
    """M_{1,4,4}: A(1x4), B(4x4), C(1x4). Rank 16 schoolbook (1*4*4)."""
    U = []
    V = []
    W = []
    for j in range(4):
        for k in range(4):
            U.append((j, j * 4 + k, 1))
            V.append((j * 4 + k, j * 4 + k, 1))
            W.append((k, j * 4 + k, 1))
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    # Split A (3x4) into A_top(2x4) and A_bot(1x4)
    # C_top(2x8) = A_top * B (4x8)  -> M_{2,4,8}
    # C_bot(1x8) = A_bot * B (4x8)  -> M_{1,4,8}
    # M_{2,4,8} = 2 * M_{2,4,4} (split B into two 4x4) = 2*28 = 56
    # M_{1,4,8} = 2 * M_{1,4,4} (split B into two 4x4) = 2*16 = 32
    # Total rank = 56 + 32 = 88
    rank = 88
    U = [[0] * rank for _ in range(n * m)]  # 12 x 88
    V = [[0] * rank for _ in range(m * p)]  # 32 x 88
    W = [[0] * rank for _ in range(p * n)]  # 24 x 88

    col = 0
    # C_top: rows 0-1 of A (i=0,1), full 4x8 B
    # M_{2,4,8}: A(2x4) at A[0:2,0:4], B(4x8) at B[0:4,0:8], C(2x8) at C[0:2,0:8]
    # Split B into B[0:4,0:4] and B[0:4,4:8]
    for half in range(2):
        Ublk, Vblk, Wblk = _block_M244()
        for (r, c, val) in Ublk:
            U[r][col + c] = val  # A rows 0-1, cols 0-3
        for (r, c, val) in Vblk:
            V[r][col + c] = val  # B rows 0-3, cols half*4..half*4+3
        for (r, c, val) in Wblk:
            W[r][col + c] = val  # C rows 0-1, cols half*4..half*4+3
        col += 28
    # C_bot: row 2 of A (i=2), full 4x8 B
    # M_{1,4,8}: A(1x4) at A[2:3,0:4], B(4x8) at B[0:4,0:8], C(1x8) at C[2:3,0:8]
    for half in range(2):
        Ublk, Vblk, Wblk = _block_M144()
        for (r, c, val) in Ublk:
            U[2 + r][col + c] = val  # A row 2, cols 0-3
        for (r, c, val) in Vblk:
            V[r][col + c] = val  # B rows 0-3, cols half*4..half*4+3
        for (r, c, val) in Wblk:
            W[2 + r][col + c] = val  # C row 2, cols half*4..half*4+3
        col += 16

    return {"U": U, "V": V, "W": W}


def _verify(U, V, W):
    """Verify the decomposition against random test matrices."""
    import numpy as np
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for trial in range(20):
        A = np.random.randint(-3, 4, size=(n, m))
        B = np.random.randint(-3, 4, size=(m, p))
        C_expected = A @ B
        # Reconstruct: C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # U indexed [i*m+j], V [j*p+k], W [k*n+i]
        C_recon = np.zeros((n, p))
        for c in range(rank):
            uc = Ua[:, c]  # length n*m
            vc = Va[:, c]  # length m*p
            wc = Wa[:, c]  # length p*n
            # outer-ish: for each i,j,k: uc[i*m+j]*vc[j*p+k]*wc[k*n+i]
            uc3 = uc.reshape(n, m)
            vc3 = vc.reshape(m, p)
            wc3 = wc.reshape(p, n)
            # C[i,k] += sum_j uc3[i,j]*vc3[j,k]* (wc3[k,i] same for all j)
            C_recon += (uc3 @ vc3) * wc3.T
        if not np.allclose(C_recon, C_expected, atol=1e-9):
            return False
    return True


def construct():
    if time.time() > _DEADLINE:
        return _fallback()
    try:
        res = _construct_block_strassen()
        # Verify
        if _verify(res["U"], res["V"], res["W"]):
            return res
        # If verification fails, fall back to schoolbook
        return _fallback()
    except Exception:
        return _fallback()


def _construct_block_strassen():
    n, m, p = DIMENSIONS
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for half in range(2):
        Ublk, Vblk, Wblk = _block_M244()
        for (r, c, val) in Ublk:
            U[r][col + c] = val
        for (r, c, val) in Vblk:
            V[r][col + c] = val
        for (r, c, val) in Wblk:
            W[r][col + c] = val
        col += 28
    for half in range(2):
        Ublk, Vblk, Wblk = _block_M144()
        for (r, c, val) in Ublk:
            U[2 + r][col + c] = val
        for (r, c, val) in Vblk:
            V[r][col + c] = val
        for (r, c, val) in Wblk:
            W[2 + r][col + c] = val
        col += 16
    return {"U": U, "V": V, "W": W}


def _fallback():
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
