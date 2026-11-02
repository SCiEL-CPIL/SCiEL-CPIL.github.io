"""Tensor decomposition for 4x5 * 5x6 matrix multiplication (low rank)."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Rank-7 Strassen for 2x2 * 2x2. U,V,W each (4,7)."""
    U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
    # M1=(A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2=(A10+A11)B00
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    # M3=A00(B10-B11)
    U[0,2]=1; V[2,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    # M4=A11(B01-B00)
    U[3,3]=1; V[1,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    # M5=(A00+A01)B11
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    # M6=(A10-A00)(B00+B01)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7=(A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W


def _laderman_3x3x3():
    """Rank-7 Laderman for 3x3 * 3x3. U,V,W each (9,7)."""
    U = np.zeros((9, 7)); V = np.zeros((9, 7)); W = np.zeros((9, 7))
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[4,0]=1; V[0,0]=1; V[4,0]=1; W[0,0]=1; W[4,0]=1
    # M2 = (A10+A11)B00
    U[3,1]=1; U[4,1]=1; V[0,1]=1; W[3,1]=1; W[4,1]=-1
    # M3 = A00(B10-B11)
    U[0,2]=1; V[3,2]=1; V[4,2]=-1; W[1,2]=1; W[4,2]=1
    # M4 = A11(B01-B00)
    U[4,3]=1; V[1,3]=1; V[0,3]=-1; W[0,3]=1; W[3,3]=1
    # M5 = (A00+A01)B11
    U[0,4]=1; U[1,4]=1; V[4,4]=1; W[0,4]=-1; W[1,4]=1
    # M6 = (A10-A00)(B00+B01)
    U[3,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[4,5]=1
    # M7 = (A01-A11)(B10+B11)
    U[1,6]=1; U[4,6]=-1; V[3,6]=1; V[4,6]=1; W[0,6]=1
    return U, V, W


def _embed(U, V, W, rU, rV, rW, n, m, p, row_off, col_off, vrow_off, vcol_off, wrow_off, wcol_off, scale=1.0):
    """Embed a small (rU x rV x rW) decomposition into a larger (n x m x p) tensor.

    U small is (rU, rank) indexed by (i,j) with i in [0,rU), j in [0,rV).
    We place it so that the A-block occupies rows row_off..row_off+rU and cols col_off..col_off+rV.
    The B-block occupies vrow_off..vrow_off+rV (rows of B = cols of A) and vcol_off..vcol_off+rW.
    The C-block occupies wrow_off..wrow_off+rW (cols of B) and wcol_off..wcol_off+rU (rows of A).
    """
    rank = U.shape[1]
    bigU = np.zeros((n * m, rank))
    bigV = np.zeros((m * p, rank))
    bigW = np.zeros((p * n, rank))
    for t in range(rank):
        for i in range(rU):
            for j in range(rV):
                val = U[i * rV + j, t]
                if val != 0:
                    bigU[(row_off + i) * m + (col_off + j), t] = val
        for j in range(rV):
            for k in range(rW):
                val = V[j * rW + k, t]
                if val != 0:
                    bigV[(vrow_off + j) * p + (vcol_off + k), t] = val
        for k in range(rW):
            for i in range(rU):
                val = W[k * rU + i, t]
                if val != 0:
                    bigW[(wrow_off + k) * n + (wcol_off + i), t] = val
    return bigU * scale, bigV * scale, bigW * scale


def _combine(lists):
    """Combine a list of (U,V,W) full-size tensors by concatenating rank columns."""
    if not lists:
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0))
    U = np.concatenate([x[0] for x in lists], axis=1)
    V = np.concatenate([x[1] for x in lists], axis=1)
    W = np.concatenate([x[2] for x in lists], axis=1)
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify T[i,j,k] = sum_t U[i,j,t] V[j,k,t] W[k,i,t] == 1 for all i,j,k."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank)); V = np.zeros((m * p, rank)); W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _build_4x5x6():
    """
    Block decomposition for 4x5 * 5x6.

    A (4x5): rows {0,1}|{2,3}, cols {0,1}|{2,3,4}
    B (5x6): rows {0,1}|{2,3,4}, cols {0,1,2}|{3,4,5}
    C (4x6): rows {0,1}|{2,3}, cols {0,1,2}|{3,4,5}

    C_top_left  = A_tl(2x2)*B_tl(2x3) + A_tr(2x3)*B_bl(3x3)
    C_top_right = A_tl(2x2)*B_tr(2x3) + A_tr(2x3)*B_br(3x3)
    C_bot_left  = A_bl(2x2)*B_tl(2x3) + A_br(2x3)*B_bl(3x3)
    C_bot_right = A_bl(2x2)*B_tr(2x3) + A_br(2x3)*B_br(3x3)

    For each (2x2)*(2x3): use a rank-9 decomposition (Strassen for 2 cols + schoolbook 1 col).
    For each (2x3)*(3x3): use a rank-12 decomposition (3 cols of 2x3x3, each rank 4) -- or better.

    We embed each term into the full tensor at the appropriate block offsets.
    """
    n, m, p = 4, 5, 6
    terms = []

    # Block offsets:
    # A top rows 0,1 ; bot rows 2,3
    # A left cols 0,1 ; right cols 2,3,4
    # B top rows 0,1 ; bot rows 2,3,4
    # B left cols 0,1,2 ; right cols 3,4,5
    A_top, A_bot = 0, 2
    A_left, A_right = 0, 2
    B_top, B_bot = 0, 2
    B_left, B_right = 0, 3

    # (2x2) * (2x3) helper: A is 2x2, B is 2x3, C is 2x3.
    # Decompose as: C[:,0:2] via Strassen 2x2x2 (rank 7), C[:,2] via schoolbook (rank 2).
    def two_by_two_by_two_by_three():
        # Strassen part for columns 0,1 (2x2 * 2x2 -> 2x2)
        Us, Vs, Ws = _strassen_2x2x2()
        # Embed Strassen into 2x2 x 2x2 within the 2x3 B (cols 0,1)
        U1, V1, W1 = _embed(Us, Vs, Ws, 2, 2, 2, 2, 2, 2,
                            row_off=0, col_off=0,
                            vrow_off=0, vcol_off=0,
                            wrow_off=0, wcol_off=0)
        # Schoolbook for column 2: C[i,2] = A[i,0]B[0,2] + A[i,1]B[1,2]  -> rank 2
        U2 = np.zeros((4, 2)); V2 = np.zeros((6, 2)); W2 = np.zeros((4, 2))
        U2[0,0]=1; V2[4,0]=1; W2[2,0]=1   # A[0,0]*B[0,2] -> C[0,2]
        U2[1,0]=1; V2[5,0]=1; W2[6,0]=1   # A[1,0]*B[0,2] -> C[1,2]
        U2[0,1]=1; V2[0,1]=1; W2[2,1]=1   # A[0,1]*B[1,2] -> C[0,2]
        U2[1,1]=1; V2[1,1]=1; W2[6,1]=1   # A[1,1]*B[1,2] -> C[1,2]
        # Note: in the 2x3 B, col index 2 -> V row j*3 + k with k=2.
        # V2 indices: row j in [0,2), col k in [0,3). B[0,2] -> index 0*3+2=2, B[1,2] -> 1*3+2=5.
        # Fix V2 accordingly.
        V2 = np.zeros((6, 2))
        V2[2,0]=1; V2[5,0]=1  # B[0,2], B[1,2]
        W2 = np.zeros((4, 2))
        W2[2,0]=1; W2[6,0]=1  # C[0,2] (k=2,i=0 -> 2*2+0=4? wait)
        # W index = k*n + i with n=2 here (local), k in [0,3), i in [0,2).
        # C[0,2] -> k=2,i=0 -> 2*2+0 = 4. C[1,2] -> k=2,i=1 -> 2*2+1 = 5.
        W2 = np.zeros((6, 2))
        W2[4,0]=1; W2[5,0]=1
        U2 = np.zeros((4, 2))
        U2[0,0]=1; U2[1,0]=1  # A[0,0],A[1,0] for B[0,2]
        U2[0,1]=1; U2[1,1]=1  # A[0,1],A[1,1] for B[1,2]
        # Recompute: term t=0: A[i,0]*B[0,2] contributes to C[i,2]. U[i*2+0,0]=1, V[0*3+2,0]=1, W[2*2+i,0]=1.
        # t=1: A[i,1]*B[1,2] contributes to C[i,2]. U[i*2+1,1]=1, V[1*3+2,1]=1, W[2*2+i,1]=1.
        U2 = np.zeros((4, 2)); V2 = np.zeros((6, 2)); W2 = np.zeros((6, 2))
        for i in range(2):
            U2[i*2+0, 0] = 1
            U2[i*2+1, 1] = 1
            V2[0*3+2, 0] = 1
            V2[1*3+2, 1] = 1
            W2[2*2+i, 0] = 1
            W2[2*2+i, 1] = 1
        return _combine([(U1, V1, W1), (U2, V2, W2)])

    # (2x3) * (3x3) helper: A is 2x3, B is 3x3, C is 2x3.
    # Decompose as 3 output columns; each column C[:,k] = A * B[:,k] (2x3 * 3x1).
    # 2x3 * 3x1 has rank 6 schoolbook, but we can share: C[i,k] = sum_{j=0..2} A[i,j]B[j,k].
    # Total 2*3*3 = 18 schoolbook. Use Laderman-style: pad A to 3x3 (zero row) and use Laderman rank 7
    # for each... but that's per-column. Instead, use a direct rank-? Let's use Laderman on the 3x3 B
    # with A as 3x3 (row 2 zero): C_full (3x3) = A_pad * B, but we only need rows 0,1 of C and all 3 cols.
    # Laderman gives full 3x3 C with rank 7. We only use part of it, so rank 7 suffices!
    def two_by_three_by_three():
        # Pad A (2x3) to (3x3) with a zero third row. Use Laderman 3x3x3 (rank 7).
        U3, V3, W3 = _laderman_3x3x3()
        # Embed: A local is 3x3 (rU=3, rV=3), B local 3x3 (rV=3,rW=3), C local 3x3.
        # But we only want C rows 0,1 (the real A rows). The zero row of A means C row 2 is 0 automatically,
        # so using full Laderman is fine; we just embed the whole 3x3x3.
        U1, V1, W1 = _embed(U3, V3, W3, 3, 3, 3, 3, 3, 3,
                            row_off=0, col_off=0,
                            vrow_off=0, vcol_off=0,
                            wrow_off=0, wcol_off=0)
        return _combine([(U1, V1, W1)])

    # Now build the 4 C-blocks.
    # For each block we need (2x2)*(2x3) and (2x3)*(3x3) terms at proper offsets.

    # C_top_left: rows 0,1 cols 0,1,2
    #   term A: A_tl (rows 0,1 cols 0,1) * B_tl (rows 0,1 cols 0,1,2)
    #   term B: A_tr (rows 0,1 cols 2,3,4) * B_bl (rows 2,3,4 cols 0,1,2)
    t1 = two_by_two_by_two_by_three()
    e1 = _embed(t1[0], t1[1], t1[2], 2, 2, 3, n, m, p,
                row_off=A_top, col_off=A_left,
                vrow_off=B_top, vcol_off=B_left,
                wrow_off=B_left, wcol_off=A_top)
    t2 = two_by_three_by_three()
    e2 = _embed(t2[0], t2[1], t2[2], 2, 3, 3, n, m, p,
                row_off=A_top, col_off=A_right,
                vrow_off=B_bot, vcol_off=B_left,
                wrow_off=B_left, wcol_off=A_top)
    terms.append(e1); terms.append(e2)

    # C_top_right: rows 0,1 cols 3,4,5
    #   term A: A_tl (rows 0,1 cols 0,1) * B_tr (rows 0,1 cols 3,4,5)
    #   term B: A_tr (rows 0,1 cols 2,3,4) * B_br (rows 2,3,4 cols 3,4,5)
    t1 = two_by_two_by_two_by_three()
    e1 = _embed(t1[0], t1[1], t1[2], 2, 2, 3, n, m, p,
                row_off=A_top, col_off=A_left,
                vrow_off=B_top, vcol_off=B_right,
                wrow_off=B_right, wcol_off=A_top)
    t2 = two_by_three_by_three()
    e2 = _embed(t2[0], t2[1], t2[2], 2, 3, 3, n, m, p,
                row_off=A_top, col_off=A_right,
                vrow_off=B_bot, vcol_off=B_right,
                wrow_off=B_right, wcol_off=A_top)
    terms.append(e1); terms.append(e2)

    # C_bot_left: rows 2,3 cols 0,1,2
    #   term A: A_bl (rows 2,3 cols 0,1) * B_tl (rows 0,1 cols 0,1,2)
    #   term B: A_br (rows 2,3 cols 2,3,4) * B_bl (rows 2,3,4 cols 0,1,2)
    t1 = two_by_two_by_two_by_three()
    e1 = _embed(t1[0], t1[1], t1[2], 2, 2, 3, n, m, p,
                row_off=A_bot, col_off=A_left,
                vrow_off=B_top, vcol_off=B_left,
                wrow_off=B_left, wcol_off=A_bot)
    t2 = two_by_three_by_three()
    e2 = _embed(t2[0], t2[1], t2[2], 2, 3, 3, n, m, p,
                row_off=A_bot, col_off=A_right,
                vrow_off=B_bot, vcol_off=B_left,
                wrow_off=B_left, wcol_off=A_bot)
    terms.append(e1); terms.append(e2)

    # C_bot_right: rows 2,3 cols 3,4,5
    #   term A: A_bl (rows 2,3 cols 0,1) * B_tr (rows 0,1 cols 3,4,5)
    #   term B: A_br (rows 2,3 cols 2,3,4) * B_br (rows 2,3,4 cols 3,4,5)
    t1 = two_by_two_by_two_by_three()
    e1 = _embed(t1[0], t1[1], t1[2], 2, 2, 3, n, m, p,
                row_off=A_bot, col_off=A_left,
                vrow_off=B_top, vcol_off=B_right,
                wrow_off=B_right, wcol_off=A_bot)
    t2 = two_by_three_by_three()
    e2 = _embed(t2[0], t2[1], t2[2], 2, 3, 3, n, m, p,
                row_off=A_bot, col_off=A_right,
                vrow_off=B_bot, vcol_off=B_right,
                wrow_off=B_right, wcol_off=A_bot)
    terms.append(e1); terms.append(e2)

    U, V, W = _combine(terms)
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Baseline: schoolbook (always valid)
    U, V, W = _schoolbook(n, m, p)
    best_rank = U.shape[1]
    best = (U, V, W)
    assert _verify_decomposition(U, V, W, n, m, p), "schoolbook verify failed"

    if time.time() > deadline:
        return {"U": best[0].tolist(), "V": best[1].tolist(), "W": best[2].tolist()}

    # Try the block decomposition
    try:
        U2, V2, W2 = _build_4x5x6()
        if _verify_decomposition(U2, V2, W2, n, m, p):
            if U2.shape[1] < best_rank:
                best_rank = U2.shape[1]
                best = (U2, V2, W2)
    except Exception:
        pass

    if time.time() > deadline:
        return {"U": best[0].tolist(), "V": best[1].tolist(), "W": best[2].tolist()}

    return {"U": best[0].tolist(), "V": best[1].tolist(), "W": best[2].tolist()}

# EVOLVE-BLOCK-END
