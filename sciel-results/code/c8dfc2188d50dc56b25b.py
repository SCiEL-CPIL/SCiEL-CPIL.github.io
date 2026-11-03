import numpy as np
import time

DIMENSIONS = [4, 4, 7]
_START = time.time()
_DEADLINE = 130.0  # safety margin before 150s kill


def _strassen_2x2x2():
    """Rank-7 Strassen for 2x2x2.
    U[i*2+j], V[j*2+k], W[k*2+i].
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # p1 = (a00+a11)(b00+b11)
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
    # p2 = (a10+a11)b00
    U[2][1]=1; U[3][1]=1; V[0][1]=1; W[1][1]=1; W[3][1]=-1
    # p3 = a00(b01-b11)
    U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[2][2]=1; W[3][2]=1
    # p4 = a11(b10-b00)
    U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[1][3]=1
    # p5 = (a00+a01)b11
    U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[2][4]=1
    # p6 = (a10-a00)(b00+b01)
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
    # p7 = (a01-a11)(b10+b11)
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
    return U, V, W


def _verify_strassen():
    """Sanity-check the Strassen 2x2x2 tensor identity."""
    U, V, W = _strassen_2x2x2()
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                a = i*2+j; b = j*2+k; c = k*2+i
                s = float(np.dot(Ua[a], Va[b]*Wa[c]))
                if abs(s-1) > 1e-9:
                    return False
    return True


def _kron_4x4x4():
    """Kronecker product of two Strassen 2x2x2 -> 4x4x4, rank 49."""
    U2, V2, W2 = _strassen_2x2x2()
    n = m = p = 4
    rank = 49
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for r1 in range(7):
        for r2 in range(7):
            r = r1*7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1*2+i2; j = j1*2+j2
                            val = U2[i1*2+j1][r1]*U2[i2*2+j2][r2]
                            if val:
                                U[i*m+j][r] += val
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1*2+j2; k = k1*2+k2
                            val = V2[j1*2+k1][r1]*V2[j2*2+k2][r2]
                            if val:
                                V[j*p+k][r] += val
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1*2+k2; i = i1*2+i2
                            val = W2[k1*2+i1][r1]*W2[k2*2+i2][r2]
                            if val:
                                W[k*n+i][r] += val
    return U, V, W


def _add_2x2x2_strassen(U, V, W, base, arow, bcol, ccol, n, m, p):
    """Add a Strassen 2x2x2 block.
    arow: starting row of A block (2x2) -> A indices arow..arow+1 (rows), col offset via global j
    bcol: starting col of B block (2x2)
    ccol: starting col of C block (2x2)
    A block rows: arow, arow+1 ; A block cols: j0, j0+1  (j0 = arow? no)
    """
    # We need full mapping: A[i,j] with i in {arow,arow+1}, j in {j0,j0+1}
    # This helper is generic; caller passes explicit index maps.
    pass


def _build_full():
    n, m, p = 4, 4, 7
    U, V, W = _kron_4x4x4()
    rank = 49
    U2, V2, W2 = _strassen_2x2x2()

    # ---- 4x4x3 part: columns k=4,5,6 ----
    # A is 4x4. Split rows into top (0,1) and bottom (2,3).
    # For a C column kc (global k), C[i,kc] = sum_j A[i,j]*B[j,kc].
    # We handle B's two rows of the 2x1 column: B[0,kc], B[1,kc] (top) and B[2,kc],B[3,kc] (bottom).
    #
    # For each global output column kc in {4,5,6}:
    #   top C rows (i=0,1): C0 = A00*Btop0 + A01*Btop1  where
    #       A00 = A[0:2,0:2] (2x2), A01 = A[0:2,2:4] (2x2)
    #       Btop = B[0:2, kc] (2x1)
    #   bottom C rows (i=2,3): C1 = A10*Btop0 + A11*Btop1
    #       A10 = A[2:4,0:2], A11 = A[2:4,2:4]
    #
    # Each "A_block (2x2) * B_col (2x1)" is a 2x2x1 tensor, rank 4 schoolbook.
    # There are 4 such products per kc (A00, A01 for top; A10, A11 for bottom) -> 16 per kc.
    # 3 columns -> 48. That's the schoolbook tail we want to replace.
    #
    # STRASSEN BLOCK IDEA: combine the 2x2x1 with a 2x2x2 to save terms.
    # Instead, treat the 4x4x3 as: for the 2x2 column-pair (k=4,5) use 2x2x2 Strassen,
    # and the single column k=6 use 2x2x1.
    #
    # Split B's 3 columns into: B[:,4:6] (4x2) and B[:,6:7] (4x1).
    #
    # For B[:,4:6] (4x2), C[:,4:6] (4x2):
    #   C_top = A00*B00 + A01*B10   (2x2 = 2x2*2x2 + 2x2*2x2) -> 2 Strassen rank 7 = 14
    #   C_bot = A10*B00 + A11*B10   (2x2) -> 2 Strassen rank 7 = 14
    #   total 28 for the 2-column part.
    #
    # For B[:,6:7] (4x1), C[:,6:7] (4x1):
    #   C_top = A00*B0 + A01*B1  (2x1) -> 2 schoolbook 2x2x1 rank 4 = 8
    #   C_bot = A10*B0 + A11*B1  (2x1) -> 2 schoolbook 2x2x1 rank 4 = 8
    #   total 16 for the 1-column part.
    #
    # Grand 4x4x3 part: 28 + 16 = 44.

    def add_strassen_2x2x2(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        """Add Strassen for A[a_r0:a_r0+2, a_c0:a_c0+2] * B[b_r0:b_r0+2, b_c0:b_c0+2]
        -> C[c_r0:c_r0+2, c_c0:c_c0+2]."""
        for r1 in range(7):
            r = rank + r1
            for i1 in range(2):
                for j1 in range(2):
                    i = a_r0 + i1; j = a_c0 + j1
                    val = U2[i1*2+j1][r1]
                    if val:
                        U[i*m+j][r] += val
            for j1 in range(2):
                for k1 in range(2):
                    j = b_r0 + j1; k = b_c0 + k1
                    val = V2[j1*2+k1][r1]
                    if val:
                        V[j*p+k][r] += val
            for k1 in range(2):
                for i1 in range(2):
                    i = c_r0 + i1; k = c_c0 + k1
                    val = W2[k1*2+i1][r1]
                    if val:
                        W[k*n+i][r] += val
            rank += 1
        return rank

    def add_school_2x2x1(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        """Add schoolbook for A[a_r0:a_r0+2, a_c0:a_c0+2] (2x2) * B[b_r0:b_r0+2, b_c0] (2x1)
        -> C[c_r0:c_r0+2, c_c0] (2x1).  Rank 4."""
        # A is 2x2 (rows a_r0..+1, cols a_c0..+1), B is 2x1 (rows b_r0..+1, col b_c0),
        # C is 2x1 (rows c_r0..+1, col c_c0).
        # C[i,c_c0] = sum_j A[i,a_c0+j]*B[b_r0+j, b_c0]
        for i1 in range(2):
            for j1 in range(2):
                r = rank
                i = a_r0 + i1; j = a_c0 + j1
                U[i*m+j][r] = 1
                V[(b_r0+j1)*p + b_c0][r] = 1
                W[b_c0*n + i][r] = 1
                rank += 1
        return rank

    # 2-column part (k=4,5): B[:,4:6]
    # C_top (i=0,1) = A00*B00 + A01*B10
    rank = add_strassen_2x2x2(0, 0, 0, 4, 0, 4)   # A[0:2,0:2]*B[0:2,4:6] -> C[0:2,4:6]
    rank = add_strassen_2x2x2(0, 2, 2, 4, 0, 4)   # A[0:2,2:4]*B[2:4,4:6] -> C[0:2,4:6]
    # C_bot (i=2,3) = A10*B00 + A11*B10
    rank = add_strassen_2x2x2(2, 0, 0, 4, 2, 4)   # A[2:4,0:2]*B[0:2,4:6] -> C[2:4,4:6]
    rank = add_strassen_2x2x2(2, 2, 2, 4, 2, 4)   # A[2:4,2:4]*B[2:4,4:6] -> C[2:4,4:6]

    # 1-column part (k=6): B[:,6:7]
    # C_top (i=0,1) = A00*B0 + A01*B1
    rank = add_school_2x2x1(0, 0, 0, 6, 0, 6)     # A[0:2,0:2]*B[0:2,6] -> C[0:2,6]
    rank = add_school_2x2x1(0, 2, 2, 6, 0, 6)     # A[0:2,2:4]*B[2:4,6] -> C[0:2,6]
    # C_bot (i=2,3) = A10*B0 + A11*B1
    rank = add_school_2x2x1(2, 0, 0, 6, 2, 6)     # A[2:4,0:2]*B[0:2,6] -> C[2:4,6]
    rank = add_school_2x2x1(2, 2, 2, 6, 2, 6)     # A[2:4,2:4]*B[2:4,6] -> C[2:4,6]

    # Pad to final rank
    U = [row + [0]*(rank - len(row)) for row in U]
    V = [row + [0]*(rank - len(row)) for row in V]
    W = [row + [0]*(rank - len(row)) for row in W]
    return U, V, W, rank


def _verify(U, V, W, n, m, p):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j; b = j*p+k; c = k*n+i
                s = float(np.dot(Ua[a], Va[b]*Wa[c]))
                if abs(s-1) > 1e-9:
                    return False
    return True


def construct():
    global _START
    if time.time() - _START > _DEADLINE:
        # Fallback to schoolbook (should not happen)
        n, m, p = 4, 4, 7
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx]=1; V[j*p+k][idx]=1; W[k*n+i][idx]=1; idx+=1
        return {"U": U, "V": V, "W": W}

    n, m, p = 4, 4, 7
    U, V, W, rank = _build_full()
    # Verify; if broken, fall back to schoolbook.
    if not _verify(U, V, W, n, m, p):
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx]=1; V[j*p+k][idx]=1; W[k*n+i][idx]=1; idx+=1
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    assert _verify_strassen(), "Strassen 2x2x2 self-check failed"
    res = construct()
    n, m, p = 4, 4, 7
    r = len(res["U"][0])
    print("rank:", r)
    print("valid:", _verify(res["U"], res["V"], res["W"], n, m, p))
