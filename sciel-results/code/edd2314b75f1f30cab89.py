"""Program for 5x5x5 matrix multiplication rank-minimal exact decomposition."""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
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
    return U, V, W

def _kron2(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """Kronecker product of two decompositions -> (n1*n2, m1*m2, p1*p2) rank r1*r2.
    U row (i1*n2+i2, j1*m2+j2), col (c1*r2+c2) = U1[i1*m1+j1,c1]*U2[i2*m2+j2,c2]
    """
    r1 = len(U1[0]); r2 = len(U2[0])
    R = r1 * r2
    nN, mM, pP = n1*n2, m1*m2, p1*p2
    U = [[0]*R for _ in range(nN*mM)]
    V = [[0]*R for _ in range(mM*pP)]
    W = [[0]*R for _ in range(pP*nN)]
    for i1 in range(n1):
        for j1 in range(m1):
            for c1 in range(r1):
                u1 = U1[i1*m1+j1][c1]
                if u1 == 0: continue
                for i2 in range(n2):
                    for j2 in range(m2):
                        u2 = U2[i2*m2+j2][c2 if False else 0]  # placeholder
    # Do it properly with full loops:
    U = [[0]*R for _ in range(nN*mM)]
    V = [[0]*R for _ in range(mM*pP)]
    W = [[0]*R for _ in range(pP*nN)]
    for i1 in range(n1):
        for j1 in range(m1):
            for c1 in range(r1):
                u1 = U1[i1*m1+j1][c1]
                if u1 == 0: continue
                for i2 in range(n2):
                    for j2 in range(m2):
                        for c2 in range(r2):
                            u2 = U2[i2*m2+j2][c2]
                            if u2 == 0: continue
                            val = u1*u2
                            if val == 0: continue
                            ii = i1*n2 + i2
                            jj = j1*m2 + j2
                            cc = c1*r2 + c2
                            U[ii*mM+jj][cc] += val
    for j1 in range(m1):
        for k1 in range(p1):
            for c1 in range(r1):
                v1 = V1[j1*p1+k1][c1]
                if v1 == 0: continue
                for j2 in range(m2):
                    for k2 in range(p2):
                        for c2 in range(r2):
                            v2 = V2[j2*p2+k2][c2]
                            if v2 == 0: continue
                            val = v1*v2
                            if val == 0: continue
                            jj = j1*m2 + j2
                            kk = k1*p2 + k2
                            cc = c1*r2 + c2
                            V[jj*pP+kk][cc] += val
    for k1 in range(p1):
        for i1 in range(n1):
            for c1 in range(r1):
                w1 = W1[k1*n1+i1][c1]
                if w1 == 0: continue
                for k2 in range(p2):
                    for i2 in range(n2):
                        for c2 in range(r2):
                            w2 = W2[k2*n2+i2][c2]
                            if w2 == 0: continue
                            val = w1*w2
                            if val == 0: continue
                            kk = k1*p2 + k2
                            ii = i1*n2 + i2
                            cc = c1*r2 + c2
                            W[kk*nN+ii][cc] += val
    return U, V, W

def construct():
    t0 = time.time()
    deadline = t0 + 120.0
    n, m, p = DIMENSIONS

    def schoolbook_result():
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

    # Build 5x5x5 = 4x4x4 (top-left, rank 16 via 2x2x2 ⊗ 2x2x2) + cross terms (schoolbook) + (4,4,4).
    # Indices 0..3 form the 4x4x4 block; index 4 is the 5th row/col.
    #
    # The 4x4x4 block: U4,V4,W4 of shape (16,16) each, rank 16.
    U2, V2, W2 = _schoolbook(2, 2, 2)
    U4, V4, W4 = _kron2(U2, V2, W2, U2, V2, W2, 2, 2, 2, 2, 2, 2)

    # Cross terms involving index 4 (the 5th), via schoolbook over the "1" dimension:
    # Terms: (i in 0..3, j=4, k in 0..3): A[i,4]*B[4,k] -> C[i,k]
    #        (i in 0..3, j in 0..3, k=4): A[i,j]*B[j,4] -> C[i,4]
    #        (i=4, j in 0..3, k in 0..3): A[4,j]*B[j,k] -> C[4,k]
    #        (i=4, j=4, k=4): A[4,4]*B[4,4] -> C[4,4]
    #
    # We'll assemble the full 5x5x5 U,V,W (shape 25 x R) by placing the 4x4x4 block
    # in rows/cols with indices 0..3 and the cross terms in the appropriate rows.

    # Full rank = 16 (block) + 4 (i,j=4,k) + 4 (i,j,k=4) + 4 (i=4,j,k) + 1 (4,4,4) = 29
    R = 29
    U = [[0]*R for _ in range(25)]
    V = [[0]*R for _ in range(25)]
    W = [[0]*R for _ in range(25)]

    # Place 4x4x4 block: for i,j,k in 0..3, U[i*5+j, c] = U4[i*4+j, c], etc.
    for i in range(4):
        for j in range(4):
            for c in range(16):
                U[i*5+j][c] = U4[i*4+j][c]
    for j in range(4):
        for k in range(4):
            for c in range(16):
                V[j*5+k][c] = V4[j*4+k][c]
    for k in range(4):
        for i in range(4):
            for c in range(16):
                W[k*5+i][c] = W4[k*4+i][c]

    # Cross term group 1: j=4, i in 0..3, k in 0..3 (4 terms)
    col = 16
    for i in range(4):
        for k in range(4):
            U[i*5+4][col] = 1
            V[4*5+k][col] = 1
            W[k*5+i][col] = 1
            col += 1
    # Cross term group 2: k=4, i in 0..3, j in 0..3 (4 terms)
    for i in range(4):
        for j in range(4):
            U[i*5+j][col] = 1
            V[j*5+4][col] = 1
            W[4*5+i][col] = 1
            col += 1
    # Cross term group 3: i=4, j in 0..3, k in 0..3 (4 terms)
    for j in range(4):
        for k in range(4):
            U[4*5+j][col] = 1
            V[j*5+k][col] = 1
            W[k*5+4][col] = 1
            col += 1
    # Cross term group 4: i=4, j=4, k=4 (1 term)
    U[4*5+4][col] = 1
    V[4*5+4][col] = 1
    W[4*5+4][col] = 1
    col += 1

    assert col == R, f"col={col} R={R}"

    # Verify tensor identity: sum_c U[i*5+j,c] V[j*5+k,c] W[k*5+i,c] == 1 for all i,j,k
    ok = True
    for i in range(5):
        for j in range(5):
            for k in range(5):
                s = 0
                for c in range(R):
                    s += U[i*5+j][c] * V[j*5+k][c] * W[k*5+i][c]
                if s != 1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break

    if not ok or time.time() > deadline - 5:
        return schoolbook_result()

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
