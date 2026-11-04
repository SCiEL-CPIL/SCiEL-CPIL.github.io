import numpy as np
import time

_DIMS = (4, 4, 9)


def _strassen_2x2():
    U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[1,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W


def _strassen_4x4():
    U2, V2, W2 = _strassen_2x2()
    U = np.zeros((16, 49)); V = np.zeros((16, 49)); W = np.zeros((16, 49))
    blocks = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    for bi, (Ab, Bb, Cb) in enumerate(blocks):
        for s in range(7):
            r = bi*7+s
            for (bi_, bk_, ca) in Ab:
                for i2 in range(2):
                    for k2 in range(2):
                        U[2*bi_+i2, r] += ca*U2[i2*2+k2, s]
                        # Note: U index is i*4+k but here we use i (4-row index)
                        # Actually U is (16,49), index i*4+k. Let me fix.
            for (bk_, bj_, cb) in Bb:
                for k2 in range(2):
                    for j2 in range(2):
                        V[2*bk_+k2, r] += cb*V2[k2*2+j2, s]
                        # V index k*4+j
            for (cj_, ci_, cc) in Cb:
                for j2 in range(2):
                    for i2 in range(2):
                        W[2*cj_+j2, r] += cc*W2[j2*2+i2, s]
    # The above has index bugs; rebuild cleanly below
    return U, V, W


def _strassen_4x4_fixed():
    U2, V2, W2 = _strassen_2x2()
    U = np.zeros((16, 49)); V = np.zeros((16, 49)); W = np.zeros((16, 49))
    blocks = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    for bi, (Ab, Bb, Cb) in enumerate(blocks):
        for s in range(7):
            r = bi*7+s
            for (bi_, bk_, ca) in Ab:
                for i2 in range(2):
                    for k2 in range(2):
                        i = 2*bi_+i2; k = 2*bk_+k2
                        U[i*4+k, r] += ca*U2[i2*2+k2, s]
            for (bk_, bj_, cb) in Bb:
                for k2 in range(2):
                    for j2 in range(2):
                        k = 2*bk_+k2; j = 2*bj_+j2
                        V[k*4+j, r] += cb*V2[k2*2+j2, s]
            for (cj_, ci_, cc) in Cb:
                for j2 in range(2):
                    for i2 in range(2):
                        j = 2*cj_+j2; i = 2*ci_+i2
                        W[j*4+i, r] += cc*W2[j2*2+i2, s]
    return U, V, W


def _direct_4x4x4():
    U = np.zeros((16, 16)); V = np.zeros((16, 16)); W = np.zeros((16, 16))
    for r in range(16):
        i = r//4; k = r%4
        U[i*4+k, r] = 1
        V[k*4+i, r] = 1
        W[i*4+k, r] = 1
    return U, V, W


def construct():
    t0 = time.time()
    DEADLINE = t0 + 135

    n, m, p = _DIMS
    U4, V4, W4 = _strassen_4x4_fixed()
    rank = 114
    U = np.zeros((16, rank)); V = np.zeros((36, rank)); W = np.zeros((36, rank))

    # Block 1: cols 0-3
    U[:, 0:49] = U4
    for k in range(4):
        for j in range(4):
            V[k*9+j, 0:49] = V4[k*4+j, :]
    for j in range(4):
        for i in range(4):
            W[j*4+i, 0:49] = W4[j*4+i, :]

    # Block 2: cols 4-7
    U[:, 49:98] = U4
    for k in range(4):
        for j in range(4):
            V[k*9+(j+4), 49:98] = V4[k*4+j, :]
    for j in range(4):
        for i in range(4):
            W[(j+4)*4+i, 49:98] = W4[j*4+i, :]

    # Block 3: col 8 (direct)
    for i in range(4):
        for k in range(4):
            r = 98 + i*4 + k
            U[i*4+k, r] = 1
            V[k*9+8, r] = 1
            W[8*4+i, r] = 1

    # Verification
    np.random.seed(42)
    A = np.random.randn(4, 4)
    B = np.random.randn(4, 9)
    C_exp = A @ B

    Ur = U.reshape(4, 4, rank)
    Vr = V.reshape(4, 9, rank)
    Wr = W.reshape(9, 4, rank)

    a = np.einsum('ikr,ik->ir', Ur, A)
    b = np.einsum('kjr,kj->jr', Vr, B)
    C = np.einsum('jir,ir,jr->ij', Wr, a, b)

    if not np.allclose(C, C_exp, atol=1e-10):
        rank = 144
        U = np.zeros((16, rank)); V = np.zeros((36, rank)); W = np.zeros((36, rank))
        idx = 0
        for i in range(4):
            for j in range(9):
                for k in range(4):
                    U[i*4+k, idx] = 1
                    V[k*9+j, idx] = 1
                    W[j*4+i, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
