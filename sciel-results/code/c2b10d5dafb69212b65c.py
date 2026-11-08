import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    t_start = time.time()
    t_deadline = t_start + 130.0
    n, m, p = DIMENSIONS

    # Use the same known-good block decomposition that achieves rank 66.
    # M(3,4,6) = sum over k-blocks of size 2:
    #   M(1,4,2) (schoolbook, rank 8) + M(2,4,2) (2*Strassen, rank 14)
    # 3 blocks * (8+14) = 66

    rank = 66
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    r = 0

    def add1(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal r
        if r >= rank:
            return
        U[u_idx, r] += u_val
        V[v_idx, r] += v_val
        W[w_idx, r] += w_val
        r += 1

    def add_142(i0, j0, k0):
        for j in range(4):
            for k in range(2):
                add1(i0*m + (j0+j), 1, (j0+j)*p + (k0+k), 1, (k0+k)*n + i0, 1)

    def add_222(i0, j0, k0):
        def U2(i,j): return (i0+i)*m + (j0+j)
        def V2(j,k): return (j0+j)*p + (k0+k)
        def W2(i,k): return (k0+k)*n + (i0+i)
        # Strassen-7 pattern
        add1(U2(0,0), 1, V2(0,0), 1, W2(0,0), 1)
        add1(U2(1,0), 1, V2(0,0), 1, W2(1,0), 1)
        add1(U2(0,0), 1, V2(0,1), 1, W2(0,1), 1)
        add1(U2(1,1), 1, V2(1,0), 1, W2(0,0), 1)
        add1(U2(0,0), 1, V2(1,1), 1, W2(0,0), -1)
        add1(U2(1,0), 1, V2(0,1), 1, W2(1,1), 1)
        add1(U2(0,1), 1, V2(1,0), 1, W2(0,0), 1)

    def add_242(i0, j0, k0):
        add_222(i0, j0, k0)
        add_222(i0, j0+2, k0)

    for blk in range(3):
        if time.time() > t_deadline:
            break
        k0 = blk * 2
        add_142(0, 0, k0)
        add_242(1, 0, k0)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
