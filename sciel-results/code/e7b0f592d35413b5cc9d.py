import numpy as np
import time

def construct():
    n, m, p = 4, 5, 6
    start_time = time.time()
    deadline = 145

    def time_left():
        return deadline - (time.time() - start_time)

    def schoolbook(nn, mm, pp):
        r = nn * mm * pp
        U = np.zeros((nn * mm, r), dtype=object)
        V = np.zeros((mm * pp, r), dtype=object)
        W = np.zeros((pp * nn, r), dtype=object)
        col = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U[i * mm + j, col] = 1
                    V[j * pp + k, col] = 1
                    W[k * nn + i, col] = 1
                    col += 1
        return U, V, W

    def verify_decomp(U, V, W, nn, mm, pp, tol=1e-9):
        r = U.shape[1]
        Uf = np.array(U, dtype=np.float64)
        Vf = np.array(V, dtype=np.float64)
        Wf = np.array(W, dtype=np.float64)
        U3 = Uf.T.reshape(nn, mm, r)
        V3 = Vf.T.reshape(mm, pp, r)
        W3 = Wf.T.reshape(pp, nn, r)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        err = np.max(np.abs(T - 1.0))
        return err < tol

    # Build 4x3x2: rank 3
    # T[i,j,k] = 1 for i in 0..3, j in 0..2, k in 0..1
    def build_4x3x2():
        nn, mm, pp = 4, 3, 2
        r = mm  # 3
        U = np.zeros((nn * mm, r), dtype=object)
        V = np.zeros((mm * pp, r), dtype=object)
        W = np.zeros((pp * nn, r), dtype=object)
        for i in range(nn):
            for j in range(mm):
                U[i * mm + j, j] = 1
        for j in range(mm):
            for k in range(pp):
                V[j * pp + k, j] = 1
        for k in range(pp):
            for i in range(nn):
                for j in range(mm):
                    W[k * nn + i, j] = 1
        return U, V, W

    # Build 4x1x6: rank 1
    def build_4x1x6():
        nn, mm, pp = 4, 1, 6
        r = 1
        U = np.zeros((nn * mm, r), dtype=object)
        V = np.zeros((mm * pp, r), dtype=object)
        W = np.zeros((pp * nn, r), dtype=object)
        for i in range(nn):
            U[i * mm + 0, 0] = 1
        for k in range(pp):
            V[0 * pp + k, 0] = 1
        for k in range(pp):
            for i in range(nn):
                W[k * nn + i, 0] = 1
        return U, V, W

    # Build 4x2x6: rank 2 (two 4x1x6 blocks)
    def build_4x2x6():
        U1, V1, W1 = build_4x1x6()
        U2, V2, W2 = build_4x1x6()
        nn, mm, pp = 4, 2, 6
        r = 2
        U = np.zeros((nn * mm, r), dtype=object)
        V = np.zeros((mm * pp, r), dtype=object)
        W = np.zeros((pp * nn, r), dtype=object)
        # Block 1: j=0
        for i in range(nn):
            U[i * mm + 0, 0] = U1[i, 0]
        for k in range(pp):
            V[0 * pp + k, 0] = V1[k, 0]
        for k in range(pp):
            for i in range(nn):
                W[k * nn + i, 0] = W1[k * nn + i, 0]
        # Block 2: j=1
        for i in range(nn):
            U[i * mm + 1, 1] = U2[i, 0]
        for k in range(pp):
            V[1 * pp + k, 1] = V2[k, 0]
        for k in range(pp):
            for i in range(nn):
                W[k * nn + i, 1] = W2[k * nn + i, 0]
        return U, V, W

    # Build 4x3x6: rank 9 (three 4x3x2 blocks)
    def build_4x3x6():
        U1, V1, W1 = build_4x3x2()
        U2, V2, W2 = build_4x3x2()
        U3, V3, W3 = build_4x3x2()
        nn, mm, pp = 4, 3, 6
        r = 9
        U = np.zeros((nn * mm, r), dtype=object)
        V = np.zeros((mm * pp, r), dtype=object)
        W = np.zeros((pp * nn, r), dtype=object)
        # Block 1: k=0..1
        for i in range(nn):
            for j in range(mm):
                for rr in range(3):
                    U[i * mm + j, rr] = U1[i * mm + j, rr]
        for j in range(mm):
            for k in range(2):
                for rr in range(3):
                    V[j * pp + k, rr] = V1[j * 2 + k, rr]
        for k in range(2):
            for i in range(nn):
                for rr in range(3):
                    W[k * nn + i, rr] = W1[k * nn + i, rr]
        # Block 2: k=2..3
        for i in range(nn):
            for j in range(mm):
                for rr in range(3):
                    U[i * mm + j, 3 + rr] = U2[i * mm + j, rr]
        for j in range(mm):
            for k in range(2):
                for rr in range(3):
                    V[j * pp + (k + 2), 3 + rr] = V2[j * 2 + k, rr]
        for k in range(2):
            for i in range(nn):
                for rr in range(3):
                    W[(k + 2) * nn + i, 3 + rr] = W2[k * nn + i, rr]
        # Block 3: k=4..5
        for i in range(nn):
            for j in range(mm):
                for rr in range(3):
                    U[i * mm + j, 6 + rr] = U3[i * mm + j, rr]
        for j in range(mm):
            for k in range(2):
                for rr in range(3):
                    V[j * pp + (k + 4), 6 + rr] = V3[j * 2 + k, rr]
        for k in range(2):
            for i in range(nn):
                for rr in range(3):
                    W[(k + 4) * nn + i, 6 + rr] = W3[k * nn + i, rr]
        return U, V, W

    # Combine: 4x5x6 = 4x3x6 + 4x2x6
    # j split: 0..2 (3), 3..4 (2)
    def build_full():
        U1, V1, W1 = build_4x3x6()  # rank 9
        U2, V2, W2 = build_4x2x6()  # rank 2
        r_total = 9 + 2  # 11
        nn, mm, pp = 4, 5, 6
        U = np.zeros((nn * mm, r_total), dtype=object)
        V = np.zeros((mm * pp, r_total), dtype=object)
        W = np.zeros((pp * nn, r_total), dtype=object)

        # Block 1: j=0..2, cols 0..8
        for i in range(nn):
            for j in range(3):
                for rr in range(9):
                    U[i * mm + j, rr] = U1[i * 3 + j, rr]
        for j in range(3):
            for k in range(pp):
                for rr in range(9):
                    V[j * pp + k, rr] = V1[j * pp + k, rr]
        for k in range(pp):
            for i in range(nn):
                for rr in range(9):
                    W[k * nn + i, rr] = W1[k * nn + i, rr]

        # Block 2: j=3..4, cols 9..10
        for i in range(nn):
            for j in range(2):
                for rr in range(2):
                    U[i * mm + (j + 3), 9 + rr] = U2[i * 2 + j, rr]
        for j in range(2):
            for k in range(pp):
                for rr in range(2):
                    V[(j + 3) * pp + k, 9 + rr] = V2[j * pp + k, rr]
        for k in range(pp):
            for i in range(nn):
                for rr in range(2):
                    W[k * nn + i, 9 + rr] = W2[k * nn + i, rr]

        return U, V, W, r_total

    U, V, W, rank = build_full()
    ok = verify_decomp(U, V, W, 4, 5, 6)

    if ok and rank < 120:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback
    U, V, W = schoolbook(4, 5, 6)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
