import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    n, m, p = DIMENSIONS  # 2, 3, 10

    # Strategy: Strassen on 2x2x2 blocks for the first 2 columns of B,
    # schoolbook for the 3rd column.
    # A is 2x3, B is 3x10, C is 2x10
    # Split B into B_top (2x10) and B_bottom (1x10)
    # C = A_left @ B_top + A_right @ B_bottom
    # A_left is 2x2, A_right is 2x1
    # For A_left @ B_top: split B_top into 5 blocks of 2 columns, use Strassen (7 terms each) -> 35 terms
    # For A_right @ B_bottom: schoolbook 2*1*10 = 20 terms
    # Total: 55

    rank = 55
    U = np.zeros((n * m, rank), dtype=np.int32)  # 6 x 55
    V = np.zeros((m * p, rank), dtype=np.int32)  # 30 x 55
    W = np.zeros((p * n, rank), dtype=np.int32)  # 20 x 55

    r = 0  # current rank index

    # Strassen's 7 products for 2x2 x 2x2:
    # Given A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h), m2 = (c+d)e, m3 = a(f-h), m4 = d(g-e)
    # m5 = (a+b)h, m6 = (c-a)(e+f), m7 = (b-d)(g+h)
    # C[0,0]=m1+m4-m5+m7, C[0,1]=m3+m5, C[1,0]=m2+m4, C[1,1]=m1-m2+m3+m6

    # A_left: A[0,0]=a->U[0], A[0,1]=b->U[1], A[1,0]=c->U[3], A[1,1]=d->U[4]
    # B_block: B[0,2t]=e->V[2t], B[0,2t+1]=f->V[2t+1], B[1,2t]=g->V[10+2t], B[1,2t+1]=h->V[10+2t+1]
    # C: C[0,2t]->W[4t], C[1,2t]->W[4t+1], C[0,2t+1]->W[4t+2], C[1,2t+1]->W[4t+3]

    strassen_u = [
        [(0, 1), (4, 1)],   # m1: a + d
        [(3, 1), (4, 1)],   # m2: c + d
        [(0, 1)],           # m3: a
        [(4, 1)],           # m4: d
        [(0, 1), (1, 1)],   # m5: a + b
        [(3, 1), (0, -1)],  # m6: c - a
        [(1, 1), (4, -1)],  # m7: b - d
    ]

    strassen_v = [
        [(0, 1), (11, 1)],  # m1: e + h
        [(0, 1)],           # m2: e
        [(1, 1), (11, -1)], # m3: f - h
        [(10, 1), (0, -1)], # m4: g - e
        [(11, 1)],          # m5: h
        [(0, 1), (1, 1)],   # m6: e + f
        [(10, 1), (11, 1)], # m7: g + h
    ]

    strassen_w = [
        [(0, 1), (3, 1)],   # m1: C[0,2t] + C[1,2t+1]
        [(1, 1), (3, -1)],  # m2: C[1,2t] - C[1,2t+1]
        [(2, 1), (3, 1)],   # m3: C[0,2t+1] + C[1,2t+1]
        [(0, 1), (1, 1)],   # m4: C[0,2t] + C[1,2t]
        [(0, -1), (2, 1)],  # m5: -C[0,2t] + C[0,2t+1]
        [(3, 1)],           # m6: C[1,2t+1]
        [(0, 1)],           # m7: C[0,2t]
    ]

    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1

        for s in range(7):
            for (u_row, coeff) in strassen_u[s]:
                U[u_row, r] += coeff

            for (v_row_offset, coeff) in strassen_v[s]:
                if v_row_offset == 0:
                    v_row = k0
                elif v_row_offset == 1:
                    v_row = k1
                elif v_row_offset == 10:
                    v_row = 10 + k0
                elif v_row_offset == 11:
                    v_row = 10 + k1
                else:
                    raise ValueError(f"Unexpected v_row_offset {v_row_offset}")
                V[v_row, r] += coeff

            for (w_row_offset, coeff) in strassen_w[s]:
                if w_row_offset == 0:
                    w_row = k0 * 2
                elif w_row_offset == 1:
                    w_row = k0 * 2 + 1
                elif w_row_offset == 2:
                    w_row = k1 * 2
                elif w_row_offset == 3:
                    w_row = k1 * 2 + 1
                else:
                    raise ValueError(f"Unexpected w_row_offset {w_row_offset}")
                W[w_row, r] += coeff

            r += 1

    # Schoolbook for A_right (2x1) @ B_bottom (1x10)
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2, r] = 1
            V[2 * 10 + k, r] = 1
            W[k * 2 + i, r] = 1
            r += 1

    assert r == rank, f"Expected rank {rank}, got {r}"

    # Verify the decomposition using numpy
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = np.dot(U[i * m + j] * V[j * p + k], W[k * n + i])
                if lhs != 1:
                    # Fallback to trivial decomposition
                    rank_trivial = n * m * p
                    U_t = np.zeros((n * m, rank_trivial), dtype=np.int32)
                    V_t = np.zeros((m * p, rank_trivial), dtype=np.int32)
                    W_t = np.zeros((p * n, rank_trivial), dtype=np.int32)
                    for rr in range(rank_trivial):
                        ii = rr // (m * p)
                        rem = rr % (m * p)
                        jj = rem // p
                        kk = rem % p
                        U_t[ii * m + jj, rr] = 1
                        V_t[jj * p + kk, rr] = 1
                        W_t[kk * n + ii, rr] = 1
                    return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
