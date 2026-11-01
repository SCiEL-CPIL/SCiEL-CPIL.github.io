import numpy as np
import time

def construct():
    _t0 = time.time()
    _deadline = _t0 + 120.0

    n, m, p = 2, 3, 8
    num_blocks = p // 2  # 4 blocks
    rank_per_block = 11
    total_rank = rank_per_block * num_blocks  # 44

    U = np.zeros((n*m, total_rank))  # (6, 44)
    V = np.zeros((m*p, total_rank))  # (24, 44)
    W = np.zeros((p*n, total_rank))  # (16, 44)

    for blk in range(num_blocks):
        base = blk * rank_per_block
        b00 = 2*blk; b10 = p + 2*blk; b20 = 2*p + 2*blk
        b01 = 2*blk+1; b11 = p + 2*blk+1; b21 = 2*p + 2*blk+1
        c00 = 4*blk; c10 = 4*blk+1; c01 = 4*blk+2; c11 = 4*blk+3

        # m1 = (A00+A11)(B00+B11)
        U[0, base+0] = 1; U[4, base+0] = 1
        V[b00, base+0] = 1; V[b11, base+0] = 1
        W[c00, base+0] = 1; W[c11, base+0] = 1

        # m2 = (A10+A11)B00
        U[3, base+1] = 1; U[4, base+1] = 1
        V[b00, base+1] = 1
        W[c10, base+1] = 1; W[c11, base+1] = -1

        # m3 = A00(B01-B11)
        U[0, base+2] = 1
        V[b01, base+2] = 1; V[b11, base+2] = -1
        W[c01, base+2] = 1; W[c11, base+2] = 1

        # m4 = A11(B10-B00)
        U[4, base+3] = 1
        V[b10, base+3] = 1; V[b00, base+3] = -1
        W[c00, base+3] = 1; W[c10, base+3] = 1

        # m5 = (A00+A01)B11
        U[0, base+4] = 1; U[1, base+4] = 1
        V[b11, base+4] = 1
        W[c00, base+4] = -1; W[c01, base+4] = 1

        # m6 = (A10-A00)(B00+B01)
        U[3, base+5] = 1; U[0, base+5] = -1
        V[b00, base+5] = 1; V[b01, base+5] = 1
        W[c11, base+5] = 1

        # m7 = (A01-A11)(B10+B11)
        U[1, base+6] = 1; U[4, base+6] = -1
        V[b10, base+6] = 1; V[b11, base+6] = 1
        W[c00, base+6] = 1

        # Extra: A[i,2]*B[2,k]
        U[2, base+7] = 1; V[b20, base+7] = 1; W[c00, base+7] = 1
        U[2, base+8] = 1; V[b21, base+8] = 1; W[c01, base+8] = 1
        U[5, base+9] = 1; V[b20, base+9] = 1; W[c10, base+9] = 1
        U[5, base+10] = 1; V[b21, base+10] = 1; W[c11, base+10] = 1

    if time.time() > _deadline - 5.0:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
