import numpy as np
import time

_START = time.monotonic()
_LIMIT = 140.0

def construct():
    """Rank-56 exact tensor decomposition for 2x4 * 4x8 matrix multiplication."""
    n, m, p = 2, 4, 8
    R = 56

    U = np.zeros((n * m, R), dtype=np.float64)
    V = np.zeros((m * p, R), dtype=np.float64)
    W = np.zeros((p * n, R), dtype=np.float64)

    # Seven Strassen sub-programs; each entry is (a_vec, b_vec, c_vec)
    # a_vec: [a00, a01, a10, a11] for the 2x2 A block
    # b_vec: [b00, b01, b10, b11] for the 2x2 B block
    # c_vec: [c00, c01, c10, c11] for the 2x2 C block
    subs = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]

    # Pre-compute global index mappings for each (s, j) pair
    # s in 0..1 selects the 2-row block of B and 2-col block of A
    # j in 0..3 selects the 2-col block of B and C
    idx_U = []
    idx_V = []
    idx_W = []
    for s in range(2):
        for j in range(4):
            # U indices: row i (0,1) x col (2s, 2s+1)
            u_idx = [0 * 4 + 2 * s, 0 * 4 + 2 * s + 1,
                      1 * 4 + 2 * s, 1 * 4 + 2 * s + 1]
            # V indices: row (2s, 2s+1) x col (2j, 2j+1)
            v_idx = [2 * s * 8 + 2 * j, 2 * s * 8 + 2 * j + 1,
                      (2 * s + 1) * 8 + 2 * j, (2 * s + 1) * 8 + 2 * j + 1]
            # W indices: k*n + i => k in (2j, 2j+1), i in (0,1)
            w_idx = [2 * j * 2 + 0, (2 * j + 1) * 2 + 0,
                      2 * j * 2 + 1, (2 * j + 1) * 2 + 1]
            idx_U.append(u_idx)
            idx_V.append(v_idx)
            idx_W.append(w_idx)

    blk = 0
    for j in range(4):
        for s in range(2):
            r0 = blk * 7
            for t in range(7):
                r = r0 + t
                a_v, b_v, c_v = subs[t]
                for k in range(4):
                    if a_v[k]:
                        U[idx_U[blk][k], r] += a_v[k]
                    if b_v[k]:
                        V[idx_V[blk][k], r] += b_v[k]
                    if c_v[k]:
                        W[idx_W[blk][k], r] += c_v[k]
            blk += 1

    if time.monotonic() - _START > _LIMIT:
        raise RuntimeError("timeout")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    out = construct()
    assert len(out["U"]) == 8
    assert len(out["V"]) == 32
    assert len(out["W"]) == 16
    assert len(out["U"][0]) == 56
    print("rank =", len(out["U"][0]))
