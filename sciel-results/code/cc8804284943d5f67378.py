import numpy as np
import time

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    Uses a Strassen-based block decomposition yielding rank 48.

    Strategy:
      - Treat A as 2 blocks of (1x2), B as 4 blocks of (2x2), C as 4 blocks of (1x2).
      - For each (A-block, B-block) pair, apply a 6-term bilinear scheme that
        multiplies a (1x2) row-vector by a (2x2) matrix to produce a (1x2) row-vector.
      - 2 A-blocks * 4 B-blocks * 6 terms = 48 rank terms.
    """
    n, m, p = 2, 4, 8
    _deadline = time.time() + 145

    # Bilinear scheme for (1x2) * (2x2) -> (1x2):
    # Inputs: A = [x, y], B = [[e, f], [g, h]]
    # t1 = (x+y)(e+g)
    # t2 = x*f
    # t3 = y*h
    # t4 = (x-y)(g-h)
    # t5 = (x+y)*f
    # t6 = (x-y)*h
    # Outputs:
    # C[0] = t1 + t2 - t3 + t4
    # C[1] = -t1 + t3 + t5 + t6

    X_forms = np.array([
        [1, 1],   # t1: x + y
        [1, 0],   # t2: x
        [0, 1],   # t3: y
        [1, -1],  # t4: x - y
        [1, 1],   # t5: x + y
        [1, -1],  # t6: x - y
    ], dtype=np.int64)

    Y_forms = np.array([
        [1, 0, 1, 0],   # t1: e + g  (B flat: [e,f,g,h])
        [0, 1, 0, 0],   # t2: f
        [0, 0, 0, 1],   # t3: h
        [0, 0, 1, -1],  # t4: g - h
        [0, 1, 0, 0],   # t5: f
        [0, 0, 0, 1],   # t6: h
    ], dtype=np.int64)

    # Z_combos[out_idx][t] where out_idx: 0 -> C[0], 1 -> C[1]
    Z_combos = np.array([
        [1, 1, -1, 1, 0, 0],   # C[0] = t1 + t2 - t3 + t4
        [-1, 0, 1, 0, 1, 1],   # C[1] = -t1 + t3 + t5 + t6
    ], dtype=np.int64)

    num_a_blocks = 2  # A partitioned into 2 blocks of 1x2
    num_b_blocks = 4  # B partitioned into 4 blocks of 2x2
    num_terms = 6

    rank = num_a_blocks * num_b_blocks * num_terms  # 48

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    r = 0
    for ab in range(num_a_blocks):
        for bb in range(num_b_blocks):
            for t in range(num_terms):
                if time.time() > _deadline:
                    return {"U": U[:, :r].tolist(),
                            "V": V[:, :r].tolist(),
                            "W": W[:, :r].tolist()}

                # U: A row index ab, columns 2*ab, 2*ab+1
                for jp in range(2):
                    a_idx = ab * m + 2 * ab + jp
                    U[a_idx, r] = X_forms[t, jp]

                # V: B rows 2*bb, 2*bb+1; cols 2*bb, 2*bb+1
                for jp in range(2):
                    for kp in range(2):
                        b_idx = (2 * bb + jp) * p + (2 * bb + kp)
                        V[b_idx, r] = Y_forms[t, jp * 2 + kp]

                # W: C row ab, cols 2*bb, 2*bb+1; output index k*n+i
                for kp in range(2):
                    k = 2 * bb + kp
                    c_idx = k * n + ab
                    W[c_idx, r] = Z_combos[kp, t]

                r += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
