import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100.0

    n, m, p = 2, 2, 16
    blocks = p // 2
    total_rank = blocks * 7

    U = np.zeros((n * m, total_rank), dtype=np.int8)
    V = np.zeros((m * p, total_rank), dtype=np.int8)
    W = np.zeros((p * n, total_rank), dtype=np.int8)

    # Strassen 2x2x2x2 decomposition: 7 factors
    # U indices: (0,1) for row0, (2,3) for row1 of the 2x2 matrix
    # V indices: (0,1) for row0, (2,3) for row1 of the 2x2 matrix  
    # W indices: (0,1) for col0, (2,3) for col1 of the 2x2 result
    strassen = [
        # s=0: (A00+A11)(B00+B11)
        ([0, 3], [0, 3], [0, 3]),
        # s=1: (A10+A11)B00
        ([2, 3], [0], [2, 3]),
        # s=2: A00(B10-B00)
        ([0], [1, 3], [1, 3]),
        # s=3: A11(B01-B11)
        ([3], [2, 0], [0, 2]),
        # s=4: (A00+A01)B11
        ([0, 1], [3], [0, 1]),
        # s=5: (A10-A00)(B00+B01)
        ([2, 0], [0, 1], [3]),
        # s=6: (A01-A11)(B10+B11)
        ([1, 3], [2, 3], [0]),
    ]

    # Signs for each factor
    signs = [
        [1, 1], [1, 1], [1, 1],
        [1, 1], [1], [1],
        [1, -1], [1, 1], [1, 1],
        [1], [1, -1], [1, 1],
        [1, 1], [1], [-1, 1],
        [1, -1], [1, 1], [1],
        [1, -1], [1, 1], [1],
    ]
    # Actually, let me use the explicit dictionary approach which is clearer
    strassen = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]

    for q in range(blocks):
        # B is 2x16, stored row-major: B[j, k] at index j*16 + k
        # For block q, columns 2q and 2q+1
        b_global = [0 * p + 2 * q, 0 * p + 2 * q + 1, 1 * p + 2 * q, 1 * p + 2 * q + 1]
        # C is 2x16, stored with convention W[k*n+i, r] where k is column, i is row
        # So C[i, k] is at index k*n + i = k*2 + i
        # For block q, columns 2q and 2q+1
        c_global = [(2 * q) * n + 0, (2 * q + 1) * n + 0, (2 * q) * n + 1, (2 * q + 1) * n + 1]

        for s in range(7):
            r = q * 7 + s
            u_dict, v_dict, w_dict = strassen[s]

            for u_row, val in u_dict.items():
                U[u_row, r] = val

            for v_local, val in v_dict.items():
                V[b_global[v_local], r] = val

            for w_local, val in w_dict.items():
                W[c_global[w_local], r] = val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
