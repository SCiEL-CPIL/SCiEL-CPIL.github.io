import numpy as np
import time

def construct():
    t0 = time.time()
    deadline = t0 + 138
    n, m, p = 6, 6, 6

    # Schoolbook 3x3 rank-27
    r3 = 27
    U33 = np.zeros((9, r3))
    V33 = np.zeros((9, r3))
    W33 = np.zeros((9, r3))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U33[i * 3 + j, col] = 1
                V33[j * 3 + k, col] = 1
                W33[k * 3 + i, col] = 1
                col += 1

    # Strassen 2x2 terms
    terms = [
        ({'A00': 1, 'A11': 1}, {'B00': 1, 'B11': 1}, {'C00': 1, 'C11': 1}),
        ({'A10': 1, 'A11': 1}, {'B00': 1}, {'C10': 1, 'C11': -1}),
        ({'A00': 1}, {'B01': 1, 'B11': -1}, {'C01': 1, 'C11': 1}),
        ({'A11': 1}, {'B10': 1, 'B00': -1}, {'C00': 1, 'C10': 1}),
        ({'A00': 1, 'A01': 1}, {'B11': 1}, {'C00': -1, 'C01': 1}),
        ({'A10': 1, 'A00': -1}, {'B00': 1, 'B01': 1}, {'C11': 1}),
        ({'A01': 1, 'A11': -1}, {'B10': 1, 'B11': 1}, {'C00': 1}),
    ]

    bmap = {
        'A00': (0, 0), 'A01': (0, 1), 'A10': (1, 0), 'A11': (1, 1),
        'B00': (0, 0), 'B01': (0, 1), 'B10': (1, 0), 'B11': (1, 1),
        'C00': (0, 0), 'C01': (0, 1), 'C10': (1, 0), 'C11': (1, 1),
    }

    R = 7 * r3
    U = np.zeros((36, R))
    V = np.zeros((36, R))
    W = np.zeros((36, R))

    for mi, (ta, tb, tc) in enumerate(terms):
        base = mi * r3
        for lc in range(r3):
            c = base + lc
            i3 = lc // 9
            rem = lc % 9
            j3 = rem // 3
            k3 = rem % 3
            u33v = U33[i3 * 3 + j3, lc]
            v33v = V33[j3 * 3 + k3, lc]
            w33v = W33[k3 * 3 + i3, lc]
            if u33v != 0:
                for ab, ca in ta.items():
                    bi, bj = bmap[ab]
                    U[(bi * 3 + i3) * 6 + (bj * 3 + j3), c] += ca * u33v
            if v33v != 0:
                for bb, cb in tb.items():
                    bk, bl = bmap[bb]
                    V[(bk * 3 + j3) * 6 + (bl * 3 + k3), c] += cb * v33v
            if w33v != 0:
                for cb2, cc in tc.items():
                    ci, ck = bmap[cb2]
                    W[(ck * 3 + k3) * 6 + (ci * 3 + i3), c] += cc * w33v

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
