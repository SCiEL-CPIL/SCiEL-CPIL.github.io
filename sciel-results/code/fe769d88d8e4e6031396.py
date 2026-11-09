import time
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen terms for 2x2 matrix multiplication
    # U entries: (local_index, coeff) where local_index = i*2+j for A[i,j]
    # V entries: (local_index, coeff) where local_index = j*2+k for B[j,k]
    # W entries: (local_index, coeff) where local_index = k*2+i for C[k,i]
    strassen_terms = [
        # M1 = (a+d)(e+h): U={a:1,d:1}, V={e:1,h:1}, W={C00:1,C11:1}
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        # M2 = (c+d)e: U={c:1,d:1}, V={e:1}, W={C10:1,C11:-1}
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(1, 1), (3, -1)]},
        # M3 = a(f-h): U={a:1}, V={f:1,h:-1}, W={C01:1,C11:1}
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(2, 1), (3, 1)]},
        # M4 = d(g-e): U={d:1}, V={g:1,e:-1}, W={C00:1,C10:1}
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (1, 1)]},
        # M5 = (a+b)h: U={a:1,b:1}, V={h:1}, W={C00:-1,C01:1}
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (2, 1)]},
        # M6 = (c-a)(e+f): U={c:1,a:-1}, V={e:1,f:1}, W={C11:1}
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        # M7 = (b-d)(g+h): U={b:1,d:-1}, V={g:1,h:1}, W={C00:1}
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]},
    ]

    # 4 blocks: split A's 4 columns into 2 groups of 2, B's 4 rows into 2 groups of 2
    blocks = [
        {'j_offset': 0, 'k_offset': 0},
        {'j_offset': 0, 'k_offset': 2},
        {'j_offset': 2, 'k_offset': 0},
        {'j_offset': 2, 'k_offset': 2},
    ]

    col = 0
    for block in blocks:
        j_off = block['j_offset']
        k_off = block['k_offset']
        for term in strassen_terms:
            for (idx, coeff) in term['U']:
                i = idx // 2
                j = idx % 2
                gi = i
                gj = j + j_off
                U[gi * m + gj][col] += coeff

            for (idx, coeff) in term['V']:
                j = idx // 2
                k = idx % 2
                gj = j + j_off
                gk = k + k_off
                V[gj * p + gk][col] += coeff

            for (idx, coeff) in term['W']:
                k = idx // 2
                i = idx % 2
                gk = k + k_off
                gi = i
                W[gk * n + gi][col] += coeff

            col += 1

    return {"U": U, "V": V, "W": W}
