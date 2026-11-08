import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 84
    
    # U: (n*m, rank) = (16, 84)
    # V: (m*p, rank) = (24, 84)
    # W: (p*n, rank) = (24, 84)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # Each entry is a list of (local_idx, coeff) for A, B, C
    # local_idx is (row, col) in 2x2 block
    strassen = [
        # r=0: P1 = (a+d)(e+h)
        {
            'U': [((0,0), 1), ((1,1), 1)],
            'V': [((0,0), 1), ((1,1), 1)],
            'W': [((0,0), 1), ((1,1), 1)]
        },
        # r=1: P2 = (c+d)e
        {
            'U': [((1,0), 1), ((1,1), 1)],
            'V': [((0,0), 1)],
            'W': [((1,0), 1), ((1,1), -1)]
        },
        # r=2: P3 = a(f-h)
        {
            'U': [((0,0), 1)],
            'V': [((0,1), 1), ((1,1), -1)],
            'W': [((0,1), 1), ((1,1), 1)]
        },
        # r=3: P4 = d(g-f)
        {
            'U': [((1,1), 1)],
            'V': [((1,0), 1), ((0,1), -1)],
            'W': [((0,0), 1), ((1,0), 1)]
        },
        # r=4: P5 = (a+b)h
        {
            'U': [((0,0), 1), ((0,1), 1)],
            'V': [((1,1), 1)],
            'W': [((0,0), -1), ((0,1), 1)]
        },
        # r=5: P6 = (c-a)(e+f)
        {
            'U': [((1,0), 1), ((0,0), -1)],
            'V': [((0,0), 1), ((0,1), 1)],
            'W': [((1,1), 1)]
        },
        # r=6: P7 = (b-d)(g+h)
        {
            'U': [((0,1), 1), ((1,1), -1)],
            'V': [((1,0), 1), ((1,1), 1)],
            'W': [((0,0), 1)]
        }
    ]
    
    col = 0
    # I: 0,1 (row block of A and C)
    # K: 0,1,2 (col block of B and C)
    for I in range(2):
        for K in range(3):
            # s: 0,1 (which column block of A and row block of B)
            for s in range(2):
                J = s # J is the column block of A and row block of B
                for r in range(7):
                    # Map local (i', j') to global indices
                    # A global: row = 2*I + i', col = 2*J + j'
                    # B global: row = 2*J + j', col = 2*K + k'
                    # C global: row = 2*I + i', col = 2*K + k'
                    
                    for (li, lj), coeff in strassen[r]['U']:
                        gi = 2 * I + li
                        gj = 2 * J + lj
                        U[gi * m + gj][col] = coeff
                    
                    for (li, lj), coeff in strassen[r]['V']:
                        gi = 2 * J + li
                        gj = 2 * K + lj
                        V[gi * p + gj][col] = coeff
                    
                    for (li, lj), coeff in strassen[r]['W']:
                        gi = 2 * K + li # Wait, W index is k*n+i?
                        # Task says W[k*n+i, r].
                        # So index is k * n + i.
                        # Here i is row of C, k is col of C.
                        # My local indices for C are (i', k').
                        # So global i = 2*I + i', global k = 2*K + k'.
                        # W index = global_k * n + global_i.
                        gk = 2 * K + li
                        gi = 2 * I + lj
                        W[gk * n + gi][col] = coeff
                    
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
