"""Rewritten program for Matrix multiplication 2,2,13"""

import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    rank = 46
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u, v, w):
        nonlocal col
        for idx, val in u.items():
            U[idx][col] += val
        for idx, val in v.items():
            V[idx][col] += val
        for idx, val in w.items():
            W[idx][col] += val
        col += 1

    # Strassen coefficients for 2x2x2
    # M1: (a00+a11)(b00+b11) -> c00 +=, c11 +=
    # M2: (a10+a11)b00 -> c10 +=, c11 -=
    # M3: a00(b01-b11) -> c01 +=, c11 +=
    # M4: a11(b10-b00) -> c00 +=, c10 +=
    # M5: (a00+a01)b11 -> c00 -=, c01 +=
    # M6: (a10-a00)(b00+b01) -> c11 +=
    # M7: (a01-a11)(b10+b11) -> c00 +=
    
    for block_idx in range(6):
        if time.time() > deadline:
            break
        k0 = block_idx * 2
        k1 = block_idx * 2 + 1
        
        # V indices: j*p + k
        # j=0: k, j=1: 13+k
        # W indices: k*n + i
        # i=0: 2k, i=1: 2k+1
        # U indices: i*m + j
        # 00:0, 01:1, 10:2, 11:3
        
        # M1
        add_term({0: 1, 3: 1}, {k0: 1, 13+k1: 1}, {2*k0: 1, 2*k1+1: 1})
        # M2
        add_term({2: 1, 3: 1}, {k0: 1}, {2*k0+1: 1, 2*k1+1: -1})
        # M3
        add_term({0: 1}, {k1: 1, 13+k1: -1}, {2*k1: 1, 2*k1+1: 1})
        # M4
        add_term({3: 1}, {13+k0: 1, k0: -1}, {2*k0: 1, 2*k0+1: 1})
        # M5
        add_term({0: 1, 1: 1}, {13+k1: 1}, {2*k0: -1, 2*k1: 1})
        # M6
        add_term({2: 1, 0: -1}, {k0: 1, k1: 1}, {2*k1+1: 1})
        # M7
        add_term({1: 1, 3: -1}, {13+k0: 1, 13+k1: 1}, {2*k0: 1})

    # Last column k=12
    if time.time() <= deadline:
        k = 12
        add_term({0: 1}, {k: 1}, {2*k: 1})
        add_term({1: 1}, {13+k: 1}, {2*k: 1})
        add_term({2: 1}, {k: 1}, {2*k+1: 1})
        add_term({3: 1}, {13+k: 1}, {2*k+1: 1})

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
