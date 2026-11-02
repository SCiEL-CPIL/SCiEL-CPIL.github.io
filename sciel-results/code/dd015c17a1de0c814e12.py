"""Rewritten program for Matrix multiplication 2,2,13"""

import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0  # 150s wall-clock limit
    
    n, m, p = DIMENSIONS
    
    # Rank 46 decomposition:
    # 6 blocks of 2 columns using Strassen (rank 7 each)
    # 1 block of 1 column using schoolbook (rank 4)
    # Total rank: 6 * 7 + 4 = 46
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]  # 4 x 46
    V = [[0] * rank for _ in range(m * p)]  # 26 x 46
    W = [[0] * rank for _ in range(p * n)]  # 26 x 46
    
    col = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, val in u_coeffs.items():
            U[idx][col] += val
        for idx, val in v_coeffs.items():
            V[idx][col] += val
        for idx, val in w_coeffs.items():
            W[idx][col] += val
        col += 1
        
        # Deadline check
        if time.time() > deadline - 5.0:
            return False
        return True

    # Process 6 blocks of 2 columns using Strassen
    for blk in range(6):
        k0 = blk * 2
        k1 = blk * 2 + 1
        
        # U indices: i*m+j -> 00:0, 01:1, 10:2, 11:3
        # V indices: j*p+k -> j=0: k, j=1: 13+k
        # W indices: k*n+i -> i=0: 2k, i=1: 2k+1
        
        # M1 = (a00+a11)(b00+b11)
        if not add_term(
            {0: 1, 3: 1},
            {k0: 1, 13 + k1: 1},
            {2 * k0: 1, 2 * k1 + 1: 1}
        ):
            break
            
        # M2 = (a10+a11)b00
        if not add_term(
            {2: 1, 3: 1},
            {k0: 1},
            {2 * k0 + 1: 1, 2 * k1 + 1: -1}
        ):
            break
            
        # M3 = a00(b01-b11)
        if not add_term(
            {0: 1},
            {k1: 1, 13 + k1: -1},
            {2 * k1: 1, 2 * k1 + 1: 1}
        ):
            break
            
        # M4 = a11(b10-b00)
        if not add_term(
            {3: 1},
            {13 + k0: 1, k0: -1},
            {2 * k0: 1, 2 * k0 + 1: 1}
        ):
            break
            
        # M5 = (a00+a01)b11
        if not add_term(
            {0: 1, 1: 1},
            {13 + k1: 1},
            {2 * k0: -1, 2 * k1: 1}
        ):
            break
            
        # M6 = (a10-a00)(b00+b01)
        if not add_term(
            {2: 1, 0: -1},
            {k0: 1, k1: 1},
            {2 * k1 + 1: 1}
        ):
            break
            
        # M7 = (a01-a11)(b10+b11)
        if not add_term(
            {1: 1, 3: -1},
            {13 + k0: 1, 13 + k1: 1},
            {2 * k0: 1}
        ):
            break
    
    # Process last column k=12 with schoolbook (4 terms)
    k = 12
    if not add_term({0: 1}, {k: 1}, {2 * k: 1}):
        pass
    if not add_term({1: 1}, {13 + k: 1}, {2 * k: 1}):
        pass
    if not add_term({2: 1}, {k: 1}, {2 * k + 1: 1}):
        pass
    if not add_term({3: 1}, {13 + k: 1}, {2 * k + 1: 1}):
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
