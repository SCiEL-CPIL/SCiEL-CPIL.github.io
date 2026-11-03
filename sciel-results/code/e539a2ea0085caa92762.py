"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # Precompute target tensor non-zeros
    # T[a, b, c] = 1 if a=im+j, b=jp+k, c=kn+i match indices
    # a in [0, nm), b in [0, mp), c in [0, pn)
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[(a, b, c)] = 1

    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def check_decomposition(U, V, W):
        r = len(U[0])
        # Verify all non-zero entries of target
        for (a, b, c), val in target.items():
            s = 0
            for r_idx in range(r):
                s += U[a][r_idx] * V[b][r_idx] * W[c][r_idx]
            if s != val:
                return False
        # Verify all zero entries (sample some or check all if fast)
        # Total entries nm*mp*pn = 12*28*21 = 7056.
        # Non-zeros are 3*4*7 = 84.
        # Zeros are 7056 - 84 = 6972.
        # Checking all is feasible.
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) not in target:
                        s = 0
                        for r_idx in range(r):
                            s += U[a][r_idx] * V[b][r_idx] * W[c][r_idx]
                        if s != 0:
                            return False
        return True

    # Try random search for lower rank
    best_result = None
    # Try ranks from 40 to 83
    for rank in range(40, 84):
        if time.time() > deadline:
            break
            
        # Try a few random initializations for this rank
        for attempt in range(10):
            if time.time() > deadline:
                break
                
            # Random matrices with entries in {-1, 0, 1}
            # Sparse initialization might be better
            U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p * n)]
            
            if check_decomposition(U, V, W):
                return {"U": U, "V": V, "W": W}
                
            # Local search: flip random entries
            # This is very slow, so we skip extensive local search in this simple version
            # unless we have lots of time.
            
    # Fallback to schoolbook
    return schoolbook(n, m, p)

# EVOLVE-BLOCK-END
