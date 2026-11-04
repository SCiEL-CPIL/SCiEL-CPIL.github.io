import numpy as np
import random
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 140  # safety margin

    def schoolbook_2x3x7():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_2x3x3(U, V, W, R):
        # U: 6 x R, V: 9 x R, W: 9 x R
        # Indices: U[i*3+j], V[j*3+k], W[k*2+i]
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    s = 0
                    for c in range(R):
                        s += U[i * 3 + j][c] * V[j * 3 + k][c] * W[k * 2 + i][c]
                    if s != 1:
                        return False
        return True

    def find_decomp_2x3x3(target_rank, max_time=5.0):
        start_time = time.time()
        # Try to find a decomposition for 2x3x3
        # U: 6 x R, V: 9 x R, W: 9 x R
        while time.time() - start_time < max_time:
            U = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(6)]
            V = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(9)]
            W = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(9)]
            
            if verify_2x3x3(U, V, W, target_rank):
                return U, V, W
        return None

    # Try to find low rank decompositions for 2x3x3
    # We will use 2 blocks of 2x3x3 and 1 block of 2x3x1
    # Rank(2x3x1) is 6 (schoolbook)
    
    found_2x3x3 = None
    best_rank_2x3x3 = 18 # schoolbook
    
    # Try ranks 12 to 17
    for r in range(12, 18):
        if time.time() > deadline - 20:
            break
        # Allocate more time for lower ranks? Or just fixed time.
        # Lower ranks are harder.
        result = find_decomp_2x3x3(r, max_time=2.0)
        if result is not None:
            found_2x3x3 = result
            best_rank_2x3x3 = r
            break # Found the lowest rank, stop searching for higher ranks? 
            # Actually, we want the MINIMAL rank. So if we find 12, we stop.
            # If we don't find 12, we try 13, etc.
            
    if found_2x3x3 is None:
        # Fallback to schoolbook for 2x3x3
        U_33, V_33, W_33 = None, None, None
        rank_33 = 18
        U_33 = [[0] * rank_33 for _ in range(6)]
        V_33 = [[0] * rank_33 for _ in range(9)]
        W_33 = [[0] * rank_33 for _ in range(9)]
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U_33[i * 3 + j][col] = 1
                    V_33[j * 3 + k][col] = 1
                    W_33[k * 2 + i][col] = 1
                    col += 1
        found_2x3x3 = (U_33, V_33, W_33)
        best_rank_2x3x3 = rank_33

    # Construct 2x3x7 decomposition
    # Split B into B1 (3x3), B2 (3x3), B3 (3x1)
    # Total rank = 2 * best_rank_2x3x3 + 6
    
    R_33 = best_rank_2x3x3
    R_31 = 6
    
    total_rank = 2 * R_33 + R_31
    
    # Initialize U, V, W for 2x3x7
    # U: 6 x total_rank
    # V: 21 x total_rank
    # W: 14 x total_rank
    
    U_total = [[0] * total_rank for _ in range(6)]
    V_total = [[0] * total_rank for _ in range(21)]
    W_total = [[0] * total_rank for _ in range(14)]
    
    U_33, V_33, W_33 = found_2x3x3
    
    # Block 1: k=0,1,2
    # U part: same U_33
    for c in range(R_33):
        for i in range(6):
            U_total[i][c] = U_33[i][c]
        
        # V part: rows j*3+k for k in 0..2
        # V_33 row index is j*3+k_local where k_local in 0..2
        # V_total row index is j*7+k_global where k_global in 0..2
        for j in range(3):
            for k in range(3):
                # V_33 index: j*3+k
                # V_total index: j*7+k
                V_total[j * 7 + k][c] = V_33[j * 3 + k][c]
                
        # W part: rows k*2+i for k in 0..2
        # W_33 row index is k*2+i
        # W_total row index is k*2+i (since k_global = k_local)
        for k in range(3):
            for i in range(2):
                W_total[k * 2 + i][c] = W_33[k * 2 + i][c]

    # Block 2: k=3,4,5
    offset = R_33
    for c in range(R_33):
        c_new = offset + c
        for i in range(6):
            U_total[i][c_new] = U_33[i][c]
            
        for j in range(3):
            for k in range(3):
                # k_global = k + 3
                V_total[j * 7 + (k + 3)][c_new] = V_33[j * 3 + k][c]
                
        for k in range(3):
            for i in range(2):
                # k_global = k + 3
                W_total[(k + 3) * 2 + i][c_new] = W_33[k * 2 + i][c]

    # Block 3: k=6
    # Rank 6 schoolbook for 2x3x1
    offset = 2 * R_33
    col = 0
    for i in range(2):
        for j in range(3):
            c_new = offset + col
            U_total[i * 3 + j][c_new] = 1
            # V index: j*7 + 6
            V_total[j * 7 + 6][c_new] = 1
            # W index: 6*2 + i
            W_total[6 * 2 + i][c_new] = 1
            col += 1

    return {"U": U_total, "V": V_total, "W": W_total}
