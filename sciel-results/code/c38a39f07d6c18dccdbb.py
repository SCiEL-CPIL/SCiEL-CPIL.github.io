import time
import numpy as np
import random

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    target = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target.add((i*m+j, j*p+k, k*n+i))
    
    def check(U, V, W, R):
        for (a, b, c) in target:
            s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(R))
            if abs(s - 1) > 1e-9:
                return False
        return True
    
    # Schoolbook
    R0 = n * m * p
    U = [[0]*R0 for _ in range(nm)]
    V = [[0]*R0 for _ in range(mp)]
    W = [[0]*R0 for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    best_U, best_V, best_W, best_R = U, V, W, R0
    
    # Try removing columns one at a time
    R = R0
    for col_to_remove in range(R0 - 1, -1, -1):
        if time.time() > deadline:
            break
        # Try zeroing out this column
        U_test = [row[:] for row in U]
        V_test = [row[:] for row in V]
        W_test = [row[:] for row in W]
        for idx in range(nm):
            U_test[idx][col_to_remove] = 0
        for idx in range(mp):
            V_test[idx][col_to_remove] = 0
        for idx in range(pn):
            W_test[idx][col_to_remove] = 0
        
        if check(U_test, V_test, W_test, R):
            U = U_test
            V = V_test
            W = W_test
            best_U, best_V, best_W = U, V, W
            best_R = R
            # Remove the column
            U = [row[:col_to_remove] + row[col_to_remove+1:] for row in U]
            V = [row[:col_to_remove] + row[col_to_remove+1:] for row in V]
            W = [row[:col_to_remove] + row[col_to_remove+1:] for row in W]
            R -= 1
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
