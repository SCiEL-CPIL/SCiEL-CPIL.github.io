import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def _schoolbook(n=2, m=2, p=12):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _check(U, V, W, n, m, p, tol=1e-8):
    R = len(U[0]) if len(U)>0 else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                # Correct check for matrix multiplication: target is 1.0
                if abs(s - 1.0) > tol:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_t = time.time()
    deadline = start_t + 105
    n, m, p = DIMENSIONS  # 2, 2, 12
    
    # Strassen Rank 42 Decomposition
    rank = 42
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 42
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0.0] * rank for _ in range(p * n)]  # 24 x 42

    # Strassen coefficients for 2x2 multiplication
    # s=0..6
    # U[i,j], V[j,k], W[k,i]
    # Note: W index is k*n + i
    
    # Predefined tuples: (U_indices, U_vals, V_indices, V_vals, W_indices, W_vals)
    # Indices are local (0,1)
    
    # s=0 (P1)
    U_idx = [(0,0), (1,1)]
    U_val = [1.0, 1.0]
    V_idx = [(0,0), (1,1)]
    V_val = [1.0, 1.0]
    W_idx = [(0,0), (1,1)] # k=0,i=0 and k=1,i=1
    W_val = [1.0, 1.0]
    
    # We will define a list of dictionaries for each s
    strassen_terms = [
        {
            "U": { (0,0): 1.0, (1,1): 1.0 },
            "V": { (0,0): 1.0, (1,1): 1.0 },
            "W": { (0,0): 1.0, (1,1): 1.0 }
        },
        {
            "U": { (1,0): 1.0, (1,1): 1.0 },
            "V": { (0,0): 1.0 },
            "W": { (0,1): 1.0, (1,1): 1.0 }
        },
        {
            "U": { (0,0): 1.0 },
            "V": { (0,1): 1.0, (1,1): -1.0 },
            "W": { (1,0): 1.0, (1,1): -1.0 }
        },
        {
            "U": { (1,1): 1.0 },
            "V": { (1,0): 1.0, (0,0): -1.0 },
            "W": { (0,0): -1.0, (0,1): 1.0 }
        },
        {
            "U": { (0,0): 1.0, (1,0): 1.0 },
            "V": { (1,1): 1.0 },
            "W": { (1,0): 1.0, (1,1): -1.0 }
        },
        {
            "U": { (1,1): 1.0, (1,0): -1.0 },
            "V": { (0,0): 1.0, (0,1): 1.0 },
            "W": { (0,0): 1.0 }
        },
        {
            "U": { (0,1): 1.0, (1,1): -1.0 },
            "V": { (1,0): 1.0, (1,1): 1.0 },
            "W": { (0,0): -1.0 }
        }
    ]

    for l in range(6):
        for s in range(7):
            r = l * 7 + s
            term = strassen_terms[s]
            
            for (i, j), val in term["U"].items():
                U[i*m+j][r] += val
                
            for (j, k), val in term["V"].items():
                k_global = 2*l + k
                V[j*p+k_global][r] += val
                
            for (k, i), val in term["W"].items():
                k_global = 2*l + k
                W[k_global*n+i][r] += val

    # Verify
    if _check(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback
    fb = _schoolbook(n, m, p)
    return fb

# EVOLVE-BLOCK-END
