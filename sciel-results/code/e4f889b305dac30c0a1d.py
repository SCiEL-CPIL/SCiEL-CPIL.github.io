import time
import numpy as np

DIMENSIONS = [2, 2, 10]
_DEADLINE = time.time() + 140

def construct():
    """
    Returns a rank-35 exact tensor decomposition for 2x2 * 2x10 matrix multiplication.
    Uses 5 blocks of Strassen's algorithm (rank 7 each).
    """
    if time.time() > _DEADLINE:
        return _construct_rank35()
    
    result = _construct_rank35()
    
    # Verify correctness
    if time.time() < _DEADLINE - 5:
        if not _verify_decomposition(result):
            return _construct_schoolbook()
            
    return result

def _construct_rank35():
    n, m, p = DIMENSIONS
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen components for 2x2 * 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    # U coeffs for A entries (a, b, c, d)
    strassen_A = [
        [1, 0, 0, 1],  # P1
        [0, 0, 1, 1],  # P2
        [1, 0, 0, 0],  # P3
        [0, 0, 0, 1],  # P4
        [1, 1, 0, 0],  # P5
        [-1, 0, 1, 0], # P6
        [0, 1, 0, -1]  # P7
    ]
    
    # V coeffs for B entries (e, f, g, h)
    strassen_B = [
        [1, 0, 0, 1],  # P1
        [1, 0, 0, 0],  # P2
        [0, 1, 0, -1], # P3
        [-1, 0, 1, 0], # P4
        [0, 0, 0, 1],  # P5
        [1, 1, 0, 0],  # P6
        [0, 0, 1, 1]   # P7
    ]
    
    # W coeffs for C entries (C00, C01, C10, C11)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    w_contrib = [
        [1, 0, 0, 1],  # P1
        [0, 0, 1, -1], # P2
        [0, 1, 0, 1],  # P3
        [1, 0, 1, 0],  # P4
        [-1, 1, 0, 0], # P5
        [0, 0, 0, 1],  # P6
        [1, 0, 0, 0]   # P7
    ]
    
    for b in range(5):
        for r in range(7):
            col = b * 7 + r
            
            # Fill U
            U[0][col] = strassen_A[r][0]
            U[1][col] = strassen_A[r][1]
            U[2][col] = strassen_A[r][2]
            U[3][col] = strassen_A[r][3]
            
            # Fill V
            k0 = 2 * b
            k1 = 2 * b + 1
            V[0 * p + k0][col] = strassen_B[r][0]
            V[0 * p + k1][col] = strassen_B[r][1]
            V[1 * p + k0][col] = strassen_B[r][2]
            V[1 * p + k1][col] = strassen_B[r][3]
            
            # Fill W
            W[(2 * b) * 2 + 0][col] = w_contrib[r][0]
            W[(2 * b + 1) * 2 + 0][col] = w_contrib[r][1]
            W[(2 * b) * 2 + 1][col] = w_contrib[r][2]
            W[(2 * b + 1) * 2 + 1][col] = w_contrib[r][3]
            
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(decomp, num_tests=5, seed=42):
    import random
    random.seed(seed)
    
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    for _ in range(num_tests):
        A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        
        # Compute C = A * B
        C = [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]
        
        # Compute using decomposition
        C_decomp = [[0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                val = 0
                for r in range(rank):
                    w_val = W[k*n+i][r]
                    if w_val == 0:
                        continue
                    u_dot = 0
                    for j1 in range(m):
                        u_dot += U[i*m+j1][r] * A[i][j1]
                    v_dot = 0
                    for j2 in range(m):
                        v_dot += V[j2*p+k][r] * B[j2][k]
                    val += w_val * u_dot * v_dot
                C_decomp[i][k] = val
        
        for i in range(n):
            for k in range(p):
                if C_decomp[i][k] != C[i][k]:
                    return False
    
    return True

def _construct_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = (i*m + j)*p + k
                U[i*m+j][r] = 1
                V[j*p+k][r] = 1
                W[k*n+i][r] = 1
    
    return {"U": U, "V": V, "W": W}
