"""Program for matrix multiplication 2,2,10 using block-diagonal Strassen decomposition."""
import time
_START_TIME = time.time()
DEADLINE = 120.0

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 2x2 * 2x10 multiplication into five independent 2x2 * 2x2 blocks.
    Each 2x2 * 2x2 block uses the rank-7 Strassen decomposition.
    Total rank = 5 * 7 = 35.
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 5 blocks
    
    # Strassen rank-7 decomposition for 2x2 matrices
    # Terms: (U_entries, V_entries, W_entries)
    # U: (i,j), V: (j,k), W: (k,i)
    
    strassen_terms = [
        # M1 = (a+d)(e+h)
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        # M2 = (c+d)e
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,1): 1, (1,1): -1}
        ),
        # M3 = a(f-h)
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1}
        ),
        # M4 = d(g-e)
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1}
        ),
        # M5 = (a+b)h
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (1,0): 1}
        ),
        # M6 = (c-a)(e+f)
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        # M7 = (b-d)(g+h)
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        ),
    ]
    
    rank = num_blocks * 7  # 35
    
    if time.time() - _START_TIME > DEADLINE - 1.0:
        raise TimeoutError("Exceeded deadline")
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(num_blocks):
        for t in range(7):
            col = r * 7 + t
            u_entries, v_entries, w_entries = strassen_terms[t]
            
            for (i, j), val in u_entries.items():
                U[i * m + j][col] = val
            
            for (j, k), val in v_entries.items():
                V[j * p + (2 * r + k)][col] = val
            
            for (k, i), val in w_entries.items():
                W[(2 * r + k) * n + i][col] = val
    
    if time.time() - _START_TIME > DEADLINE - 0.5:
        raise TimeoutError("Exceeded deadline")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
