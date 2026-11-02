"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 120.0
    
    n, m, p = DIMENSIONS
    # rank = 7 * (p // 2) = 7 * 6 = 42
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 linear forms for A (2x2)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U_forms = [
        [1, 0, 0, 1],  # S0: a00 + a11
        [0, 0, 1, 1],  # S1: a10 + a11
        [1, 0, 0, 0],  # S2: a00
        [0, 0, 0, 1],  # S3: a11
        [1, 1, 0, 0],  # S4: a00 + a01
        [-1, 0, 1, 0], # S5: a10 - a00
        [0, 1, 0, -1]  # S6: a01 - a11
    ]
    
    # Precompute V and W contributions for each Strassen form t
    # V_t is a list of (row, val) pairs for the 2x2 block
    # W_t is a list of (row, val) pairs for the 2x2 block
    
    # Standard Strassen V forms (for B):
    # M0 = (a00+a11)(b00+b11)
    # M1 = (a10+a11)(b00)
    # M2 = (a00)(b01-b11)
    # M3 = (a11)(b10-b00)
    # M4 = (a00+a01)(b11)
    # M5 = (a10-a00)(b10+b11)
    # M6 = (a01-a11)(b00+b10)
    
    # Note: The current code uses a specific variant. 
    # I will keep the exact logic of the verified rank-42 solution.
    
    for u in range(6):
        if time.time() - start_time > timeout - 5:
            break
        for t in range(7):
            c = u * 7 + t
            
            # Fill U
            for i in range(4):
                U[i][c] = U_forms[t][i]
                
            # Fill V
            if t == 0:
                V[0 * p + 2*u][c] += 1
                V[1 * p + 2*u + 1][c] += 1
            elif t == 1:
                V[0 * p + 2*u][c] += 1
            elif t == 2:
                V[0 * p + 2*u + 1][c] += 1
                V[1 * p + 2*u + 1][c] -= 1
            elif t == 3:
                V[1 * p + 2*u][c] += 1
                V[0 * p + 2*u][c] -= 1
            elif t == 4:
                V[1 * p + 2*u + 1][c] += 1
            elif t == 5:
                V[0 * p + 2*u][c] += 1
                V[0 * p + 2*u + 1][c] += 1
            elif t == 6:
                V[1 * p + 2*u][c] += 1
                V[1 * p + 2*u + 1][c] += 1
                
            # Fill W
            if t == 0:
                W[(2*u) * n + 0][c] += 1
                W[(2*u+1) * n + 1][c] += 1
            elif t == 1:
                W[(2*u) * n + 1][c] += 1
                W[(2*u+1) * n + 1][c] -= 1
            elif t == 2:
                W[(2*u+1) * n + 0][c] += 1
                W[(2*u+1) * n + 1][c] += 1
            elif t == 3:
                W[(2*u) * n + 0][c] += 1
                W[(2*u) * n + 1][c] += 1
            elif t == 4:
                W[(2*u) * n + 0][c] -= 1
                W[(2*u+1) * n + 0][c] += 1
            elif t == 5:
                W[(2*u+1) * n + 1][c] += 1
            elif t == 6:
                W[(2*u) * n + 0][c] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
