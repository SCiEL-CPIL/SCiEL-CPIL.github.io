"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    n, m, p = DIMENSIONS
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors definitions
    # M1: (A00+A11)(B00+B11) -> C00+1, C11+1
    # M2: (A10+A11)B00      -> C10+1, C11-1
    # M3: A00(B10-B11)       -> C01+1, C11+1
    # M4: A11(B01-B00)       -> C00+1, C10+1
    # M5: (A00+A01)B11       -> C00-1, C01+1
    # M6: (A10-A00)(B00+B01) -> C11+1
    # M7: (A01-A11)(B10+B11) -> C00+1
    
    # U indices: 0:A00, 1:A01, 2:A10, 3:A11
    # V indices for block k: 2k:B00, 2k+1:B01, 14+2k:B10, 14+2k+1:B11
    # W indices for block k: 4k:C00, 4k+1:C10, 4k+2:C01, 4k+3:C11
    
    col = 0
    for k in range(7):
        if time.time() > deadline:
            break
            
        # V base indices
        v_b00 = 2 * k
        v_b01 = 2 * k + 1
        v_b10 = 14 + 2 * k
        v_b11 = 14 + 2 * k + 1
        
        # W base indices
        w_c00 = 4 * k
        w_c10 = 4 * k + 1
        w_c01 = 4 * k + 2
        w_c11 = 4 * k + 3
        
        # M1
        U[0][col] = 1
        U[3][col] = 1
        V[v_b00][col] = 1
        V[v_b11][col] = 1
        W[w_c00][col] = 1
        W[w_c11][col] = 1
        col += 1
        
        # M2
        U[2][col] = 1
        U[3][col] = 1
        V[v_b00][col] = 1
        W[w_c10][col] = 1
        W[w_c11][col] = -1
        col += 1
        
        # M3
        U[0][col] = 1
        V[v_b10][col] = 1
        V[v_b11][col] = -1
        W[w_c01][col] = 1
        W[w_c11][col] = 1
        col += 1
        
        # M4
        U[3][col] = 1
        V[v_b01][col] = 1
        V[v_b00][col] = -1
        W[w_c00][col] = 1
        W[w_c10][col] = 1
        col += 1
        
        # M5
        U[0][col] = 1
        U[1][col] = 1
        V[v_b11][col] = 1
        W[w_c00][col] = -1
        W[w_c01][col] = 1
        col += 1
        
        # M6
        U[2][col] = 1
        U[0][col] = -1
        V[v_b00][col] = 1
        V[v_b01][col] = 1
        W[w_c11][col] = 1
        col += 1
        
        # M7
        U[1][col] = 1
        U[3][col] = -1
        V[v_b10][col] = 1
        V[v_b11][col] = 1
        W[w_c00][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
