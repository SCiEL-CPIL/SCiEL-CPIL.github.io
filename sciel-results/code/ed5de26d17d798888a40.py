"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # U: n*m = 8 rows
    # V: m*p = 16 rows
    # W: p*n = 8 rows
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 4 blocks of 2x2x2
    # Blocks defined by (i_base, j_base, k_base)
    # i_base is always 0 for A (since n=2, we use both rows)
    # j_base is 0 or 2 (splitting A's columns and B's rows)
    # k_base is 0 or 2 (splitting B's columns and C's columns)
    
    blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 2, 0),
        (0, 2, 2)
    ]
    
    col = 0
    for ib, jb, kb in blocks:
        # Local A indices
        a00 = ib * m + jb
        a01 = ib * m + jb + 1
        a10 = (ib + 1) * m + jb
        a11 = (ib + 1) * m + jb + 1
        
        # Local B indices
        b00 = jb * p + kb
        b01 = jb * p + kb + 1
        b10 = (jb + 1) * p + kb
        b11 = (jb + 1) * p + kb + 1
        
        # Local C indices (W index is k*n + i)
        c00 = kb * n + ib
        c01 = kb * n + ib + 1
        c10 = (kb + 1) * n + ib
        c11 = (kb + 1) * n + ib + 1
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        U[a00][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M2: (A10+A11)B00 -> C10
        U[a10][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # M3: A00(B10-B11) -> C01
        U[a00][col] += 1
        V[b10][col] += 1
        V[b11][col] -= 1
        W[c01][col] += 1
        col += 1
        
        # M4: A11(B01-B00) -> C00, C10
        U[a11][col] += 1
        V[b01][col] += 1
        V[b00][col] -= 1
        W[c00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # M5: (A00+A10)B11 -> C00, C01
        U[a00][col] += 1
        U[a10][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1
        W[c01][col] += 1
        col += 1
        
        # M6: (A11-A00)(B00+B10) -> C11
        U[a11][col] += 1
        U[a00][col] -= 1
        V[b00][col] += 1
        V[b10][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7: (A01-A11)(B11+B01) -> C00, C11
        U[a01][col] += 1
        U[a11][col] -= 1
        V[b11][col] += 1
        V[b01][col] += 1
        W[c00][col] += 1
        W[c11][col] -= 1
        col += 1
        
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
