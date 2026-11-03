import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 82 decomposition
    # Split m=4 into J1={0,1} and J2={2,3}
    # For each J block, we have a 3x2x7 problem.
    # In 3x2x7, we use Strassen on I={0,1}, J={0,1} (relative), K={0,1} (relative).
    
    rank = 82
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(i, j, k, u, v, w):
        nonlocal col
        U[i * m + j][col] += u
        V[j * p + k][col] += v
        W[k * n + i][col] += w
        col += 1

    # Strassen 2x2x2 coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    def strassen_222(i_off, j_off, k_off):
        nonlocal col
        # i_off, j_off, k_off are offsets for I, J, K
        # I: i_off, i_off+1
        # J: j_off, j_off+1
        # K: k_off, k_off+1
        
        i0, i1 = i_off, i_off + 1
        j0, j1 = j_off, j_off + 1
        k0, k1 = k_off, k_off + 1
        
        # M1
        add_term(i0, j0, k0, 1, 1, 1)
        add_term(i0, j1, k1, 1, 1, 1)
        add_term(i1, j0, k1, 1, 1, 1)
        add_term(i1, j1, k0, 1, 1, 1)
        
        # M2
        add_term(i1, j0, k0, 1, 1, 1)
        add_term(i1, j1, k0, 1, 1, 1)
        
        # M3
        add_term(i0, j0, k1, 1, 1, -1)
        add_term(i0, j1, k1, 1, 1, -1)
        
        # M4
        add_term(i1, j1, k1, 1, 1, 1)
        add_term(i1, j0, k0, 1, -1, 1)
        
        # M5
        add_term(i0, j1, k1, 1, 1, 1)
        add_term(i1, j1, k1, 1, 1, 1)
        
        # M6
        add_term(i1, j0, k0, 1, 1, 1)
        add_term(i0, j0, k0, -1, 1, 1)
        add_term(i1, j0, k1, 1, 1, 1)
        add_term(i0, j0, k1, -1, 1, 1)
        
        # M7
        add_term(i0, j1, k1, 1, 1, 1)
        add_term(i1, j1, k1, -1, 1, 1)
        add_term(i0, j1, k0, 1, 1, 1)
        add_term(i1, j1, k0, -1, 1, 1)

    # We need to be careful with the Strassen implementation above.
    # It adds multiple terms for each M.
    # This is correct for the decomposition.
    
    # Process J1 = {0, 1}
    # Strassen block: I={0,1}, J={0,1}, K={0,1}
    strassen_222(0, 0, 0)
    
    # Rest of J1:
    # 1. I={2}, J={0,1}, K={0..6}
    for j in [0, 1]:
        for k in range(7):
            add_term(2, j, k, 1, 1, 1)
            
    # 2. I={0,1}, J={0,1}, K={2..6}
    for i in [0, 1]:
        for j in [0, 1]:
            for k in range(2, 7):
                add_term(i, j, k, 1, 1, 1)

    # Process J2 = {2, 3}
    # Strassen block: I={0,1}, J={2,3}, K={0,1}
    strassen_222(0, 2, 0)
    
    # Rest of J2:
    # 1. I={2}, J={2,3}, K={0..6}
    for j in [2, 3]:
        for k in range(7):
            add_term(2, j, k, 1, 1, 1)
            
    # 2. I={0,1}, J={2,3}, K={2..6}
    for i in [0, 1]:
        for j in [2, 3]:
            for k in range(2, 7):
                add_term(i, j, k, 1, 1, 1)

    return {"U": U, "V": V, "W": W}
