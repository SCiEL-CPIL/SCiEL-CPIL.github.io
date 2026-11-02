import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Total rank: 12 (bottom row) + 8 (top right) + 12 (top left 2x2x4) = 32
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a standard multiplication term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1
    
    # 1. Bottom row: i=2, j=0..2, k=0..3 (rank 12)
    for j in range(3):
        for k in range(4):
            add_term(2 * m + j, j * p + k, k * n + 2)
    
    # 2. Top right: i=0..1, j=2, k=0..3 (rank 8)
    for i in range(2):
        for k in range(4):
            add_term(i * m + 2, 2 * p + k, k * n + i)
    
    # 3. Top left 2x2x4: i=0..1, j=0..1, k=0..3 (rank 12)
    # Use optimized decomposition
    # A indices: (0,0)=0, (0,1)=1, (1,0)=3, (1,1)=4
    # B indices: (0,0)=0, (0,1)=1, (0,2)=2, (0,3)=3, (1,0)=4, (1,1)=5, (1,2)=6, (1,3)=7
    # C indices: k*n+i -> (0,0)=0, (1,0)=3, (2,0)=6, (3,0)=9, (0,1)=1, (1,1)=4, (2,1)=7, (3,1)=10
    
    # Term 1: (a00+a11) * (b00+b10) -> contributes to c00, c11
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # Term 2: (a10+a11) * b00 -> contributes to c10, c11
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    # Term 3: a00 * (b01-b11) -> contributes to c01, c11
    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # Term 4: a11 * (b10-b00) -> contributes to c00, c10
    U[4][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # Term 5: (a00+a01) * b11 -> contributes to c00, c01
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    # Term 6: (a10-a00) * (b00+b01) -> contributes to c11
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # Term 7: (a01-a11) * (b10+b11) -> contributes to c00
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1
    
    # Now for k=2,3, use similar pattern
    # Term 8: (a00+a11) * (b02+b12) -> contributes to c02, c12
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[6][col] = 1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    # Term 9: (a10+a11) * b02 -> contributes to c12, c13
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[7][col] = 1; W[7][col] = -1  # Wait, this is wrong
    # Let me redo this more carefully
    
    # Actually, I realize my 2x2x4 decomposition is getting complex. Let me use a simpler approach.
    # Use two separate 2x2x2 Strassen blocks for k=0,1 and k=2,3
    # But that gives rank 14, not 12.
    
    # Let me reset and use the original approach with rank 34, but verify it's correct.
    # Actually, let me just return the original program since my optimization attempt is getting messy.
    
    # Reset
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # Term 1: M1
        U[i_off * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1
        # Term 2: M2
        U[(i_off+1) * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = -1
        col += 1
        # Term 3: M3
        U[i_off * m + j_off][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = -1
        W[(k_off+1) * n + i_off][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1
        # Term 4: M4
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + k_off][col] = 1
        V[j_off * p + k_off][col] = -1
        W[k_off * n + i_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1
        col += 1
        # Term 5: M5
        U[i_off * m + j_off][col] = 1
        U[i_off * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = -1
        W[(k_off+1) * n + i_off][col] = 1
        col += 1
        # Term 6: M6
        U[(i_off+1) * m + j_off][col] = 1
        U[i_off * m + j_off][col] = -1
        V[j_off * p + k_off][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1
        # Term 7: M7
        U[i_off * m + (j_off+1)][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = -1
        V[(j_off+1) * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = 1
        col += 1
    
    # 1. Bottom row
    for j in range(3):
        for k in range(4):
            add_term(2 * m + j, j * p + k, k * n + 2)
    
    # 2. Top right
    for i in range(2):
        for k in range(4):
            add_term(i * m + 2, 2 * p + k, k * n + i)
    
    # 3. Top left k=0,1
    add_strassen(0, 0, 0)
    
    # 4. Top left k=2,3
    add_strassen(0, 0, 2)
    
    return {"U": U, "V": V, "W": W}
