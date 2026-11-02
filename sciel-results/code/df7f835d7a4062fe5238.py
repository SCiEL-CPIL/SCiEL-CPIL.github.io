"""
Matrix multiplication 2,2,15
Find a rank-minimal exact tensor decomposition.
"""

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # Rank calculation:
    # 7 pairs of columns use Strassen (rank 7 each) -> 49
    # 1 single column uses naive (rank 4) -> 4
    # Total rank = 53
    
    rank = 53
    # U: (n*m, rank) = (4, 53)
    # V: (m*p, rank) = (30, 53)
    # W: (p*n, rank) = (30, 53)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices in U:
    # 0: A00, 1: A01, 2: A10, 3: A11
    
    # B indices in V:
    # row j (0 or 1), col k (0..14) -> index j*p + k
    
    # C indices in W:
    # col k (0..14), row i (0 or 1) -> index k*n + i
    
    def add_strassen_block(base_r, k0, k1):
        """
        Adds Strassen decomposition for columns k0, k1.
        """
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        r5 = base_r + 4
        r6 = base_r + 5
        r7 = base_r + 6
        
        e_idx = 0 * p + k0
        f_idx = 0 * p + k1
        g_idx = 1 * p + k0
        h_idx = 1 * p + k1
        
        # P1 = (A00 + A11) * (e + h)
        U[0][r1] = 1
        U[3][r1] = 1
        V[e_idx][r1] = 1
        V[h_idx][r1] = 1
        
        # P2 = (A10 + A11) * e
        U[2][r2] = 1
        U[3][r2] = 1
        V[e_idx][r2] = 1
        
        # P3 = A00 * (f - h)
        U[0][r3] = 1
        V[f_idx][r3] = 1
        V[h_idx][r3] = -1
        
        # P4 = A11 * (g - e)
        U[3][r4] = 1
        V[g_idx][r4] = 1
        V[e_idx][r4] = -1
        
        # P5 = (A00 + A01) * h
        U[0][r5] = 1
        U[1][r5] = 1
        V[h_idx][r5] = 1
        
        # P6 = (A10 - A00) * (e + f)
        U[2][r6] = 1
        U[0][r6] = -1
        V[e_idx][r6] = 1
        V[f_idx][r6] = 1
        
        # P7 = (A01 - A11) * (g + h)
        U[1][r7] = 1
        U[3][r7] = -1
        V[g_idx][r7] = 1
        V[h_idx][r7] = 1
        
        # W reconstruction
        # C0k0 (index 2*k0)
        W[2*k0][r1] = 1
        W[2*k0][r4] = 1
        W[2*k0][r5] = -1
        W[2*k0][r7] = 1
        
        # C1k0 (index 2*k0+1)
        W[2*k0+1][r2] = 1
        W[2*k0+1][r4] = 1
        
        # C0k1 (index 2*k1)
        W[2*k1][r3] = 1
        W[2*k1][r5] = 1
        
        # C1k1 (index 2*k1+1)
        W[2*k1+1][r1] = 1
        W[2*k1+1][r2] = -1
        W[2*k1+1][r3] = 1
        W[2*k1+1][r6] = 1

    # Add 7 Strassen blocks for column pairs (0,1), (2,3), ..., (12,13)
    for i in range(7):
        k0 = 2 * i
        k1 = 2 * i + 1
        base_r = 7 * i
        add_strassen_block(base_r, k0, k1)
    
    # Add naive block for column 14
    base_r = 49
    r1 = base_r
    r2 = base_r + 1
    r3 = base_r + 2
    r4 = base_r + 3
    
    e_idx = 0 * p + 14
    g_idx = 1 * p + 14
    
    # Q1 = A00 * B014 -> C014
    U[0][r1] = 1
    V[e_idx][r1] = 1
    W[2*14][r1] = 1
    
    # Q2 = A01 * B114 -> C014
    U[1][r2] = 1
    V[g_idx][r2] = 1
    W[2*14][r2] = 1
    
    # Q3 = A10 * B014 -> C114
    U[2][r3] = 1
    V[e_idx][r3] = 1
    W[2*14+1][r3] = 1
    
    # Q4 = A11 * B114 -> C114
    U[3][r4] = 1
    V[g_idx][r4] = 1
    W[2*14+1][r4] = 1
    
    return {"U": U, "V": V, "W": W}
