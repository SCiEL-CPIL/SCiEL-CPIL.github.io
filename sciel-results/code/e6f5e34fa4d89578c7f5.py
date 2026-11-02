"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = _start_time + 145.0

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # Strategy: Split 15 columns into 5 groups of 3
    # Each group of 3 uses a rank-10 decomposition for 2x2 x 2x3
    # Total rank = 5 * 10 = 50
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # 4 x 50
    V = [[0] * rank for _ in range(m * p)]  # 30 x 50
    W = [[0] * rank for _ in range(p * n)]  # 30 x 50
    
    # A indices in U:
    # 0: A00 (a), 1: A01 (b), 2: A10 (c), 3: A11 (d)
    
    # B indices in V:
    # j*p + k where j is row (0 or 1) and k is column (0..14)
    
    # C indices in W:
    # k*n + i where k is column (0..14) and i is row (0 or 1)
    
    def add_rank10_block_2x2x3(base_r, k0, k1, k2):
        """Add rank-10 decomposition for 2x2 x 2x3 matrix multiplication.
        
        B submatrix columns k0, k1, k2:
        e = B0k0, f = B0k1, g2 = B0k2
        h = B1k0, i = B1k1, j2 = B1k2
        
        First 7 terms: Strassen for columns (k0, k1)
        Last 3 terms: schoolbook for column k2
        """
        # --- Strassen terms for columns k0, k1 (ranks 0-6) ---
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        r5 = base_r + 4
        r6 = base_r + 5
        r7 = base_r + 6
        
        e_idx = 0 * p + k0
        f_idx = 0 * p + k1
        h_idx = 1 * p + k0
        i_idx = 1 * p + k1
        
        # P1 = (a+d)(e+h)
        U[0][r1] = 1
        U[3][r1] = 1
        V[e_idx][r1] = 1
        V[h_idx][r1] = 1
        
        # P2 = (c+d)e
        U[2][r2] = 1
        U[3][r2] = 1
        V[e_idx][r2] = 1
        
        # P3 = a(f-h)
        U[0][r3] = 1
        V[f_idx][r3] = 1
        V[h_idx][r3] = -1
        
        # P4 = d(h-e)
        U[3][r4] = 1
        V[h_idx][r4] = 1
        V[e_idx][r4] = -1
        
        # P5 = (a+b)i
        U[0][r5] = 1
        U[1][r5] = 1
        V[i_idx][r5] = 1
        
        # P6 = (c-a)(e+f)
        U[2][r6] = 1
        U[0][r6] = -1
        V[e_idx][r6] = 1
        V[f_idx][r6] = 1
        
        # P7 = (b-d)(h+i)
        U[1][r7] = 1
        U[3][r7] = -1
        V[h_idx][r7] = 1
        V[i_idx][r7] = 1
        
        # W reconstruction for Strassen part
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
        
        # --- Schoolbook terms for column k2 (ranks 7-9) ---
        r8 = base_r + 7
        r9 = base_r + 8
        r10 = base_r + 9
        
        g2_idx = 0 * p + k2
        j2_idx = 1 * p + k2
        
        # Q1 = a * g2 -> C0k2
        U[0][r8] = 1
        V[g2_idx][r8] = 1
        W[2*k2][r8] = 1
        
        # Q2 = b * j2 -> C0k2
        U[1][r9] = 1
        V[j2_idx][r9] = 1
        W[2*k2][r9] = 1
        
        # Q3 = c * g2 -> C1k2
        U[2][r10] = 1
        V[g2_idx][r10] = 1
        W[2*k2+1][r10] = 1
        
        # Wait, we need d*j2 -> C1k2 too. That's 4 terms, not 3.
        # Let me reconsider: rank-10 for 2x2x3 means we need to save 2 terms
        # compared to schoolbook (12 terms). So we use 7+4=11? No...
        # Actually rank of M_{2,2,3} is known to be 10.
        # Let me use a proper rank-10 scheme.
    
    # The above has an issue - let me use a verified rank-10 construction.
    # For 2x2 x 2x3, we can use:
    # - 7 Strassen terms for the first 2 columns
    # - 3 terms for the 3rd column (using the fact that C0k2 = a*g2 + b*j2 
    #   and C1k2 = c*g2 + d*j2, but we can share some structure)
    # Actually the standard result is rank 10. Let me just use 7+4=11 
    # which is still better than 12, giving total rank 5*11=55. 
    # That's worse than 53. Let me stick with the 7-pair + 1-single = 53 approach
    # but verify correctness carefully.
    
    # Reset and use the proven 7-pair + 1-single approach
    rank = 53
    U = [[0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * rank for _ in range(p * n)]  # 30 x 53
    
    def add_strassen_block(base_r, k0, k1):
        """Add Strassen decomposition for column pair (k0, k1)."""
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
        
        # P1 = (a+d)(e+h)
        U[0][r1] = 1
        U[3][r1] = 1
        V[e_idx][r1] = 1
        V[h_idx][r1] = 1
        
        # P2 = (c+d)e
        U[2][r2] = 1
        U[3][r2] = 1
        V[e_idx][r2] = 1
        
        # P3 = a(f-h)
        U[0][r3] = 1
        V[f_idx][r3] = 1
        V[h_idx][r3] = -1
        
        # P4 = d(g-e)
        U[3][r4] = 1
        V[g_idx][r4] = 1
        V[e_idx][r4] = -1
        
        # P5 = (a+b)h
        U[0][r5] = 1
        U[1][r5] = 1
        V[h_idx][r5] = 1
        
        # P6 = (c-a)(e+f)
        U[2][r6] = 1
        U[0][r6] = -1
        V[e_idx][r6] = 1
        V[f_idx][r6] = 1
        
        # P7 = (b-d)(g+h)
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
    
    # Add trivial block for column 14
    base_r = 49
    r1 = base_r
    r2 = base_r + 1
    r3 = base_r + 2
    r4 = base_r + 3
    
    e_idx = 0 * p + 14
    g_idx = 1 * p + 14
    
    # Q1 = a * e
    U[0][r1] = 1
    V[e_idx][r1] = 1
    W[2*14][r1] = 1  # C014
    
    # Q2 = b * g
    U[1][r2] = 1
    V[g_idx][r2] = 1
    W[2*14][r2] = 1  # C014
    
    # Q3 = c * e
    U[2][r3] = 1
    V[e_idx][r3] = 1
    W[2*14+1][r3] = 1  # C114
    
    # Q4 = d * g
    U[3][r4] = 1
    V[g_idx][r4] = 1
    W[2*14+1][r4] = 1  # C114
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
