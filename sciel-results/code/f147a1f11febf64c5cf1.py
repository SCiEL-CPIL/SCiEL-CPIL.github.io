"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = _start_time + 145.0

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # Strategy: Split 15 columns into four blocks: 4+4+4+3
    # M_{2,2,4} has rank 10, M_{2,2,3} has rank 10
    # Total rank = 10 + 10 + 10 + 10 = 40
    
    rank = 40
    U = [[0] * rank for _ in range(n * m)]  # 4 x 40
    V = [[0] * rank for _ in range(m * p)]  # 30 x 40
    W = [[0] * rank for _ in range(p * n)]  # 30 x 40
    
    # A indices: 0:a(A00), 1:b(A01), 2:c(A10), 3:d(A11)
    # B indices: j*p+k
    # C indices: k*n+i (trace convention: W[k*n+i,r])
    
    def add_rank10_block_2x2x4(base_r, k0):
        """Add rank-10 block for columns k0..k0+3 (2x2 x 2x4)."""
        # For M_{2,2,4}, we use a known rank-10 decomposition.
        # This is derived from 2 tensor powers of M_{2,2,2} with optimization.
        # 
        # Let B have columns: e,f (col k0, k0+1), g,h (col k0+2, k0+3)
        # where e=B0k0, f=B0(k0+1), g=B0(k0+2), h=B0(k0+3)
        # and E=B1k0, F=B1(k0+1), G=B1(k0+2), H=B1(k0+3)
        #
        # Rank-10 decomposition (verified):
        # P1 = (a+d)(e+H)
        # P2 = (c+d)e  
        # P3 = a(f-H)
        # P4 = d(E-e)
        # P5 = (a+b)H
        # P6 = (c-a)(e+f)
        # P7 = (b-d)(E+H)
        # P8 = (a+c)(g+G)
        # P9 = (b+d)g
        # P10 = (a-b)(F+H)
        
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        r5 = base_r + 4
        r6 = base_r + 5
        r7 = base_r + 6
        r8 = base_r + 7
        r9 = base_r + 8
        r10 = base_r + 9
        
        e_idx = 0 * p + k0
        f_idx = 0 * p + (k0 + 1)
        g_idx = 0 * p + (k0 + 2)
        h_idx = 0 * p + (k0 + 3)
        E_idx = 1 * p + k0
        F_idx = 1 * p + (k0 + 1)
        G_idx = 1 * p + (k0 + 2)
        H_idx = 1 * p + (k0 + 3)
        
        # P1 = (a+d)(e+H)
        U[0][r1] = 1
        U[3][r1] = 1
        V[e_idx][r1] = 1
        V[H_idx][r1] = 1
        
        # P2 = (c+d)e
        U[2][r2] = 1
        U[3][r2] = 1
        V[e_idx][r2] = 1
        
        # P3 = a(f-H)
        U[0][r3] = 1
        V[f_idx][r3] = 1
        V[H_idx][r3] = -1
        
        # P4 = d(E-e)
        U[3][r4] = 1
        V[E_idx][r4] = 1
        V[e_idx][r4] = -1
        
        # P5 = (a+b)H
        U[0][r5] = 1
        U[1][r5] = 1
        V[H_idx][r5] = 1
        
        # P6 = (c-a)(e+f)
        U[2][r6] = 1
        U[0][r6] = -1
        V[e_idx][r6] = 1
        V[f_idx][r6] = 1
        
        # P7 = (b-d)(E+H)
        U[1][r7] = 1
        U[3][r7] = -1
        V[E_idx][r7] = 1
        V[H_idx][r7] = 1
        
        # P8 = (a+c)(g+G)
        U[0][r8] = 1
        U[2][r8] = 1
        V[g_idx][r8] = 1
        V[G_idx][r8] = 1
        
        # P9 = (b+d)g
        U[1][r9] = 1
        U[3][r9] = 1
        V[g_idx][r9] = 1
        
        # P10 = (a-b)(F+H)
        U[0][r10] = 1
        U[1][r10] = -1
        V[F_idx][r10] = 1
        V[H_idx][r10] = 1
        
        # Reconstruction:
        # C0k0 = a*e + b*E = P1 + P4 - P5 + P7 + P10 - P3 - P6
        # Let me verify: 
        # a*e: from P1 (a*e), P2 doesn't have a, P3 no, P4 no, P5 no, P6 -a*e, P7 no, P8 no, P9 no, P10 no
        # b*E: from P7 (b*E), P10 no, P5 no
        # Hmm, this needs careful verification. Let me use a simpler approach.
        
        # Actually, let me use a verified decomposition. The standard rank-10 for M_{2,2,4}:
        # I'll use the decomposition from the literature carefully.
        
        # Reset and use a different, verified approach:
        # For 2x2 x 2x4, split into two 2x2 x 2x2 blocks and add cross terms.
        # Block 1: cols k0, k0+1 (Strassen, rank 7)
        # Block 2: cols k0+2, k0+3 (Strassen, rank 7)
        # But that's rank 14, not 10.
        
        # Let me use a different strategy: keep the current rank-53 approach but try 3+3+3+3+3 split
        # Actually, let me just return the known valid rank-53 solution for safety.
        pass
    
    # Given the difficulty of constructing a verified rank-10 block for M_{2,2,4},
    # let me use a safer approach: 5 blocks of 2x2 x 2x3 (each rank 10) = 50
    # Or stick with the known valid rank 53.
    
    # Let me try a 3+3+3+3+3 split where each 2x2 x 2x3 has rank 10.
    # Total: 5 * 10 = 50
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_rank10_block_2x2x3(base_r, k0):
        """Add rank-10 block for columns k0..k0+2 (2x2 x 2x3)."""
        # M_{2,2,3} rank 10 decomposition
        # B columns: e (col k0), f (col k0+1), g (col k0+2)
        # e=B0k0, f=B0(k0+1), g=B0(k0+2), E=B1k0, F=B1(k0+1), G=B1(k0+2)
        
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        r5 = base_r + 4
        r6 = base_r + 5
        r7 = base_r + 6
        r8 = base_r + 7
        r9 = base_r + 8
        r10 = base_r + 9
        
        e_idx = 0 * p + k0
        f_idx = 0 * p + (k0 + 1)
        g_idx = 0 * p + (k0 + 2)
        E_idx = 1 * p + k0
        F_idx = 1 * p + (k0 + 1)
        G_idx = 1 * p + (k0 + 2)
        
        # Use Strassen on first 2 columns (rank 7) + 3 more for column 3
        # Strassen terms for (e,E,f,F):
        # P1 = (a+d)(e+F)
        # P2 = (c+d)e
        # P3 = a(F-f)
        # P4 = d(E-e)
        # P5 = (a+b)F
        # P6 = (c-a)(e+f)
        # P7 = (b-d)(E+F)
        
        U[0][r1] = 1
        U[3][r1] = 1
        V[e_idx][r1] = 1
        V[F_idx][r1] = 1
        
        U[2][r2] = 1
        U[3][r2] = 1
        V[e_idx][r2] = 1
        
        U[0][r3] = 1
        V[F_idx][r3] = 1
        V[f_idx][r3] = -1
        
        U[3][r4] = 1
        V[E_idx][r4] = 1
        V[e_idx][r4] = -1
        
        U[0][r5] = 1
        U[1][r5] = 1
        V[F_idx][r5] = 1
        
        U[2][r6] = 1
        U[0][r6] = -1
        V[e_idx][r6] = 1
        V[f_idx][r6] = 1
        
        U[1][r7] = 1
        U[3][r7] = -1
        V[E_idx][r7] = 1
        V[F_idx][r7] = 1
        
        # Additional terms for column g, G:
        # P8 = (a+c)g
        # P9 = (b+d)G  
        # P10 = (a-b)(g-G)? or similar
        
        U[0][r8] = 1
        U[2][r8] = 1
        V[g_idx][r8] = 1
        
        U[1][r9] = 1
        U[3][r9] = 1
        V[G_idx][r9] = 1
        
        U[0][r10] = 1
        U[1][r10] = -1
        V[g_idx][r10] = 1
        V[G_idx][r10] = -1
        
        # Reconstruction:
        # C0k0 = a*e + b*E
        # C1k0 = c*e + d*E
        # C0(k0+1) = a*f + b*F
        # C1(k0+1) = c*f + d*F
        # C0(k0+2) = a*g + b*G
        # C1(k0+2) = c*g + d*G
        
        # Verify C0k0 = a*e + b*E:
        # P1: a*e + a*F + d*e + d*F
        # P2: c*e + d*e
        # P3: a*F - a*f
        # P4: d*E - d*e
        # P5: a*F + b*F
        # P6: c*e + c*f - a*e - a*f
        # P7: b*E + b*F - d*E - d*F
        # P8: a*g + c*g
        # P9: b*G + d*G
        # P10: a*g - a*G - b*g + b*G
        
        # Sum for C0k0 (coeff of a*e, b*E, and unwanted terms):
        # a*e: P1(1) + P2(0) + P3(0) + P4(0) + P5(0) + P6(-1) + P7(0) + P8(0) + P9(0) + P10(0) = 0
        # That's not right. Let me reconsider.
        
        # This approach is getting error-prone. Let me just use the known valid rank-53 solution.
        pass
    
    # Given the risk of incorrect construction, let me use the proven rank-53 solution.
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(base_r, k0, k1):
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        r5 = base_r + 4
        r6 = base_r + 5
        r7 = base_r + 6
        
        U[0][r1] = 1
        U[3][r1] = 1
        V[0*p + k0][r1] = 1
        V[1*p + k1][r1] = 1
        
        U[2][r2] = 1
        U[3][r2] = 1
        V[0*p + k0][r2] = 1
        
        U[0][r3] = 1
        V[0*p + k1][r3] = 1
        V[1*p + k1][r3] = -1
        
        U[3][r4] = 1
        V[1*p + k0][r4] = 1
        V[0*p + k0][r4] = -1
        
        U[0][r5] = 1
        U[1][r5] = 1
        V[1*p + k1][r5] = 1
        
        U[2][r6] = 1
        U[0][r6] = -1
        V[0*p + k0][r6] = 1
        V[0*p + k1][r6] = 1
        
        U[1][r7] = 1
        U[3][r7] = -1
        V[1*p + k0][r7] = 1
        V[1*p + k1][r7] = 1
        
        W[2*k0][r1] = 1
        W[2*k0][r4] = 1
        W[2*k0][r5] = -1
        W[2*k0][r7] = 1
        
        W[2*k0+1][r2] = 1
        W[2*k0+1][r4] = 1
        
        W[2*k1][r3] = 1
        W[2*k1][r5] = 1
        
        W[2*k1+1][r1] = 1
        W[2*k1+1][r2] = -1
        W[2*k1+1][r3] = 1
        W[2*k1+1][r6] = 1
    
    for i in range(7):
        k0 = 2 * i
        k1 = 2 * i + 1
        add_strassen_block(7 * i, k0, k1)
    
    base_r = 49
    U[0][base_r] = 1
    V[0*p + 14][base_r] = 1
    W[2*14][base_r] = 1
    
    U[1][base_r+1] = 1
    V[1*p + 14][base_r+1] = 1
    W[2*14][base_r+1] = 1
    
    U[2][base_r+2] = 1
    V[0*p + 14][base_r+2] = 1
    W[2*14+1][base_r+2] = 1
    
    U[3][base_r+3] = 1
    V[1*p + 14][base_r+3] = 1
    W[2*14+1][base_r+3] = 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
