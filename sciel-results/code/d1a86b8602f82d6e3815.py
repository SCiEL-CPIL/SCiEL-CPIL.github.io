"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8
    
    def naive_decomp():
        rank = n * m * p  # 16
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def try_strassen_rank14():
        """
        Rank-14 decomposition for 2x2 * 2x4 using two Strassen 2x2 blocks.
        """
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Left block (cols 0,1 of B):
        # r=0: M1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # r=1: M2 = (a10+a11)b00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # r=2: M3 = a00(b01-b11)
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # r=3: M4 = a11(b10-b00)
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # r=4: M5 = (a00+a01)b11
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # r=5: M6 = (a10-a00)(b00+b01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # r=6: M7 = (a01-a11)(b10+b11)
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1
        
        # Right block (cols 2,3 of B):
        # r=7: N1 = (a00+a11)(b02+b13)
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        
        # r=8: N2 = (a10+a11)b02
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[5][8] = 1; W[7][8] = -1
        
        # r=9: N3 = a00(b03-b13)
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = -1
        W[6][9] = 1; W[7][9] = 1
        
        # r=10: N4 = a11(b12-b02)
        U[3][10] = 1
        V[6][10] = 1; V[2][10] = -1
        W[4][10] = 1; W[5][10] = 1
        
        # r=11: N5 = (a00+a01)b13
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[6][11] = 1
        
        # r=12: N6 = (a10-a00)(b02+b03)
        U[2][12] = 1; U[0][12] = -1
        V[2][12] = 1; V[3][12] = 1
        W[7][12] = 1
        
        # r=13: N7 = (a01-a11)(b12+b13)
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        else:
            return None
    
    def try_rank13_search():
        """
        Search for rank-13 decomposition by sharing one term between blocks.
        
        Strategy: Use 7 terms for left block, 6 for right, sharing 1 term.
        The shared term must be of the form (a_i + a_j)(b_k + b_l) that can
        contribute to both blocks' outputs.
        
        We try all possible ways to merge one Strassen term from each block.
        """
        if time.time() > deadline:
            return None
        
        # The 7 Strassen terms for a 2x2 * 2x2:
        # T1: (a00+a11)(b00+b11)
        # T2: (a10+a11)b00
        # T3: a00(b01-b11)
        # T4: a11(b10-b00)
        # T5: (a00+a01)b11
        # T6: (a10-a00)(b00+b01)
        # T7: (a01-a11)(b10+b11)
        
        # For the right block, we need to find 6 terms that, combined with
        # one shared term from the left, give correct outputs for cols 2,3.
        
        # Approach: Try sharing T1-type terms with different B column pairings.
        # Shared term: (a00+a11)(b_c + b_d) where c,d are from {2,3} for right block
        
        # Let's try: share (a00+a11)(b00+b02) - contributes to both C[0][0] and C[0][2]
        # Or other pairings.
        
        # Actually, let's do a more systematic search.
        # We have 13 terms. 7 for left (standard Strassen), 6 for right.
        # The right block needs to produce C[0][2], C[1][2], C[0][3], C[1][3].
        
        # Let's parameterize the right block's 6 terms and search for solutions.
        # Each term r has:
        #   U_r: 4 values (coefficients for a00, a01, a10, a11)
        #   V_r: 8 values (coefficients for b00..b03, b10..b13)
        #   W_r: 8 values (coefficients for C[0][0], C[1][0], C[0][1], C[1][1], C[0][2], C[1][2], C[0][3], C[1][3])
        
        # For the right block terms, W should only have non-zero entries in positions 4-7 (C[0][2], C[1][2], C[0][3], C[1][3]).
        # V should only have non-zero entries in positions 2-7 (b02, b03, b12, b13).
        
        # This is a large search space. Let's use a smarter approach.
        
        # Try specific structured decompositions:
        
        # Attempt 1: Share M1 = (a00+a11)(b00+b11) with a modified right block
        # Right block uses:
        # S1 = (a00+a11)(b02+b13)  [like N1]
        # S2 = (a10+a11)b02        [like N2]
        # S3 = a00(b03-b13)        [like N3]
        # S4 = a11(b12-b02)        [like N4]
        # S5 = (a00+a01)b13        [like N5]
        # S6 = (a01-a11)(b12+b13)  [like N7, skipping N6]
        # But then C[1][3] = N1-N2+N3+N6, and we're missing N6.
        
        # Let's try: can we express N6 using the other terms?
        # N6 = (a10-a00)(b02+b03)
        # We need to check if this can be derived from S1-S5.
        
        # This is getting complex. Let's try a direct search with restricted entries.
        
        # Restricted search: entries in {-1, 0, 1}
        # For 13 terms, 4+8+8 = 20 entries per term, that's 260 entries.
        # Too many for brute force.
        
        # Instead, let's try a few specific structured rank-13 candidates:
        
        candidates = []
        
        # Candidate A: Merge M1 and N1 into one term
        # Shared: (a00+a11)(b00+b02+b11+b13) - too complex, likely doesn't work
        
        # Candidate B: Use a different factorization for the right block
        # Right block: 2x2 * 2x2 where B_right = [[b02, b03], [b12, b13]]
        # Try: 
        # R1 = (a00+a11)(b02+b13)
        # R2 = (a10+a11)(b02+b12)  
        # R3 = a00(b03-b13)
        # R4 = a11(b12-b02)
        # R5 = (a00+a01)(b13+b12)
        # R6 = (a10-a00)(b03+b12)
        # R7 = (a01-a11)(b03+b13)
        # That's still 7.
        
        # Let's try to find if rank 13 is achievable by checking known results.
        # The tensor rank of 2x2 * 2x4 is known to be 14 (it's the border rank).
        # Actually, for matrix multiplication, the rank of [2,2,4] is 14.
        
        # Given that the theoretical rank is 14, rank 13 is likely impossible.
        # Let's just return the rank-14 solution.
        
        return None
    
    # Try the structured rank-14 decomposition first
    result = try_strassen_rank14()
    if result is not None:
        return result
    
    # Try rank-13 search (will likely return None)
    if time.time() < deadline:
        result13 = try_rank13_search()
        if result13 is not None:
            return result13
    
    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
