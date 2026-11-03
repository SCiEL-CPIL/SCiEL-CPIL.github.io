import time
import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [4, 4, 7]
WALL_DEADLINE = 145.0
SAFETY_MARGIN = 5.0

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7
    
    def check_valid(U, V, W, tol=1e-9):
        """Check if the decomposition is valid."""
        r = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U_arr[i*m + j, c] * V_arr[j*p + k, c] * W_arr[k*n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def build_schoolbook():
        """Build the trivial schoolbook decomposition of rank n*m*p."""
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return U, V, W
    
    def build_strassen_2x2x2():
        """
        Build Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
        Returns U (4x7), V (4x7), W (4x7) for the 2x2x2 case.
        U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # p1 = (a11 + a22)(b11 + b22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # p2 = (a21 + a22)b11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # p3 = a11(b12 - b22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # p4 = a22(b21 - b11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # p5 = (a11 + a12)b22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # p6 = (a21 - a11)(b11 + b12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # p7 = (a12 - a22)(b21 + b22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W
    
    def kronecker_decompose_4x4x4():
        """
        Build a decomposition for 4x4x4 using Kronecker product of two 2x2x2 Strassen decompositions.
        This gives rank 7*7 = 49.
        """
        U2, V2, W2 = build_strassen_2x2x2()
        
        U4 = [[0]*49 for _ in range(16)]
        V4 = [[0]*49 for _ in range(16)]
        W4 = [[0]*49 for _ in range(16)]
        
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                                if val != 0:
                                    U4[i*4+j][r] += val
                                  
                for j1 in range(2):
                    for k1 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                j = j1 * 2 + j2
                                k = k1 * 2 + k2
                                val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                                if val != 0:
                                    V4[j*4+k][r] += val
                                  
                for k1 in range(2):
                    for i1 in range(2):
                        for k2 in range(2):
                            for i2 in range(2):
                                k = k1 * 2 + k2
                                i = i1 * 2 + i2
                                val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                                if val != 0:
                                    W4[k*4+i][r] += val
        
        return U4, V4, W4
    
    def build_2x2x3_rank11():
        """
        Build rank-11 decomposition for 2x2x3 matrix multiplication.
        A is 2x2, B is 2x3, C is 2x2.
        U: 4x11 (indices i*2+j), V: 6x11 (indices j*3+k), W: 6x11 (indices k*2+i)
        """
        U = [[0]*11 for _ in range(4)]
        V = [[0]*11 for _ in range(6)]
        W = [[0]*11 for _ in range(6)]
        
        # First 7 terms: Strassen for 2x2x2 (columns 0,1 of B)
        U2, V2, W2 = build_strassen_2x2x2()
        for r in range(7):
            for idx in range(4):
                U[idx][r] = U2[idx][r]
            for idx in range(4):
                j = idx // 2
                k = idx % 2
                V[j*3 + k][r] = V2[idx][r]
            for idx in range(4):
                W[idx][r] = W2[idx][r]
        
        # Last 4 terms: schoolbook for 2x2x1 (column 2 of B)
        for i in range(2):
            for j in range(2):
                r = 7 + i*2 + j
                U[i*2 + j][r] = 1
                V[j*3 + 2][r] = 1
                W[2*2 + i][r] = 1
        
        return U, V, W
    
    def search_2x2x3_rank10():
        """
        Search for a rank-10 decomposition of the 2x2x3 matrix multiplication tensor.
        Uses linear algebra over rationals.
        """
        if time.time() - start_time > WALL_DEADLINE - SAFETY_MARGIN - 30:
            return None
        
        # 2x2x3 tensor: U is 4x10, V is 6x10, W is 6x10
        # We want: for all i,j,k: sum_c U[i*2+j,c] * V[j*3+k,c] * W[k*2+i,c] = 1
        
        # Strategy: Start from rank-11 and try to eliminate one term by finding
        # linear dependencies.
        
        U11, V11, W11 = build_2x2x3_rank11()
        
        # Try to find if any column is a linear combination of others
        # by setting up the tensor equations and checking rank
        
        # Build the tensor as a 3D array T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
        # We need T[i,j,k] = 1 for all i,j,k
        
        # Let's try a different approach: use the fact that the 2x2x3 tensor
        # can be viewed as a 4x6 matrix (flattening) with entries being linear
        # in the W factors.
        
        # Alternative: Try specific patterns. The 2x2x3 tensor has:
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
        
        # Let me try a manual construction with 10 terms.
        # Use Strassen-like structure but merge some terms.
        
        U = [[Fraction(0)]*10 for _ in range(4)]
        V = [[Fraction(0)]*10 for _ in range(6)]
        W = [[Fraction(0)]*10 for _ in range(6)]
        
        # Term 0: (a00+a11)(b00+b10) -> contributes to C[0,0], C[1,0], C[0,1], C[1,1]
        # Actually, let me use a known approach: the 2x2x3 tensor has rank at most 10
        # based on the upper bound min(n*m*p, ...) but let's try to construct it.
        
        # Let me try: use 6 terms for the "main" part and 4 for adjustments.
        
        # Actually, let me try a systematic approach using the border rank ideas.
        # For now, let's try a few random perturbations of the rank-11 solution
        # and check if we can drop one term.
        
        # Try: set one of the last 4 schoolbook terms to zero and see if the
        # remaining 10 terms can be adjusted.
        
        # This is complex. Let me try a different strategy:
        # Use the fact that 2x2x3 can be decomposed using the 2x2x2 Strassen
        # plus 3 additional terms (not 4).
        
        # C[:,2] = A @ B[:,2]
        # This is a 2x2 result from 2x2 times 2x1.
        # We can use: c02 = (a00+a10)*b02 - a10*b12 ... no, that's not right.
        # Actually for 2x2 @ 2x1, schoolbook is 4 terms. But we can use
        # (a00+a11)*(b02+b12) - a11*b02 - a00*b12 ... that's 3 terms but
        # it gives c02 + c12, not just c02.
        
        # Let me try: 
        # c02 = a00*b02 + a01*b12
        # c12 = a10*b02 + a11*b12
        # 
        # Using: (a00+a11)*b02 = a00*b02 + a11*b02
        #       (a01+a10)*b12 = a01*b12 + a10*b12
        #       a11*b02 - a10*b12 ... hmm
        #
        # Actually: c02 + c12 = (a00+a11)*b02 + (a01+a10)*b12 ... no
        # c02 + c12 = a00*b02 + a01*b12 + a10*b02 + a11*b12
        #            = (a00+a10)*b02 + (a01+a11)*b12
        #
        # c02 - c12 = a00*b02 + a01*b12 - a10*b02 - a11*b12
        #            = (a00-a10)*b02 + (a01-a11)*b12
        #
        # So: c02 = ((a00+a10)*b02 + (a01+a11)*b12 + (a00-a10)*b02 + (a01-a11)*b12) / 2
        # That's 4 terms still.
        
        # Let me just try the direct linear algebra approach with numpy.
        
        # Set up: we want to find U (4x10), V (6x10), W (6x10) such that
        # for all (i,j,k): sum_c U[i*2+j, c] * V[j*3+k, c] * W[k*2+i, c] = 1
        
        # This is a system of 24 equations (4*6=24 distinct (i,j,k) tuples... 
        # wait, i in {0,1}, j in {0,1}, k in {0,1,2}, so 2*2*3 = 12 equations)
        # with 3*10*... variables. This is underdetermined.
        
        # Let me fix V and W and solve for U, or use an iterative approach.
        
        # Strategy: Fix random V and W, then for each (i,j,k), the equation is:
        # sum_c U[i*2+j, c] * (V[j*3+k, c] * W[k*2+i, c]) = 1
        # This is a linear system in U.
        
        # But U appears in multiple equations. Let me set up the full system.
        
        # Variables: U[0..3, 0..9], V[0..5, 0..9], W[0..5, 0..9]
        # That's 4*10 + 6*10 + 6*10 = 160 variables.
        # 12 equations. Very underdetermined.
        
        # Let me fix V and W to simple values and solve for U.
        
        # Fix V: V[j*3+k, c] = 1 if j==c%6//3 and k==c%3, else 0
        # i.e., V is a "selection" matrix
        # Fix W: W[k*2+i, c] = 1 if k==c%6//2 and i==c%2, else 0
        
        # Then the equation becomes:
        # sum_c U[i*2+j, c] * V[j*3+k, c] * W[k*2+i, c] = 1
        # For fixed (i,j,k): V[j*3+k,c] is nonzero only when c is in a specific set
        # and W[k*2+i,c] is nonzero only when c is in another specific set.
        
        # This is getting complex. Let me just try to verify if rank 10 is possible
        # by checking the border rank or using a known result.
        
        # Known: The 2x2x3 tensor has rank at most 10 (from the general bound).
        # Let me try to construct it explicitly.
        
        # Use the following 10-term decomposition:
        # Terms 0-6: Modified Strassen for the 2x2x2 part (columns 0,1)
        # Terms 7-9: 3 terms for column 2
        
        # For column 2, we need:
        # c02 = a00*b02 + a01*b12
        # c12 = a10*b02 + a11*b12
        
        # Using 3 terms:
        # t7 = (a00+a11) * b02 -> contributes to c02 and c12
        # t8 = a01 * b12 -> contributes to c02
        # t9 = a10 * b12 -> contributes to c12
        # But then c02 = t7_part + t8, c12 = t7_part + t9
        # We need to separate the t7 contributions.
        
        # Alternative:
        # t7 = a00 * b02
        # t8 = a01 * b12  
        # t9 = (a10+a11) * b12 ... no, that gives a10*b12 + a11*b12 for c12
        # but we also need a10*b02 for c12.
        
        # It seems 4 terms are needed for the 2x2x1 part unless we can
        # share with the Strassen terms.
        
        # Let me try a completely different approach: use the 2x2x3 tensor
        # as a sub-tensor of a larger structure.
        
        # For now, return None to indicate rank 10 was not found.
        return None
    
    def build_4x4x7_decomposition():
        """
        Build a rank-93 decomposition for 4x4x7.
        
        Strategy: Use Kronecker Strassen (rank 49) for the 4x4x4 part.
        For the 4x4x3 part (columns 4,5,6 of B), use four 2x2x3 blocks
        each of rank 11, total 44. Grand total: 49 + 44 = 93.
        """
        U4, V4, W4 = kronecker_decompose_4x4x4()
        r4 = len(U4[0])  # 49
        
        U223, V223, W223 = build_2x2x3_rank11()
        r223 = len(U223[0])  # 11
        
        r_total = r4 + 4 * r223  # 49 + 44 = 93
        U = [[0] * r_total for _ in range(16)]
        V = [[0] * r_total for _ in range(28)]
        W = [[0] * r_total for _ in range(28)]
        
        # Part 1: 4x4x4 Kronecker (columns 0-3 of B)
        for i in range(16):
            for r in range(r4):
                U[i][r] = U4[i][r]
        
        for j in range(4):
            for k in range(4):
                for r in range(r4):
                    V[j*7 + k][r] = V4[j*4 + k][r]
        
        for k in range(4):
            for i in range(4):
                for r in range(r4):
                    W[k*4 + i][r] = W4[k*4 + i][r]
        
        # Part 2: 4x4x3 using four 2x2x3 blocks (columns 4-6 of B)
        col_offset = r4
        for row_group in range(2):
            for col_group in range(2):
                base_r = col_offset + (row_group * 2 + col_group) * r223
                
                for r_local in range(r223):
                    r_global = base_r + r_local
                    
                    for i_local in range(2):
                        for j_local in range(2):
                            i = 2 * row_group + i_local
                            j = 2 * col_group + j_local
                            U[i*4 + j][r_global] = U223[i_local*2 + j_local][r_local]
                    
                    for j_local in range(2):
                        for k_local in range(3):
                            j = 2 * col_group + j_local
                            k = 4 + k_local
                            V[j*7 + k][r_global] = V223[j_local*3 + k_local][r_local]
                    
                    for k_local in range(3):
                        for i_local in range(2):
                            k = 4 + k_local
                            i = 2 * row_group + i_local
                            W[k*4 + i][r_global] = W223[k_local*2 + i_local][r_local]
        
        return U, V, W
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = build_schoolbook()
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = 112
    
    # Try the optimized approach
    if time.time() - start_time < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            U_opt, V_opt, W_opt = build_4x4x7_decomposition()
            r_opt = len(U_opt[0])
            
            if check_valid(U_opt, V_opt, W_opt):
                if r_opt < best_rank:
                    best_result = {"U": U_opt, "V": V_opt, "W": W_opt}
                    best_rank = r_opt
        except Exception as e:
            pass
    
    return best_result
