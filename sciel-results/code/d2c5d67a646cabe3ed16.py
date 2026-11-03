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

    def build_strassen_2x2_2x2():
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
        U2, V2, W2 = build_strassen_2x2_2x2()
        
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

    def build_2x2x3_rank12():
        """
        Build a rank-12 decomposition for 2x2x3 matrix multiplication.
        Uses Strassen for 2x2x2 (rank 7) + schoolbook for 2x2x1 (rank 4) = 11, 
        but let's use a known valid rank-12 or try to construct rank-11/12.
        
        Actually, let's use: Strassen 2x2x2 (7) + 2x2x1 schoolbook (4) = 11
        U: 4x11, V: 6x11, W: 6x11
        """
        U2, V2, W2 = build_strassen_2x2_2x2()
        
        r = 11
        U = [[0]*r for _ in range(4)]
        V = [[0]*r for _ in range(6)]
        W = [[0]*r for _ in range(6)]
        
        # First 7: Strassen for columns 0,1 of B
        for c in range(7):
            for idx in range(4):
                i = idx // 2
                j = idx % 2
                U[i*2+j][c] = U2[idx][c]
            
            for idx in range(4):
                j = idx // 2
                k = idx % 2
                V[j*3+k][c] = V2[idx][c]
            
            for idx in range(4):
                k = idx // 2
                i = idx % 2
                W[k*2+i][c] = W2[idx][c]
        
        # Last 4: schoolbook for column 2 of B
        for i in range(2):
            for j in range(2):
                c = 7 + i*2 + j
                U[i*2+j][c] = 1
                V[j*3+2][c] = 1
                W[2*2+i][c] = 1
        
        return U, V, W

    def build_2x2x3_rank9():
        """
        Build a rank-9 decomposition for 2x2x3 matrix multiplication.
        This is a candidate that needs verification.
        U: 4x9 (indices i*2+j), V: 6x9 (indices j*3+k), W: 6x9 (indices k*2+i)
        """
        # Try a construction based on combining Strassen terms
        # p1 = a11*b11, p2 = a11*b12, p3 = a11*b13
        # p4 = a21*b21, p5 = a21*b22, p6 = a21*b23
        # p7 = a12*b11, p8 = a12*b12, p9 = a12*b13 + a22*b21 + a22*b22 + a22*b23 - corrections
        # This is complex; let's use a verified rank-11 instead
        return build_2x2x3_rank12()[:1], build_2x2x3_rank12()[:1], build_2x2x3_rank12()[:1]

    def build_2x2x3_rank11():
        """
        Build rank-11 decomposition for 2x2x3 matrix multiplication (fallback).
        """
        U = [[0]*11 for _ in range(4)]
        V = [[0]*11 for _ in range(6)]
        W = [[0]*11 for _ in range(6)]
        
        # First 7 terms: Strassen for 2x2x2 (columns 0,1 of B)
        U2, V2, W2 = build_strassen_2x2_2x2()
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

    def build_4x4x7_decomposition(use_rank9):
        """
        Build a decomposition for 4x4x7.
        Rank 49 (4x4x4) + 4 * (9 or 11) for 4x4x3.
        """
        U4, V4, W4 = kronecker_decompose_4x4x4()
        r4 = len(U4[0])  # 49
        
        if use_rank9:
            U223, V223, W223 = build_2x2x3_rank9()
            r223 = 9
        else:
            U223, V223, W223 = build_2x2x3_rank11()
            r223 = 11
            
        r_total = r4 + 4 * r223
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
    
    # Build rank-11 version (known valid: 49 + 4*11 = 93)
    if time.time() - start_time < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            U_opt, V_opt, W_opt = build_4x4x7_decomposition(use_rank9=False)
            r_opt = len(U_opt[0])
            if check_valid(U_opt, V_opt, W_opt):
                if r_opt < best_rank:
                    best_result = {"U": U_opt, "V": V_opt, "W": W_opt}
                    best_rank = r_opt
        except Exception as e:
            pass

    return best_result
