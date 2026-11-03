"""Tensor decomposition for 4x4x7 matrix multiplication using Strassen + Kronecker."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7
    
    def build_strassen_2x2_2x2():
        """
        Strassen's rank-7 decomposition for 2x2x2.
        U: 4x7 indexed by i*2+j (i=row of A, j=col of A)
        V: 4x7 indexed by j*2+k (j=row of B, k=col of B)
        W: 4x7 indexed by k*2+i (k=col of C, i=row of C)
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # p1 = (a11+a22)(b11+b22)
        U[0][0]=1; U[3][0]=1
        V[0][0]=1; V[3][0]=1
        W[0][0]=1; W[3][0]=1
        
        # p2 = (a21+a22)b11
        U[2][1]=1; U[3][1]=1
        V[0][1]=1
        W[1][1]=1; W[3][1]=-1
        
        # p3 = a11(b12-b22)
        U[0][2]=1
        V[1][2]=1; V[3][2]=-1
        W[2][2]=1; W[3][2]=1
        
        # p4 = a22(b21-b11)
        U[3][3]=1
        V[2][3]=1; V[0][3]=-1
        W[0][3]=1; W[1][3]=1
        
        # p5 = (a11+a12)b22
        U[0][4]=1; U[1][4]=1
        V[3][4]=1
        W[0][4]=-1; W[2][4]=1
        
        # p6 = (a21-a11)(b11+b12)
        U[2][5]=1; U[0][5]=-1
        V[0][5]=1; V[1][5]=1
        W[3][5]=1
        
        # p7 = (a12-a22)(b21+b22)
        U[1][6]=1; U[3][6]=-1
        V[2][6]=1; V[3][6]=1
        W[0][6]=1
        
        return U, V, W
    
    def kronecker_4x4x4():
        """
        Kronecker product of two Strassen 2x2x2 decompositions.
        Gives rank 49 for 4x4x4.
        U: 16x49, V: 16x49, W: 16x49
        """
        U2, V2, W2 = build_strassen_2x2_2x2()
        r = 49
        U4 = [[0]*r for _ in range(16)]
        V4 = [[0]*r for _ in range(16)]
        W4 = [[0]*r for _ in range(16)]
        
        for r1 in range(7):
            for r2 in range(7):
                rc = r1*7 + r2
                # U part: (i1,j1) from first, (i2,j2) from second
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i1*2+i2
                                j = j1*2+j2
                                val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                                if val != 0:
                                    U4[i*4+j][rc] += val
                # V part: (j1,k1) from first, (j2,k2) from second
                for j1 in range(2):
                    for k1 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                j = j1*2+j2
                                k = k1*2+k2
                                val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                                if val != 0:
                                    V4[j*4+k][rc] += val
                # W part: (k1,i1) from first, (k2,i2) from second
                for k1 in range(2):
                    for i1 in range(2):
                        for k2 in range(2):
                            for i2 in range(2):
                                k = k1*2+k2
                                i = i1*2+i2
                                val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                                if val != 0:
                                    W4[k*4+i][rc] += val
        
        return U4, V4, W4
    
    def build_2x2x3():
        """
        Rank-11 decomposition for 2x2x3.
        A is 2x2, B is 2x3, C is 2x3.
        U: 4x11 indexed by i*2+j
        V: 6x11 indexed by j*3+k
        W: 6x11 indexed by k*2+i
        """
        U2, V2, W2 = build_strassen_2x2_2x2()
        r = 11
        U = [[0]*r for _ in range(4)]
        V = [[0]*r for _ in range(6)]
        W = [[0]*r for _ in range(6)]
        
        # First 7 terms: Strassen for 2x2x2 (first 2 columns of B)
        for rc in range(7):
            for idx in range(4):
                U[idx][rc] = U2[idx][rc]
            for idx in range(4):
                j = idx // 2
                k = idx % 2
                V[j*3+k][rc] = V2[idx][rc]
            for idx in range(4):
                W[idx][rc] = W2[idx][rc]
        
        # Last 4 terms: schoolbook for 2x2x1 (column 2 of B)
        for i in range(2):
            for j in range(2):
                rc = 7 + i*2 + j
                U[i*2+j][rc] = 1
                V[j*3+2][rc] = 1
                W[2*2+i][rc] = 1  # k=2, so W[4+i]
        
        return U, V, W
    
    # Build the full 4x4x7 decomposition
    U4, V4, W4 = kronecker_4x4x4()
    U223, V223, W223 = build_2x2x3()
    
    r4 = 49
    r223 = 11
    r_total = r4 + 4*r223  # 49 + 44 = 93
    
    U = [[0]*r_total for _ in range(16)]
    V = [[0]*r_total for _ in range(28)]
    W = [[0]*r_total for _ in range(28)]
    
    # Part 1: 4x4x4 (columns 0-3 of B)
    for i in range(16):
        for rc in range(r4):
            U[i][rc] = U4[i][rc]
    for j in range(4):
        for k in range(4):
            for rc in range(r4):
                V[j*7+k][rc] = V4[j*4+k][rc]
    for k in range(4):
        for i in range(4):
            for rc in range(r4):
                W[k*4+i][rc] = W4[k*4+i][rc]
    
    # Part 2: 4x4x3 using four 2x2x3 blocks (columns 4-6 of B)
    col_offset = r4
    for row_group in range(2):
        for col_group in range(2):
            base_r = col_offset + (row_group*2+col_group)*r223
            for rc_local in range(r223):
                rc_global = base_r + rc_local
                # U: rows 2*row_group and 2*row_group+1, all 4 cols of A
                for i_local in range(2):
                    for j_local in range(2):
                        i = 2*row_group + i_local
                        j = 2*col_group + j_local
                        U[i*4+j][rc_global] = U223[i_local*2+j_local][rc_local]
                # V: rows 2*col_group and 2*col_group+1 of B, cols 4-6
                for j_local in range(2):
                    for k_local in range(3):
                        j = 2*col_group + j_local
                        k = 4 + k_local
                        V[j*7+k][rc_global] = V223[j_local*3+k_local][rc_local]
                # W: cols 4-6 of C, rows 2*row_group and 2*row_group+1
                for k_local in range(3):
                    for i_local in range(2):
                        k = 4 + k_local
                        i = 2*row_group + i_local
                        W[k*4+i][rc_global] = W223[k_local*2+i_local][rc_local]
    
    return {"U": U, "V": V, "W": W}
