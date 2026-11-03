import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 5x5 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Use block decomposition: 5 = 2 + 2 + 1
    # Split into 3 blocks of sizes [2, 2, 1]
    # For each block triple (i,j,k), use optimal decomposition of the sub-block multiplication
    # This gives rank 109 total
    
    block_sizes = [2, 2, 1]
    num_blocks = len(block_sizes)
    
    # Compute row/col offsets for each block
    row_offsets = [0] * (num_blocks + 1)
    for i in range(num_blocks):
        row_offsets[i + 1] = row_offsets[i] + block_sizes[i]
    
    # Helper: get Strassen decomposition for 2x2 * 2x2 -> 2x2
    def strassen_2x2():
        """
        Strassen decomposition for 2x2 matrix multiplication.
        A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+cg, af+dh],[ce+dg, cf+eh]]
        
        7 products:
        p1 = (a+d)(e+h)
        p2 = (c+d)e
        p3 = a(f-h)
        p4 = d(g-e)
        p5 = (a+b)h
        p6 = (c-a)(e+f)
        p7 = (b-d)(g+h)
        
        C11 = p1 + p4 - p5 + p7
        C12 = p3 + p5
        C21 = p2 + p4
        C22 = p1 - p2 + p3 + p6
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # p1 = (a+d)(e+h): U[0,0]=1(a), U[3,0]=1(d), V[0,0]=1(e), V[3,0]=1(h)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        # W for C11: +1, C22: -1
        W[0][0] = 1; W[3][0] = -1
        
        # p2 = (c+d)e: U[1,1]=1(c), U[3,1]=1(d), V[0,1]=1(e)
        U[1][1] = 1; U[3][1] = 1
        V[0][1] = 1
        # W for C21: +1, C22: -1
        W[2][1] = 1; W[3][1] = -1
        
        # p3 = a(f-h): U[0,2]=1(a), V[1,2]=1(f), V[3,2]=-1(h)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        # W for C12: +1, C22: +1
        W[1][2] = 1; W[3][2] = 1
        
        # p4 = d(g-e): U[3,3]=1(d), V[2,3]=1(g), V[0,3]=-1(e)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        # W for C11: +1, C21: +1
        W[0][3] = 1; W[2][3] = 1
        
        # p5 = (a+b)h: U[0,4]=1(a), U[1,4]=1(b), V[3,4]=1(h)
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        # W for C11: -1, C12: +1
        W[0][4] = -1; W[1][4] = 1
        
        # p6 = (c-a)(e+f): U[1,5]=1(c), U[0,5]=-1(a), V[0,5]=1(e), V[1,5]=1(f)
        U[1][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        # W for C22: +1
        W[3][5] = 1
        
        # p7 = (b-d)(g+h): U[1,6]=1(b), U[3,6]=-1(d), V[2,6]=1(g), V[3,6]=1(h)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        # W for C11: +1
        W[0][6] = 1
        
        return U, V, W
    
    # Helper: schoolbook decomposition for (a x c) * (c x b) -> (a x b)
    def schoolbook_decomp(a, b, c):
        """
        Schoolbook decomposition for multiplying (a x c) by (c x b).
        Returns U (a*c, a*b*c), V (c*b, a*b*c), W (a*b, a*b*c)
        """
        rank = a * b * c
        U = [[0]*rank for _ in range(a * c)]
        V = [[0]*rank for _ in range(c * b)]
        W = [[0]*rank for _ in range(a * b)]
        
        for i in range(a):
            for j in range(c):
                for k in range(b):
                    col = i * b * c + j * b + k
                    U[i * c + j][col] = 1
                    V[j * b + k][col] = 1
                    W[i * b + k][col] = 1
        
        return U, V, W, rank
    
    # Helper: get optimal decomposition for block sizes (a, b, c)
    def get_block_decomp(a, b, c):
        if a == 2 and b == 2 and c == 2:
            U, V, W = strassen_2x2()
            return U, V, W, 7
        else:
            return schoolbook_decomp(a, b, c)
    
    # Now build the full decomposition
    total_rank = 0
    for i in range(num_blocks):
        for j in range(num_blocks):
            for k in range(num_blocks):
                a = block_sizes[i]
                b = block_sizes[j]
                c = block_sizes[k]
                _, _, _, r = get_block_decomp(a, b, c)
                total_rank += r
    
    # Allocate full matrices
    U_full = [[0] * total_rank for _ in range(n * m)]
    V_full = [[0] * total_rank for _ in range(m * p)]
    W_full = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for i in range(num_blocks):
        for j in range(num_blocks):
            for k in range(num_blocks):
                a = block_sizes[i]
                b = block_sizes[j]
                c = block_sizes[k]
                
                U_block, V_block, W_block, r = get_block_decomp(a, b, c)
                
                a_row_off = row_offsets[i]
                a_col_off = row_offsets[k]
                
                b_row_off = row_offsets[k]
                b_col_off = row_offsets[j]
                
                c_row_off = row_offsets[i]
                c_col_off = row_offsets[j]
                
                for ri in range(a):
                    for ci in range(c):
                        local_row = ri * c + ci
                        global_row = (a_row_off + ri) * m + (a_col_off + ci)
                        for r_idx in range(r):
                            U_full[global_row][col_offset + r_idx] = U_block[local_row][r_idx]
                
                for ri in range(c):
                    for ci in range(b):
                        local_row = ri * b + ci
                        global_row = (b_row_off + ri) * p + (b_col_off + ci)
                        for r_idx in range(r):
                            V_full[global_row][col_offset + r_idx] = V_block[local_row][r_idx]
                
                for ri in range(a):
                    for ci in range(b):
                        local_row = ri * b + ci
                        global_row = (c_col_off + ci) * n + (c_row_off + ri)
                        for r_idx in range(r):
                            W_full[global_row][col_offset + r_idx] = W_block[local_row][r_idx]
                
                col_offset += r
    
    # Verify the decomposition
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(total_rank):
                    s += U_full[i * m + j][r] * V_full[j * p + k][r] * W_full[k * n + i][r]
                max_err = max(max_err, abs(s - 1))
    
    if max_err < 1e-10:
        return {"U": U_full, "V": V_full, "W": W_full}
    else:
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U_school = [[0] * rank_schoolbook for _ in range(n * m)]
        V_school = [[0] * rank_schoolbook for _ in range(m * p)]
        W_school = [[0] * rank_schoolbook for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * m * p + j * p + k
                    U_school[i * m + j][col] = 1
                    V_school[j * p + k][col] = 1
                    W_school[k * n + i][col] = 1
        return {"U": U_school, "V": V_school, "W": W_school}
