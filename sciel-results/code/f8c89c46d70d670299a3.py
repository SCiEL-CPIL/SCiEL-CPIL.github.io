"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x7 matrix multiplication.
    Uses block decomposition with Strassen for 2x2x2 sub-blocks.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0
    
    # We'll build the decomposition block by block.
    # Split: n=3=1+2, m=4=2+2, p=7=3+4
    # This gives 8 sub-problems, each handled with minimal rank.
    
    # Helper: create a Strassen M_{2,2,2} decomposition
    # Returns U (4x7), V (4x7), W (4x7) where U corresponds to A (2x2), V to B (2x2), W to C (2x2)
    def strassen_222():
        # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        # C = A*B
        # U maps A entries to the 7 Strassen products
        # V maps B entries to the 7 Strassen products
        # W maps C entries to the 7 Strassen products
        
        # A indices: 0=a00, 1=a01, 2=a10, 3=a11
        # B indices: 0=b00, 1=b01, 2=b10, 3=b11
        # C indices: 0=c00, 1=c01, 2=c10, 3=c11
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # m1 = (a00+a11)(b00+b11)
        U[0][0] = 1  # a00
        U[3][0] = 1  # a11
        V[0][0] = 1  # b00
        V[3][0] = 1  # b11
        
        # m2 = (a10+a11)b00
        U[2][1] = 1  # a10
        U[3][1] = 1  # a11
        V[0][1] = 1  # b00
        
        # m3 = a00(b01-b11)
        U[0][2] = 1  # a00
        V[1][2] = 1  # b01
        V[3][2] = -1  # b11
        
        # m4 = a11(b10-b00)
        U[3][3] = 1  # a11
        V[2][3] = 1  # b10
        V[0][3] = -1  # b00
        
        # m5 = (a00+a01)b11
        U[0][4] = 1  # a00
        U[1][4] = 1  # a01
        V[3][4] = 1  # b11
        
        # m6 = (a10-a00)(b00+b01)
        U[2][5] = 1  # a10
        U[0][5] = -1  # a00
        V[0][5] = 1  # b00
        V[1][5] = 1  # b01
        
        # m7 = (a01-a11)(b10+b11)
        U[1][6] = 1  # a01
        U[3][6] = -1  # a11
        V[2][6] = 1  # b10
        V[3][6] = 1  # b11
        
        # C entries:
        # c00 = m1 + m4 - m5 + m7
        W[0][0] = 1   # m1
        W[0][3] = 1   # m4
        W[0][4] = -1  # m5
        W[0][6] = 1   # m7
        
        # c01 = m3 + m5
        W[1][2] = 1   # m3
        W[1][4] = 1   # m5
        
        # c10 = m2 + m4
        W[2][1] = 1   # m2
        W[2][3] = 1   # m4
        
        # c11 = m1 - m2 + m3 + m6
        W[3][0] = 1   # m1
        W[3][1] = -1  # m2
        W[3][2] = 1   # m3
        W[3][5] = 1   # m6
        
        return U, V, W
    
    # Helper: schoolbook M_{a,b,c} decomposition
    # Returns U (a*b x rank), V (b*c x rank), W (c*a x rank)
    def schoolbook(a, b, c):
        rank = a * b * c
        U = [[0]*rank for _ in range(a*b)]
        V = [[0]*rank for _ in range(b*c)]
        W = [[0]*rank for _ in range(c*a)]
        col = 0
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    U[i*b + j][col] = 1
                    V[j*c + k][col] = 1
                    W[k*a + i][col] = 1
                    col += 1
        return U, V, W
    
    # Helper: M_{1,2,2} schoolbook (rank 4)
    def m_1_2_2():
        # A is 1x2, B is 2x2, C is 1x2
        # A = [a0, a1], B = [[b00,b01],[b10,b11]], C = [c0, c1]
        # c0 = a0*b00 + a1*b10
        # c1 = a0*b01 + a1*b11
        U = [[0]*4 for _ in range(2)]  # A has 2 entries
        V = [[0]*4 for _ in range(4)]  # B has 4 entries
        W = [[0]*4 for _ in range(2)]  # C has 2 entries
        
        # c0 = a0*b00 + a1*b10
        U[0][0] = 1  # a0
        V[0][0] = 1  # b00
        W[0][0] = 1  # c0
        
        U[1][1] = 1  # a1
        V[2][1] = 1  # b10
        W[0][1] = 1  # c0
        
        # c1 = a0*b01 + a1*b11
        U[0][2] = 1  # a0
        V[1][2] = 1  # b01
        W[1][2] = 1  # c1
        
        U[1][3] = 1  # a1
        V[3][3] = 1  # b11
        W[1][3] = 1  # c1
        
        return U, V, W
    
    # Helper: M_{1,2,1} schoolbook (rank 2)
    def m_1_2_1():
        # A is 1x2, B is 2x1, C is 1x1
        # A = [a0, a1], B = [b0, b1], C = [c0]
        # c0 = a0*b0 + a1*b1
        U = [[0]*2 for _ in range(2)]
        V = [[0]*2 for _ in range(2)]
        W = [[0]*2 for _ in range(1)]
        
        U[0][0] = 1  # a0
        V[0][0] = 1  # b0
        W[0][0] = 1  # c0
        
        U[1][1] = 1  # a1
        V[1][1] = 1  # b1
        W[0][1] = 1  # c0
        
        return U, V, W
    
    # Helper: M_{2,2,1} schoolbook (rank 4)
    def m_2_2_1():
        # A is 2x2, B is 2x1, C is 2x1
        # A = [[a00,a01],[a10,a11]], B = [b0, b1], C = [c0, c1]
        # c0 = a00*b0 + a01*b1
        # c1 = a10*b0 + a11*b1
        U = [[0]*4 for _ in range(4)]
        V = [[0]*4 for _ in range(2)]
        W = [[0]*4 for _ in range(2)]
        
        # c0 = a00*b0 + a01*b1
        U[0][0] = 1  # a00
        V[0][0] = 1  # b0
        W[0][0] = 1  # c0
        
        U[1][1] = 1  # a01
        V[1][1] = 1  # b1
        W[0][1] = 1  # c0
        
        # c1 = a10*b0 + a11*b1
        U[2][2] = 1  # a10
        V[0][2] = 1  # b0
        W[1][2] = 1  # c1
        
        U[3][3] = 1  # a11
        V[1][3] = 1  # b1
        W[1][3] = 1  # c1
        
        return U, V, W
    
    # Now build the full decomposition
    # Split: i in {0} vs {1,2}, j in {0,1} vs {2,3}, k in {0,1,2} vs {3,4,5,6}
    
    # We'll accumulate U, V, W matrices
    # U has 3*4 = 12 rows
    # V has 4*7 = 28 rows  
    # W has 7*3 = 21 rows
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_block(U_block, V_block, W_block, i_offset, j_offset, k_offset, a, b, c):
        """
        Add a block decomposition to the global U, V, W.
        U_block: (a*b) x rank matrix for A[i_offset:i_offset+a, j_offset:j_offset+b]
        V_block: (b*c) x rank matrix for B[j_offset:j_offset+b, k_offset:k_offset+c]
        W_block: (c*a) x rank matrix for C[k_offset:k_offset+c, i_offset:i_offset+a]
        """
        rank = len(U_block[0]) if U_block else 0
        if rank == 0:
            return
        
        # Extend U, V, W with zeros for new columns
        while len(U_parts) > 0 and len(U_parts[0]) < len(U_block[0]):
            for i in range(len(U_parts)):
                U_parts[i].append(0)
        while len(V_parts) > 0 and len(V_parts[0]) < len(V_block[0]):
            for i in range(len(V_parts)):
                V_parts[i].append(0)
        while len(W_parts) > 0 and len(W_parts[0]) < len(W_block[0]):
            for i in range(len(W_parts)):
                W_parts[i].append(0)
        
        # Initialize if empty
        if not U_parts:
            U_parts = [[0]*rank for _ in range(n*m)]
            V_parts = [[0]*rank for _ in range(m*p)]
            W_parts = [[0]*rank for _ in range(p*n)]
        
        # Add U_block to appropriate rows
        for di in range(a):
            for dj in range(b):
                u_row = (i_offset + di) * m + (j_offset + dj)
                for c in range(rank):
                    U_parts[u_row][c] += U_block[di*b + dj][c]
        
        # Add V_block to appropriate rows
        for dj in range(b):
            for dk in range(c):
                v_row = (j_offset + dj) * p + (k_offset + dk)
                for c in range(rank):
                    V_parts[v_row][c] += V_block[dj*c + dk][c]
        
        # Add W_block to appropriate rows
        for dk in range(c):
            for di in range(a):
                w_row = (k_offset + dk) * n + (i_offset + di)
                for c in range(rank):
                    W_parts[w_row][c] += W_block[dk*a + di][c]
    
    # Block 1: i=0, j=0,1, k=0,1,2 → M_{1,2,3} = M_{1,2,2} + M_{1,2,1}
    U1, V1, W1 = m_1_2_2()
    U1b, V1b, W1b = m_1_2_1()
    # Concatenate: M_{1,2,3} has rank 4+2=6
    U_1_2_3 = [row + row2 for row, row2 in zip(U1, U1b)]
    V_1_2_3 = [row + row2 for row, row2 in zip(V1, V1b)]
    W_1_2_3 = [row + row2 for row, row2 in zip(W1, W1b)]
    add_block(U_1_2_3, V_1_2_3, W_1_2_3, 0, 0, 0, 1, 2, 3)
    
    # Block 2: i=0, j=0,1, k=3,4,5,6 → M_{1,2,4} = M_{1,2,2} + M_{1,2,2}
    U2a, V2a, W2a = m_1_2_2()
    U2b, V2b, W2b = m_1_2_2()
    U_1_2_4 = [row + row2 for row, row2 in zip(U2a, U2b)]
    V_1_2_4 = [row + row2 for row, row2 in zip(V2a, V2b)]
    W_1_2_4 = [row + row2 for row, row2 in zip(W2a, W2b)]
    add_block(U_1_2_4, V_1_2_4, W_1_2_4, 0, 0, 3, 1, 2, 4)
    
    # Block 3: i=1,2, j=0,1, k=0,1,2 → M_{2,2,3} = M_{2,2,2} + M_{2,2,1}
    U3a, V3a, W3a = strassen_222()
    U3b, V3b, W3b = m_2_2_1()
    U_2_2_3 = [row + row2 for row, row2 in zip(U3a, U3b)]
    V_2_2_3 = [row + row2 for row, row2 in zip(V3a, V3b)]
    W_2_2_3 = [row + row2 for row, row2 in zip(W3a, W3b)]
    add_block(U_2_2_3, V_2_2_3, W_2_2_3, 1, 0, 0, 2, 2, 3)
    
    # Block 4: i=1,2, j=0,1, k=3,4,5,6 → M_{2,2,4} = M_{2,2,2} + M_{2,2,2}
    U4a, V4a, W4a = strassen_222()
    U4b, V4b, W4b = strassen_222()
    U_2_2_4 = [row + row2 for row, row2 in zip(U4a, U4b)]
    V_2_2_4 = [row + row2 for row, row2 in zip(V4a, V4b)]
    W_2_2_4 = [row + row2 for row, row2 in zip(W4a, W4b)]
    add_block(U_2_2_4, V_2_2_4, W_2_2_4, 1, 0, 3, 2, 2, 4)
    
    # Block 5: i=0, j=2,3, k=0,1,2 → M_{1,2,3}
    add_block(U_1_2_3, V_1_2_3, W_1_2_3, 0, 2, 0, 1, 2, 3)
    
    # Block 6: i=0, j=2,3, k=3,4,5,6 → M_{1,2,4}
    add_block(U_1_2_4, V_1_2_4, W_1_2_4, 0, 2, 3, 1, 2, 4)
    
    # Block 7: i=1,2, j=2,3, k=0,1,2 → M_{2,2,3}
    add_block(U_2_2_3, V_2_2_3, W_2_2_3, 1, 2, 0, 2, 2, 3)
    
    # Block 8: i=1,2, j=2,3, k=3,4,5,6 → M_{2,2,4}
    add_block(U_2_2_4, V_2_2_4, W_2_2_4, 1, 2, 3, 2, 2, 4)
    
    # Now U_parts, V_parts, W_parts should have all rows
    U = U_parts
    V = V_parts
    W = W_parts
    
    rank = len(U[0]) if U else 0
    
    # Verify
    def verify(U, V, W):
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if s != 1:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m + j][col] = 1
                    V_school[j*p + k][col] = 1
                    W_school[k*n + i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
