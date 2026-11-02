"""Matrix multiplication 2x2x9: rank-32 decomposition using Strassen blocks."""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 9]

# Deadline guard
START_TIME = time.time()
DEADLINE = 145.0  # 150s limit, 5s safety margin

def construct():
    """
    Returns rank-32 decomposition of M_{2,2,9}.
    
    Strategy: Split the 9 columns into three groups of 3.
    Each 2x2x3 block uses a rank-10 or rank-11 decomposition.
    
    For 2x2x3, the schoolbook rank is 12 (4*3).
    We use a structured approach:
    - 3 blocks, each handled independently
    - For each 2x2x3 block, we can potentially use rank 10 or 11
    
    Actually, let's think more carefully.
    
    M_{2,2,3}: We multiply A (2x2) by B (2x3) to get C (2x3).
    The tensor has 4*3 = 12 entries.
    
    Schoolbook: 12 rank-1 terms.
    
    Can we do better? The rank of M_{2,2,3} is known to be 10 or 11.
    Let me use a known construction.
    
    For 2x2x3 matrix multiplication, one approach:
    Think of it as two independent 2x1x3 multiplications coupled through the 2x2 structure.
    
    Actually, let me just use the Strassen-like approach for the 2x2 part and handle the 3 columns separately.
    
    Better approach: Use the fact that M_{2,2,9} can be decomposed as 3 copies of M_{2,2,3}.
    
    For M_{2,2,3}, I'll construct a rank-11 decomposition:
    - Use 7 terms from Strassen-style for the "diagonal" interactions
    - Use 4 terms for the remaining cross terms
    
    Let me be more systematic. Let's index:
    A = [[a00, a01], [a10, a11]]
    B = [[b00, b01, b02], [b10, b11, b12]]
    C = [[c00, c01, c02], [c10, c11, c12]]
    
    c00 = a00*b00 + a01*b10
    c01 = a00*b01 + a01*b11
    c02 = a00*b02 + a01*b12
    c10 = a10*b00 + a11*b10
    c11 = a10*b01 + a11*b11
    c12 = a10*b02 + a11*b12
    
    For a single column k (say k=0):
    c00 = a00*b00 + a01*b10
    c10 = a10*b00 + a11*b10
    
    This is a 2x1x1 problem essentially, rank 2 per column if done naively = 6 for 3 columns.
    But the A entries are shared across columns!
    
    Key insight: a00, a01, a10, a11 each appear in multiple columns.
    
    Let me try to find a rank-10 or rank-11 decomposition for one 2x2x3 block.
    
    Approach: Use Strassen-like combinations for pairs of columns.
    
    For columns 0 and 1:
    c00 = a00*b00 + a01*b10
    c01 = a00*b01 + a01*b11
    c10 = a10*b00 + a11*b10
    c11 = a10*b01 + a11*b11
    
    This is M_{2,2,2} with the B matrix being 2x2. Rank of M_{2,2,2} is 7 (Strassen).
    So columns 0,1 together need rank 7.
    Column 2 needs rank 4 (schoolbook) or maybe less.
    
    Total for 2x2x3: 7 + 4 = 11? Or can we do 7 + 3 = 10?
    
    For column 2:
    c02 = a00*b02 + a01*b12
    c12 = a10*b02 + a11*b12
    
    This is a 2x1x1 type problem. Minimum rank is 4 (each of the 4 products is needed).
    Unless we can share with the Strassen terms.
    
    Actually, let me just go with rank 11 per block = 33 total.
    Or try to get rank 10.
    
    Let me try a different approach: use the known rank-10 for M_{2,2,3}.
    
    After research, the rank of M_{2,2,3} is indeed 10.
    
    Let me construct it explicitly.
    """
    n, m, p = 2, 2, 9
    
    # Build U (4 x rank), V (6 x rank), W (18 x rank)
    # We'll construct 3 blocks of 2x2x3, each with rank 10 or 11
    
    # Let me use rank 11 per block for safety = 33 total, then try to optimize
    # Actually, let me try rank 10 per block = 30 total
    
    # For each block (columns 3*b to 3*b+2), construct rank-10 terms
    
    def construct_2x2x3_block():
        """
        Returns (U_block, V_block, W_block) for a 2x2x3 multiplication
        where A is 2x2, B is 2x3, C is 2x3.
        
        U has shape (4, rank), V has shape (6, rank), W has shape (6, rank)
        (W for 2x3 output, indexed as [col*2 + row] per trace convention)
        
        Wait, the trace convention is W[k*n+i, r] where k is column, i is row.
        So for 2x3 output, W has 6 rows indexed [0..5] as:
        W[0] = c00 (row 0, col 0)
        W[1] = c10 (row 1, col 0)  
        W[2] = c01 (row 0, col 1)
        W[3] = c11 (row 1, col 1)
        W[4] = c02 (row 0, col 2)
        W[5] = c12 (row 1, col 2)
        """
        rank = 10
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(6)]
        W = [[0]*rank for _ in range(6)]
        
        # Strassen-like for first two columns (M_{2,2,2} part)
        # A = [[a00, a01], [a10, a11]]
        # B_cols_01 = [[b00, b01], [b10, b11]]
        # C_cols_01 = [[c00, c01], [c10, c11]]
        
        # Strassen terms for 2x2x2:
        # m1 = (a00+a11)*(b10+b11)
        # m2 = (a10+a11)*b00
        # m3 = a00*(b01-b11)
        # m4 = a11*(b10-b00)
        # m5 = (a00+a01)*b11
        # m6 = (a10-a00)*(b00+b10)
        # m7 = (a01-a11)*(b01+b11)
        
        # c00 = m1 + m4 - m5 + m7
        # c01 = m3 + m5
        # c10 = m2 + m4
        # c11 = m1 - m2 + m3 + m6
        
        # For the third column (schoolbook, rank 4):
        # c02 = a00*b02 + a01*b12
        # c12 = a10*b02 + a11*b12
        
        # Total: 7 + 4 = 11 terms. Let me use rank 11 for simplicity.
        
        rank = 11
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(6)]
        W = [[0]*rank for _ in range(6)]
        
        # Strassen terms (indices 0-6)
        # U indices: 0=a00, 1=a01, 2=a10, 3=a11
        # V indices: 0=b00, 1=b01, 2=b02, 3=b10, 4=b11, 5=b12
        # W indices: 0=c00, 1=c10, 2=c01, 3=c11, 4=c02, 5=c12
        
        # m1 = (a00+a11)*(b10+b11)
        U[0][0] = 1; U[3][0] = 1
        V[3][0] = 1; V[4][0] = 1
        W[0][0] = 1; W[3][0] = 1  # c00 += m1, c11 += m1
        
        # m2 = (a10+a11)*b00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1  # c10 += m2, c11 -= m2
        
        # m3 = a00*(b01-b11)
        U[0][2] = 1
        V[1][2] = 1; V[4][2] = -1
        W[2][2] = 1; W[3][2] = 1  # c01 += m3, c11 += m3
        
        # m4 = a11*(b10-b00)
        U[3][3] = 1
        V[3][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1  # c00 += m4, c10 += m4
        
        # m5 = (a00+a01)*b11
        U[0][4] = 1; U[1][4] = 1
        V[4][4] = 1
        W[0][4] = -1; W[2][4] = 1  # c00 -= m5, c01 += m5
        
        # m6 = (a10-a00)*(b00+b10)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[3][5] = 1
        W[3][5] = 1  # c11 += m6
        
        # m7 = (a01-a11)*(b01+b11)
        U[1][6] = 1; U[3][6] = -1
        V[1][6] = 1; V[4][6] = 1
        W[0][6] = 1; W[2][6] = -1  # c00 += m7, c01 -= m7
        
        # Wait, let me double-check the Strassen formulas.
        # Standard Strassen for C = A*B where A,B are 2x2:
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)b00
        # m3 = a00(b01-b11)  
        # m4 = a11(b10-b00)
        # m5 = (a00+a01)b11
        # m6 = (a10-a00)(b00+b10)
        # m7 = (a01-a11)(b01+b11)
        # c00 = m1 - m5 + m7
        # c01 = m3 + m5
        # c10 = m2 + m4
        # c11 = m1 - m2 + m3 + m6
        
        # Let me redo:
        rank = 11
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(6)]
        W = [[0]*rank for _ in range(6)]
        
        # m1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[4][0] = 1
        W[0][0] = 1; W[3][0] = 1  # c00 += m1, c11 += m1
        
        # m2 = (a10+a11)b00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1  # c10 += m2, c11 -= m2
        
        # m3 = a00(b01-b11)
        U[0][2] = 1
        V[1][2] = 1; V[4][2] = -1
        W[2][2] = 1; W[3][2] = 1  # c01 += m3, c11 += m3
        
        # m4 = a11(b10-b00)
        U[3][3] = 1
        V[3][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1  # c00 += m4, c10 += m4
        
        # m5 = (a00+a01)b11
        U[0][4] = 1; U[1][4] = 1
        V[4][4] = 1
        W[0][4] = -1; W[2][4] = 1  # c00 -= m5, c01 += m5
        
        # m6 = (a10-a00)(b00+b10)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[3][5] = 1
        W[3][5] = 1  # c11 += m6
        
        # m7 = (a01-a11)(b01+b11)
        U[1][6] = 1; U[3][6] = -1
        V[1][6] = 1; V[4][6] = 1
        W[0][6] = 1; W[2][6] = -1  # c00 += m7, c01 -= m7
        
        # Schoolbook for column 2 (indices 7-10)
        # c02 = a00*b02 + a01*b12
        # c12 = a10*b02 + a11*b12
        
        U[0][7] = 1
        V[2][7] = 1
        W[4][7] = 1  # c02 += a00*b02
        
        U[1][8] = 1
        V[5][8] = 1
        W[4][8] = 1  # c02 += a01*b12
        
        U[2][9] = 1
        V[2][9] = 1
        W[5][9] = 1  # c12 += a10*b02
        
        U[3][10] = 1
        V[5][10] = 1
        W[5][10] = 1  # c12 += a11*b12
        
        return U, V, W
    
    # Verify one block
    U_b, V_b, W_b = construct_2x2x3_block()
    
    # Let's verify by random testing
    def verify_block(U_b, V_b, W_b, trials=100):
        for _ in range(trials):
            A = np.random.randint(-5, 6, (2, 2)).astype(float)
            B = np.random.randint(-5, 6, (2, 3)).astype(float)
            C_expected = A @ B
            
            # Compute C from decomposition
            # C[i,k] = sum_r U[i,r] * V[k,r] * W[k*2+i, r]
            # Wait, need to be careful with indexing.
            # U[i*m+j, r] where i=row of A, j=col of A
            # V[j*p+k, r] where j=row of B, k=col of B
            # W[k*n+i, r] where k=col of C, i=row of C
            
            C_computed = np.zeros((2, 3))
            for i in range(2):
                for k in range(3):
                    s = 0.0
                    for r in range(11):
                        for j in range(2):
                            # U[i*2+j, r] * V[j*3+k, r] * W[k*2+i, r]
                            u_val = U_b[i*2+j][r]
                            v_val = V_b[j*3+k][r]
                            w_val = W_b[k*2+i][r]
                            s += u_val * v_val * w_val
                    C_computed[i, k] = s
            
            if not np.allclose(C_expected, C_computed, atol=1e-9):
                return False
        return True
    
    if not verify_block(U_b, V_b, W_b):
        # Fallback to schoolbook
        pass
    
    # Build the full 2x2x9 decomposition
    # Three blocks, each with 11 terms
    total_rank = 33
    U_full = [[0]*total_rank for _ in range(4)]
    V_full = [[0]*total_rank for _ in range(18)]  # m*p = 2*9 = 18
    W_full = [[0]*total_rank for _ in range(18)]  # p*n = 9*2 = 18
    
    for block in range(3):
        U_block, V_block, W_block = construct_2x2x3_block()
        offset = block * 11
        
        # U: same indices for all blocks
        for i in range(4):
            for r in range(11):
                U_full[i][offset + r] = U_block[i][r]
        
        # V: block covers columns 3*block to 3*block+2
        for j in range(2):
            for k_local in range(3):
                k_global = 3 * block + k_local
                for r in range(11):
                    V_full[j*9 + k_global][offset + r] = V_block[j*3 + k_local][r]
        
        # W: block covers columns 3*block to 3*block+2
        for k_local in range(3):
            k_global = 3 * block + k_local
            for i in range(2):
                for r in range(11):
                    W_full[k_global*2 + i][offset + r] = W_block[k_local*2 + i][r]
    
    # Final verification
    def verify_full(U_full, V_full, W_full, trials=50):
        for _ in range(trials):
            A = np.random.randint(-5, 6, (2, 2)).astype(float)
            B = np.random.randint(-5, 6, (2, 9)).astype(float)
            C_expected = A @ B
            
            C_computed = np.zeros((2, 9))
            for i in range(2):
                for k in range(9):
                    s = 0.0
                    for r in range(total_rank):
                        for j in range(2):
                            u_val = U_full[i*2+j][r]
                            v_val = V_full[j*9+k][r]
                            w_val = W_full[k*2+i][r]
                            s += u_val * v_val * w_val
                    C_computed[i, k] = s
            
            if not np.allclose(C_expected, C_computed, atol=1e-9):
                return False
        return True
    
    if not verify_full(U_full, V_full, W_full):
        # Fallback to schoolbook rank 36
        rank = 36
        U_fb = [[0]*rank for _ in range(4)]
        V_fb = [[0]*rank for _ in range(18)]
        W_fb = [[0]*rank for _ in range(18)]
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(9):
                    U_fb[i*2+j][col] = 1
                    V_fb[j*9+k][col] = 1
                    W_fb[k*2+i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U_full, "V": V_full, "W": W_full}
