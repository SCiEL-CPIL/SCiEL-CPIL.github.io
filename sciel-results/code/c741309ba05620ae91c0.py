"""Matrix multiplication 3x4 * 4x5 with rank-56 decomposition using Strassen on 2x2 blocks."""
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 5  # A is n x m, B is m x p, C is n x p
    
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (15, rank)
    # W indexing: W[k*n + i, r] where k is column of C (0..p-1), i is row of C (0..n-1)
    
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        """Add a single rank-1 term."""
        nonlocal col
        U[u_idx, col] += val
        V[v_idx, col] += val
        W[w_idx, col] += val
        col += 1
    
    def add_1x2_2x5(a_row, a_col_start, b_row_start, b_col_start, c_row, c_col_start):
        """
        Compute 1x2 * 2x5 = 1x5 using 10 schoolbook multiplications.
        A element: A[a_row, a_col_start] and A[a_row, a_col_start+1]
        B elements: B[b_row_start, b_col_start..b_col_start+4] and B[b_row_start+1, b_col_start..b_col_start+4]
        C element: C[c_row, c_col_start..c_col_start+4]
        """
        nonlocal col
        for j in range(5):
            # Term 1: A[a_row, a_col_start] * B[b_row_start, b_col_start+j]
            u1 = a_row * m + (a_col_start)
            v1 = b_row_start * p + (b_col_start + j)
            w1 = (c_col_start + j) * n + c_row
            add_term(u1, v1, w1)
            
            # Term 2: A[a_row, a_col_start+1] * B[b_row_start+1, b_col_start+j]
            u2 = a_row * m + (a_col_start + 1)
            v2 = (b_row_start + 1) * p + (b_col_start + j)
            w2 = (c_col_start + j) * n + c_row
            add_term(u2, v2, w2)
    
    def add_strassen_2x2(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        """
        Compute 2x2 * 2x2 = 2x2 using Strassen's 7 multiplications.
        A: rows a_row_start..a_row_start+1, cols a_col_start..a_col_start+1
        B: rows b_row_start..b_row_start+1, cols b_col_start..b_col_start+1
        C: rows c_row_start..c_row_start+1, cols c_col_start..c_col_start+1
        """
        nonlocal col
        
        # A indices
        a00 = a_row_start * m + a_col_start
        a01 = a_row_start * m + (a_col_start + 1)
        a10 = (a_row_start + 1) * m + a_col_start
        a11 = (a_row_start + 1) * m + (a_col_start + 1)
        
        # B indices
        b00 = b_row_start * p + b_col_start
        b01 = b_row_start * p + (b_col_start + 1)
        b10 = (b_row_start + 1) * p + b_col_start
        b11 = (b_row_start + 1) * p + (b_col_start + 1)
        
        # C indices (W[k*n+i] where k=col, i=row)
        c00 = c_col_start * n + c_row_start
        c01 = (c_col_start + 1) * n + c_row_start
        c10 = c_col_start * n + (c_row_start + 1)
        c11 = (c_col_start + 1) * n + (c_row_start + 1)
        
        # P1 = (a00+a11)(b00+b11)
        # U: a00=1, a11=1; V: b00=1, b11=1
        U[a00, col] = 1
        U[a11, col] = 1
        V[b00, col] = 1
        V[b11, col] = 1
        W[c00, col] = 1
        W[c11, col] = 1
        col += 1
        
        # P2 = (a10+a11)b00
        # U: a10=1, a11=1; V: b00=1
        U[a10, col] = 1
        U[a11, col] = 1
        V[b00, col] = 1
        W[c10, col] = 1
        W[c11, col] = -1
        col += 1
        
        # P3 = a00(b01-b11)
        # U: a00=1; V: b01=1, b11=-1
        U[a00, col] = 1
        V[b01, col] = 1
        V[b11, col] = -1
        W[c01, col] = 1
        W[c11, col] = 1
        col += 1
        
        # P4 = a11(b10-b00)
        # U: a11=1; V: b10=1, b00=-1
        U[a11, col] = 1
        V[b10, col] = 1
        V[b00, col] = -1
        W[c00, col] = 1
        W[c10, col] = 1
        col += 1
        
        # P5 = (a00+a01)b11
        # U: a00=1, a01=1; V: b11=1
        U[a00, col] = 1
        U[a01, col] = 1
        V[b11, col] = 1
        W[c01, col] = 1
        W[c11, col] = -1
        col += 1
        
        # P6 = (a10-a00)(b00+b01)
        # U: a10=1, a00=-1; V: b00=1, b01=1
        U[a10, col] = 1
        U[a00, col] = -1
        V[b00, col] = 1
        V[b01, col] = 1
        W[c10, col] = 1
        W[c01, col] = -1
        col += 1
        
        # P7 = (a01-a11)(b10+b11)
        # U: a01=1, a11=-1; V: b10=1, b11=1
        U[a01, col] = 1
        U[a11, col] = -1
        V[b10, col] = 1
        V[b11, col] = 1
        W[c00, col] = -1
        W[c11, col] = 1
        col += 1
    
    def add_2x2_2x1(a_row_start, a_col_start, b_row_start, b_col, c_row_start, c_col):
        """
        Compute 2x2 * 2x1 = 2x1 using 4 schoolbook multiplications.
        A: rows a_row_start..a_row_start+1, cols a_col_start..a_col_start+1
        B: rows b_row_start..b_row_start+1, col b_col
        C: rows c_row_start..c_row_start+1, col c_col
        """
        nonlocal col
        
        a00 = a_row_start * m + a_col_start
        a01 = a_row_start * m + (a_col_start + 1)
        a10 = (a_row_start + 1) * m + a_col_start
        a11 = (a_row_start + 1) * m + (a_col_start + 1)
        
        b0 = b_row_start * p + b_col
        b1 = (b_row_start + 1) * p + b_col
        
        c0 = c_col * n + c_row_start
        c1 = c_col * n + (c_row_start + 1)
        
        # C[0] = a00*b0 + a01*b1
        U[a00, col] = 1
        V[b0, col] = 1
        W[c0, col] = 1
        col += 1
        
        U[a01, col] = 1
        V[b1, col] = 1
        W[c0, col] = 1
        col += 1
        
        # C[1] = a10*b0 + a11*b1
        U[a10, col] = 1
        V[b0, col] = 1
        W[c1, col] = 1
        col += 1
        
        U[a11, col] = 1
        V[b1, col] = 1
        W[c1, col] = 1
        col += 1
    
    # Split A (3x4) into A1 (3x2, cols 0-1) and A2 (3x2, cols 2-3)
    # Split B (4x5) into Btop (2x5, rows 0-1) and Bbot (2x5, rows 2-3)
    
    # C = A1 * Btop + A2 * Bbot
    
    # Process A1 (cols 0-1 of A) * Btop (rows 0-1 of B)
    # A1: rows 0-2, cols 0-1
    # Btop: rows 0-1, cols 0-4
    # C: rows 0-2, cols 0-4
    
    # A1[0,:]*Btop: 1x2 * 2x5 -> C[0,:]
    add_1x2_2x5(0, 0, 0, 0, 0, 0)  # 10 terms
    
    # A1[1:3,:]*Btop: 2x2 * 2x5 -> C[1:3,:]
    # Split Btop into Btop_left (2x2, cols 0-1), Btop_mid (2x2, cols 2-3), Btop_right (2x1, col 4)
    # A1[1:3,:]*Btop_left: 2x2 * 2x2 -> C[1:3, 0:2]
    add_strassen_2x2(1, 0, 0, 0, 1, 0)  # 7 terms
    # A1[1:3,:]*Btop_mid: 2x2 * 2x2 -> C[1:3, 2:4]
    add_strassen_2x2(1, 0, 0, 2, 1, 2)  # 7 terms
    # A1[1:3,:]*Btop_right: 2x2 * 2x1 -> C[1:3, 4]
    add_2x2_2x1(1, 0, 0, 4, 1, 4)  # 4 terms
    
    # Process A2 (cols 2-3 of A) * Bbot (rows 2-3 of B)
    # A2: rows 0-2, cols 2-3
    # Bbot: rows 2-3, cols 0-4
    # C: rows 0-2, cols 0-4
    
    # A2[0,:]*Bbot: 1x2 * 2x5 -> C[0,:]
    add_1x2_2x5(0, 2, 2, 0, 0, 0)  # 10 terms
    
    # A2[1:3,:]*Bbot: 2x2 * 2x5 -> C[1:3,:]
    # Split Bbot into Bbot_left (2x2, cols 0-1), Bbot_mid (2x2, cols 2-3), Bbot_right (2x1, col 4)
    # A2[1:3,:]*Bbot_left: 2x2 * 2x2 -> C[1:3, 0:2]
    add_strassen_2x2(1, 2, 2, 0, 1, 0)  # 7 terms
    # A2[1:3,:]*Bbot_mid: 2x2 * 2x2 -> C[1:3, 2:4]
    add_strassen_2x2(1, 2, 2, 2, 1, 2)  # 7 terms
    # A2[1:3,:]*Bbot_right: 2x2 * 2x1 -> C[1:3, 4]
    add_2x2_2x1(1, 2, 2, 4, 1, 4)  # 4 terms
    
    # Total: 2 * (10 + 7 + 7 + 4) = 2 * 28 = 56
    assert col == rank, f"Expected {rank}, got {col}"
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
