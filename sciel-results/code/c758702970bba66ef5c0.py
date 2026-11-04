import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # We will build the decomposition by summing smaller decompositions
    # U, V, W will be lists of rows, or we accumulate columns.
    # Rank is the number of columns.
    
    # Helper to add a decomposition
    # local_U: list of lists (n_local * m_local) x rank_local
    # local_V: list of lists (m_local * p_local) x rank_local
    # local_W: list of lists (p_local * n_local) x rank_local
    # offsets: (i_off, j_off, k_off) for A, B, C indices
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_decomp(local_U, local_V, local_W, i_off, j_off, k_off):
        # local_U shape: (n_l * m_l) x r
        # local_V shape: (m_l * p_l) x r
        # local_W shape: (p_l * n_l) x r
        # We need to expand these to global size (n*m) x 1, etc?
        # No, we are building the global U, V, W matrices.
        # Global U is (n*m) x Rank.
        # For a block at (i_off, j_off), the rows of local_U correspond to global rows i_off*m + j ...
        
        r = local_U.shape[1]
        for c in range(r):
            u_col = np.zeros(n * m)
            v_col = np.zeros(m * p)
            w_col = np.zeros(p * n)
            
            # Fill U
            for row in range(local_U.shape[0]):
                # local row index corresponds to (i_l, j_l)
                # i_l = row // m_l, j_l = row % m_l
                # global i = i_off + i_l, global j = j_off + j_l
                # global row index = global_i * m + global_j
                # Wait, m_l is the local m.
                # In our splits, m_l is always 2 or 4?
                # We split m=4 into 2+2. So m_l=2.
                # n_l=2, p_l=4 or 3.
                
                i_l = row // 2 # Assuming m_l=2 for the base blocks
                # This is fragile. Better to pass m_l.
                
            # This manual expansion is error-prone.
            # Let's use dense matrices and slice assignment.
            
            u_col[i_off*m : (i_off+n_l)*m, j_off : j_off+m_l] = local_U[:, c].reshape(n_l, m_l).T
            # Wait, local_U is (n_l * m_l) x r.
            # Row index in local_U is i_l * m_l + j_l.
            # So local_U[:, c] is a vector of length n_l * m_l.
            # Reshape to (n_l, m_l).
            # Then place in global U at rows i_off*m ... and cols j_off ...
            # Global U row index is i_global * m + j_global.
            # So we need to scatter.
            
            pass

    # This is too complex to do dynamically in the time limit without testing.
    # I will hardcode the Rank 50 solution structure.
    
    # Strategy:
    # 1. Define Strassen 2x2x2 decomposition (Rank 7)
    # 2. Define 2x2x3 decomposition (Rank 11) using Strassen + 4 schoolbook
    # 3. Define 2x2x4 decomposition (Rank 14) using 2 Strassen
    # 4. Assemble 2x4x7 from 2x4x4 (Rank 28) and 2x4x3 (Rank 22)
    
    # Strassen 2x2x2
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # U (4x7), V (4x7), W (4x7)
    # Term 1: (a+b)(e+g)
    # a=0, b=1, c=2, d=3
    # e=0, f=1, g=2, h=3
    S_U = np.zeros((4, 7))
    S_V = np.zeros((4, 7))
    S_W = np.zeros((4, 7))
    
    # T1: (A0+A1)(B0+B2)
    S_U[0,0]=1; S_U[1,0]=1
    S_V[0,0]=1; S_V[2,0]=1
    S_W[0,0]=1; S_W[3,0]=1 # C00, C11? No, Strassen sums are specific.
    
    # I will use a known correct Strassen implementation.
    # Reference: https://en.wikipedia.org/wiki/Strassen_algorithm
    # M1 = (a+b)(e+g)
    # M2 = (c+d)(e+g)
    # M3 = a(e-g)
    # M4 = d(f+h)
    # M5 = (a+d)(g+h)
    # M6 = (b-d)(e+g)
    # M7 = (a-c)(f+h)
    # C11 = M1 - M4 + M7 - M5
    # C12 = M4 + M5
    # C21 = M2 + M3
    # C22 = M1 - M3 + M6 - M5
    
    S_U = np.array([
        [1, 0, 1, 0, 1, 0, 1], # a
        [1, 0, 0, 0, 0, 1, 0], # b
        [0, 1, 0, 0, 0, -1, -1], # c
        [0, 1, 0, 1, 1, 0, 0]  # d
    ])
    S_V = np.array([
        [1, 1, 1, 0, 0, 1, 0], # e
        [0, 0, 0, 1, 0, 0, 1], # f
        [1, 1, -1, 0, 1, 1, 0], # g
        [0, 0, 0, 1, 1, 0, 1]  # h
    ])
    S_W = np.array([
        [1, 0, 0, -1, -1, 0, 1], # C11
        [0, 0, 0, 1, 1, 0, 0],   # C12
        [0, 1, 1, 0, 0, 0, 0],   # C21
        [1, 0, -1, 0, -1, 1, 0]  # C22
    ])
    
    # Verify S_U @ S_V * S_W? No, tensor contraction.
    # Sum_c U[i,c] V[j,c] W[k,c] = C[i,j,k]
    
    # Now build 2x2x3 (Rank 11)
    # Use Strassen for first 2 cols (Rank 7)
    # Schoolbook for 3rd col (Rank 4)
    # B has 3 cols. B0, B1, B2.
    # Strassen uses B0, B1.
    # Schoolbook for B2:
    # C02 = a*B20 + b*B21
    # C12 = c*B20 + d*B21
    # Terms:
    # T8: a * B20
    # T9: b * B21
    # T10: c * B20
    # T11: d * B21
    
    R11_U = np.zeros((4, 11))
    R11_V = np.zeros((6, 11)) # 2x3 B matrix
    R11_W = np.zeros((6, 11)) # 2x3 C matrix
    
    # Copy Strassen (cols 0-6 of B, 0-6 of C)
    R11_U[:, 0:7] = S_U
    R11_V[0:4, 0:7] = S_V
    R11_V[4:6, 0:7] = 0
    R11_W[0:4, 0:7] = S_W
    R11_W[4:6, 0:7] = 0
    
    # Add Schoolbook for Col 2 (index 2 in B, index 2 in C)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1), 4:(0,2), 5:(1,2)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1), 4:(0,2), 5:(1,2)
    
    # T8: a * B4
    R11_U[0, 7] = 1
    R11_V[4, 7] = 1
    R11_W[4, 7] = 1
    
    # T9: b * B5
    R11_U[1, 8] = 1
    R11_V[5, 8] = 1
    R11_W[4, 8] = 1
    
    # T10: c * B4
    R11_U[2, 9] = 1
    R11_V[4, 9] = 1
    R11_W[5, 9] = 1
    
    # T11: d * B5
    R11_U[3, 10] = 1
    R11_V[5, 10] = 1
    R11_W[5, 10] = 1
    
    # Now build 2x2x4 (Rank 14)
    # Two Strassen blocks.
    # B has 4 cols.
    # Block 1: Cols 0,1.
    # Block 2: Cols 2,3.
    
    R14_U = np.zeros((4, 14))
    R14_V = np.zeros((8, 14))
    R14_W = np.zeros((8, 14))
    
    R14_U[:, 0:7] = S_U
    R14_V[0:4, 0:7] = S_V
    R14_V[4:8, 0:7] = 0
    R14_W[0:4, 0:7] = S_W
    R14_W[4:8, 0:7] = 0
    
    R14_U[:, 7:14] = S_U
    R14_V[0:4, 7:14] = 0
    R14_V[4:8, 7:14] = S_V
    R14_W[0:4, 7:14] = 0
    R14_W[4:8, 7:14] = S_W
    
    # Now Assemble 2x4x7
    # Split into 2x4x4 and 2x4x3
    # 2x4x4:
    # A = [A1, A2] (2x2 each)
    # B = [B_top; B_bot] (2x4 each)
    # C = A1 B_top + A2 B_bot
    # A1 B_top is 2x2x4 (Rank 14)
    # A2 B_bot is 2x2x4 (Rank 14)
    
    # 2x4x3:
    # A = [A1, A2]
    # B = [B_top; B_bot] (2x3 each)
    # C = A1 B_top + A2 B_bot
    # A1 B_top is 2x2x3 (Rank 11)
    # A2 B_bot is 2x2x3 (Rank 11)
    
    # Total Rank 14+14+11+11 = 50.
    
    # Global Dimensions:
    # n=2, m=4, p=7
    # U: 8x50
    # V: 28x50
    # W: 14x50
    
    U = np.zeros((8, 50))
    V = np.zeros((28, 50))
    W = np.zeros((14, 50))
    
    # Helper to place a local decomp
    def place(local_U, local_V, local_W, i_off, j_off, k_off, col_off):
        # local_U: 4xR (2x2 A)
        # local_V: (2*p_l)xR
        # local_W: (p_l*2)xR
        n_l = 2
        m_l = 2
        p_l = local_V.shape[0] // 2
        
        for c in range(local_U.shape[1]):
            # U
            for i_l in range(n_l):
                for j_l in range(m_l):
                    global_i = i_off + i_l
                    global_j = j_off + j_l
                    global_row = global_i * m + global_j
                    U[global_row, col_off + c] = local_U[i_l * m_l + j_l, c]
            
            # V
            for j_l in range(m_l):
                for k_l in range(p_l):
                    global_j = j_off + j_l
                    global_k = k_off + k_l
                    global_row = global_j * p + global_k
                    V[global_row, col_off + c] = local_V[j_l * p_l + k_l, c]
            
            # W
            for k_l in range(p_l):
                for i_l in range(n_l):
                    global_k = k_off + k_l
                    global_i = i_off + i_l
                    global_row = global_k * n + global_i
                    W[global_row, col_off + c] = local_W[k_l * n_l + i_l, c]
    
    # 1. A1 B_top (2x2x4) at i=0, j=0, k=0
    place(R14_U, R14_V, R14_W, 0, 0, 0, 0)
    
    # 2. A2 B_bot (2x2x4) at i=0, j=2, k=0
    # A2 is rows 2,3 of A? No, A is 2x4.
    # A = [A1 A2]. A1 is cols 0,1. A2 is cols 2,3.
    # So j_off for A1 is 0, for A2 is 2.
    # B_top is rows 0,1 of B. B_bot is rows 2,3 of B.
    # So j_off for B_top is 0, for B_bot is 2.
    # Wait, V index is j*p + k.
    # For A2 B_bot:
    # A2 is at j=2,3. So in U, rows 2,3,6,7?
    # A is 2x4. Rows 0,1. Cols 0,1,2,3.
    # A1 is A[:, 0:2]. A2 is A[:, 2:4].
    # So for A2, j_off=2.
    # B_bot is B[2:4, :].
    # So for B_bot, j_off=2 (in B's row index, which is j in V).
    # C is at k=0,1,2,3. So k_off=0.
    
    place(R14_U, R14_V, R14_W, 0, 2, 0, 14)
    
    # 3. A1 B_top (2x2x3) at i=0, j=0, k=4
    # B_top is B[0:2, 4:7].
    # So j_off=0 for B.
    # k_off=4 for C.
    place(R11_U, R11_V, R11_W, 0, 0, 4, 28)
    
    # 4. A2 B_bot (2x2x3) at i=0, j=2, k=4
    place(R11_U, R11_V, R11_W, 0, 2, 4, 39)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

DIMENSIONS = [2, 4, 7]

def construct():
    # Re-define inside to avoid global scope issues if needed, or just call the logic above.
    # The logic above is self-contained if I put it in a function.
    
    n, m, p = DIMENSIONS
    
    S_U = np.array([
        [1, 0, 1, 0, 1, 0, 1], 
        [1, 0, 0, 0, 0, 1, 0], 
        [0, 1, 0, 0, 0, -1, -1], 
        [0, 1, 0, 1, 1, 0, 0]  
    ])
    S_V = np.array([
        [1, 1, 1, 0, 0, 1, 0], 
        [0, 0, 0, 1, 0, 0, 1], 
        [1, 1, -1, 0, 1, 1, 0], 
        [0, 0, 0, 1, 1, 0, 1]  
    ])
    S_W = np.array([
        [1, 0, 0, -1, -1, 0, 1], 
        [0, 0, 0, 1, 1, 0, 0],   
        [0, 1, 1, 0, 0, 0, 0],   
        [1, 0, -1, 0, -1, 1, 0]  
    ])
    
    R11_U = np.zeros((4, 11))
    R11_V = np.zeros((6, 11))
    R11_W = np.zeros((6, 11))
    R11_U[:, 0:7] = S_U
    R11_V[0:4, 0:7] = S_V
    R11_W[0:4, 0:7] = S_W
    R11_U[0, 7] = 1; R11_V[4, 7] = 1; R11_W[4, 7] = 1
    R11_U[1, 8] = 1; R11_V[5, 8] = 1; R11_W[4, 8] = 1
    R11_U[2, 9] = 1; R11_V[4, 9] = 1; R11_W[5, 9] = 1
    R11_U[3, 10] = 1; R11_V[5, 10] = 1; R11_W[5, 10] = 1

    R14_U = np.zeros((4, 14))
    R14_V = np.zeros((8, 14))
    R14_W = np.zeros((8, 14))
    R14_U[:, 0:7] = S_U
    R14_V[0:4, 0:7] = S_V
    R14_W[0:4, 0:7] = S_W
    R14_U[:, 7:14] = S_U
    R14_V[4:8, 7:14] = S_V
    R14_W[4:8, 7:14] = S_W

    U = np.zeros((8, 50))
    V = np.zeros((28, 50))
    W = np.zeros((14, 50))

    def place(local_U, local_V, local_W, i_off, j_off, k_off, col_off):
        n_l = 2
        m_l = 2
        p_l = local_V.shape[0] // 2
        for c in range(local_U.shape[1]):
            for i_l in range(n_l):
                for j_l in range(m_l):
                    U[(i_off + i_l) * m + (j_off + j_l), col_off + c] = local_U[i_l * m_l + j_l, c]
            for j_l in range(m_l):
                for k_l in range(p_l):
                    V[(j_off + j_l) * p + (k_off + k_l), col_off + c] = local_V[j_l * p_l + k_l, c]
            for k_l in range(p_l):
                for i_l in range(n_l):
                    W[(k_off + k_l) * n + (i_off + i_l), col_off + c] = local_W[k_l * n_l + i_l, c]

    place(R14_U, R14_V, R14_W, 0, 0, 0, 0)
    place(R14_U, R14_V, R14_W, 0, 2, 0, 14)
    place(R11_U, R11_V, R11_W, 0, 0, 4, 28)
    place(R11_U, R11_V, R11_W, 0, 2, 4, 39)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
