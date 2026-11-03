import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # We will build the decomposition block by block.
    # Total rank R = 21 + 33 = 54
    
    U = []
    V = []
    W = []
    
    current_rank = 0
    
    # Helper to add a block decomposition
    # block_n, block_m, block_p
    # offset_U, offset_V, offset_W in the main matrices
    # The block multiplication is C_block = A_block * B_block
    # A_block is block_n x block_m
    # B_block is block_m x block_p
    # C_block is block_n x block_p
    
    # We need to map the local indices to global indices.
    # Global U row: i * m + j
    # Global V row: j * p + k
    # Global W row: k * n + i
    
    # Part 1: A1 (2x2) * B1 (2x6)
    # A1 is rows 0,1 of A (j=0,1)
    # B1 is rows 0,1 of B (j=0,1)
    # B1 is split into 3 blocks of 2x2.
    # Block 1: B1[:, 0:2] (k=0,1)
    # Block 2: B1[:, 2:4] (k=2,3)
    # Block 3: B1[:, 4:6] (k=4,5)
    
    for block_idx in range(3):
        k_start = block_idx * 2
        # Local k indices are 0, 1
        # Global k indices are k_start, k_start+1
        
        # Use Strassen 2x2x2
        # Rank 7
        # Strassen terms:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to add these to U, V, W.
        # U rows for A1: i=0,1; j=0,1. Global rows: 0, 1, 2, 3.
        # V rows for B1 block: j=0,1; k=k_start, k_start+1. Global rows: k_start, k_start+1, k_start+2, k_start+3?
        # Wait, V row index is j*p + k.
        # j=0: 0*p + k = k.
        # j=1: 1*p + k = p+k.
        # So for j=0,1 and k in {k_start, k_start+1}, V rows are:
        # k_start, k_start+1, p+k_start, p+k_start+1.
        
        # W rows for C1 block: i=0,1; k=k_start, k_start+1.
        # W row index is k*n + i.
        # k=k_start: k_start*2 + 0, k_start*2 + 1.
        # k=k_start+1: (k_start+1)*2 + 0, (k_start+1)*2 + 1.
        
        # We will add 7 columns to U, V, W.
        
        for t in range(7):
            u_col = [0] * (n*m)
            v_col = [0] * (m*p)
            w_col = [0] * (p*n)
            
            # Define the coefficients for this Strassen term
            # A indices: (i, j)
            # B indices: (j, k)
            
            if t == 0: # M1
                # A: (0,0)=1, (1,1)=1
                u_col[0*m+0] = 1
                u_col[1*m+1] = 1
                # B: (0,0)=1, (1,1)=1
                v_col[0*p+k_start] = 1
                v_col[1*p+k_start+1] = 1
                # W: C00 += 1, C11 += 1
                w_col[k_start*n+0] = 1
                w_col[(k_start+1)*n+1] = 1
            elif t == 1: # M2
                # A: (1,0)=1, (1,1)=1
                u_col[1*m+0] = 1
                u_col[1*m+1] = 1
                # B: (0,0)=1
                v_col[0*p+k_start] = 1
                # W: C10 += 1
                w_col[k_start*n+1] = 1
            elif t == 2: # M3
                # A: (0,0)=1
                u_col[0*m+0] = 1
                # B: (0,1)=1, (1,1)=-1
                v_col[0*p+k_start+1] = 1
                v_col[1*p+k_start+1] = -1
                # W: C01 += 1, C11 += 1
                w_col[(k_start+1)*n+0] = 1
                w_col[(k_start+1)*n+1] = 1
            elif t == 3: # M4
                # A: (1,1)=1
                u_col[1*m+1] = 1
                # B: (1,0)=1, (0,0)=-1
                v_col[1*p+k_start] = 1
                v_col[0*p+k_start] = -1
                # W: C00 += 1, C10 += 1
                w_col[k_start*n+0] = 1
                w_col[k_start*n+1] = 1
            elif t == 4: # M5
                # A: (0,0)=1, (0,1)=1
                u_col[0*m+0] = 1
                u_col[0*m+1] = 1
                # B: (1,1)=1
                v_col[1*p+k_start+1] = 1
                # W: C00 -= 1, C01 += 1
                w_col[k_start*n+0] = -1
                w_col[(k_start+1)*n+0] = 1
            elif t == 5: # M6
                # A: (1,0)=1, (0,0)=-1
                u_col[1*m+0] = 1
                u_col[0*m+0] = -1
                # B: (0,0)=1, (0,1)=1
                v_col[0*p+k_start] = 1
                v_col[0*p+k_start+1] = 1
                # W: C11 += 1
                w_col[(k_start+1)*n+1] = 1
            elif t == 6: # M7
                # A: (0,1)=1, (1,1)=-1
                u_col[0*m+1] = 1
                u_col[1*m+1] = -1
                # B: (1,0)=1, (1,1)=1
                v_col[1*p+k_start] = 1
                v_col[1*p+k_start+1] = 1
                # W: C00 += 1, C11 -= 1
                w_col[k_start*n+0] = 1
                w_col[(k_start+1)*n+1] = -1
            
            U.append(u_col)
            V.append(v_col)
            W.append(w_col)
            current_rank += 1

    # Part 2: A2 (2x3) * B2 (3x6)
    # A2 is rows 2,3,4 of A (j=2,3,4)
    # B2 is rows 2,3,4 of B (j=2,3,4)
    # B2 is split into 3 blocks of 3x2.
    # Block 1: B2[:, 0:2] (k=0,1)
    # Block 2: B2[:, 2:4] (k=2,3)
    # Block 3: B2[:, 4:6] (k=4,5)
    
    for block_idx in range(3):
        k_start = block_idx * 2
        
        # Use 2x3x2 decomposition (Rank 11)
        # Split A2 (2x3) into A2a (2x2) and A2b (2x1)
        # A2a uses j=2,3. A2b uses j=4.
        # Split B2 block (3x2) into B2a (2x2) and B2b (1x2)
        # B2a uses j=2,3. B2b uses j=4.
        
        # Term 1: A2a * B2a (2x2x2, Rank 7)
        # Term 2: A2b * B2b (2x1x2, Rank 4)
        
        # Term 1: Strassen on j=2,3
        # Local j indices 0,1 correspond to global j=2,3.
        # Local k indices 0,1 correspond to global k=k_start, k_start+1.
        
        for t in range(7):
            u_col = [0] * (n*m)
            v_col = [0] * (m*p)
            w_col = [0] * (p*n)
            
            if t == 0: # M1
                u_col[0*m+2] = 1
                u_col[1*m+3] = 1
                v_col[2*p+k_start] = 1
                v_col[3*p+k_start+1] = 1
                w_col[k_start*n+0] = 1
                w_col[(k_start+1)*n+1] = 1
            elif t == 1: # M2
                u_col[1*m+2] = 1
                u_col[1*m+3] = 1
                v_col[2*p+k_start] = 1
                w_col[k_start*n+1] = 1
            elif t == 2: # M3
                u_col[0*m+2] = 1
                v_col[2*p+k_start+1] = 1
                v_col[3*p+k_start+1] = -1
                w_col[(k_start+1)*n+0] = 1
                w_col[(k_start+1)*n+1] = 1
            elif t == 3: # M4
                u_col[1*m+3] = 1
                v_col[3*p+k_start] = 1
                v_col[2*p+k_start] = -1
                w_col[k_start*n+0] = 1
                w_col[k_start*n+1] = 1
            elif t == 4: # M5
                u_col[0*m+2] = 1
                u_col[0*m+3] = 1
                v_col[3*p+k_start+1] = 1
                w_col[k_start*n+0] = -1
                w_col[(k_start+1)*n+0] = 1
            elif t == 5: # M6
                u_col[1*m+2] = 1
                u_col[0*m+2] = -1
                v_col[2*p+k_start] = 1
                v_col[2*p+k_start+1] = 1
                w_col[(k_start+1)*n+1] = 1
            elif t == 6: # M7
                u_col[0*m+3] = 1
                u_col[1*m+3] = -1
                v_col[3*p+k_start] = 1
                v_col[3*p+k_start+1] = 1
                w_col[k_start*n+0] = 1
                w_col[(k_start+1)*n+1] = -1
            
            U.append(u_col)
            V.append(v_col)
            W.append(w_col)
            current_rank += 1
            
        # Term 2: A2b * B2b (2x1x2, Rank 4)
        # A2b is j=4. B2b is j=4.
        # C = A2b * B2b.
        # C[i, k] = sum_j A[i,j] B[j,k] = A[i,4] B[4,k].
        # This is outer product of column 4 of A and row 4 of B.
        # Rank 4 terms:
        # T1: A[0,4] * B[4, k_start] -> C[0, k_start]
        # T2: A[0,4] * B[4, k_start+1] -> C[0, k_start+1]
        # T3: A[1,4] * B[4, k_start] -> C[1, k_start]
        # T4: A[1,4] * B[4, k_start+1] -> C[1, k_start+1]
        
        for t in range(4):
            u_col = [0] * (n*m)
            v_col = [0] * (m*p)
            w_col = [0] * (p*n)
            
            if t == 0:
                u_col[0*m+4] = 1
                v_col[4*p+k_start] = 1
                w_col[k_start*n+0] = 1
            elif t == 1:
                u_col[0*m+4] = 1
                v_col[4*p+k_start+1] = 1
                w_col[(k_start+1)*n+0] = 1
            elif t == 2:
                u_col[1*m+4] = 1
                v_col[4*p+k_start] = 1
                w_col[k_start*n+1] = 1
            elif t == 3:
                u_col[1*m+4] = 1
                v_col[4*p+k_start+1] = 1
                w_col[(k_start+1)*n+1] = 1
                
            U.append(u_col)
            V.append(v_col)
            W.append(w_col)
            current_rank += 1
            
    if time.time() > deadline:
        # Fallback, though we should be well within time
        pass

    return {
        "U": U,
        "V": V,
        "W": W
    }

