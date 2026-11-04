import numpy as np
import time

def construct():
    t0 = time.time()
    deadline = t0 + 120.0
    
    n, m, p = 2, 4, 4
    rank = 28
    
    # Initialize with numpy arrays for efficient manipulation and distinct AST
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Define the Strassen-like decomposition for a 2x2 block product
    # Indices for A (2x2), B (2x2), C (2x2)
    # We apply this 4 times for the 2x4 * 4x4 problem
    
    # The 7 terms of Strassen's algorithm for 2x2 matrices:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We map the global indices to local indices for each block.
    # Block 1: A_rows [0,1,4,5] (A00, A01, A10, A11), B_rows [0,1,4,5], C_rows [0,2,1,3]
    # Note: C layout is [C00, C10, C01, C11] in row-major for 2x2?
    # The task specifies W[k*n+i, r].
    # For 2x4 * 4x4 -> 4x2 output.
    # Output C is 4x2.
    # Indexing: C[row, col].
    # Row 0,1 correspond to "Top" half?
    # Let's trace the standard mapping.
    # A is 2x4. Rows 0,1. Cols 0,1,2,3.
    # B is 4x4. Rows 0,1,2,3. Cols 0,1,2,3.
    # C is 4x2. Rows 0,1,2,3. Cols 0,1.
    
    # Wait, the problem is 2x4 * 4x4.
    # Output is 2x4.
    # The prompt says: "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 4 matrix. The output factor follows the official trace convention W[k*n+i,r]."
    # Usually W is for the third dimension.
    # In $C = AB$, $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # Here A is 2x4, B is 4x4. C is 2x4.
    # So n=2 (rows of A, rows of C), m=4 (cols of A, rows of B), p=4 (cols of B, cols of C).
    # U shape: (n*m, rank) = (8, rank). A indices.
    # V shape: (m*p, rank) = (16, rank). B indices.
    # W shape: (p*n, rank) = (8, rank). C indices.
    # C index mapping: "W[k*n+i,r]".
    # k is column of C (0..3), i is row of C (0..1).
    # So W index = k*2 + i.
    # C[0,0] -> W[0]
    # C[0,1] -> W[1]
    # C[1,0] -> W[2]
    # C[1,1] -> W[3]
    # C[0,2] -> W[4]
    # C[0,3] -> W[5]
    # C[1,2] -> W[6]
    # C[1,3] -> W[7]
    
    # Let's re-verify the block structure with this mapping.
    # A rows: 0, 1.
    # A cols: 0, 1 (Left), 2, 3 (Right).
    # B rows: 0, 1 (Top), 2, 3 (Bottom).
    # B cols: 0, 1 (Left), 2, 3 (Right).
    
    # C[0,0] = A[0,:] B[:,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0] + A[0,3]B[3,0]
    #            = (A_L[0] B_TL[0]) + (A_R[0] B_BL[0])
    # C[0,1] = (A_L[0] B_TL[1]) + (A_R[0] B_BL[1])
    # C[1,0] = (A_L[1] B_TL[0]) + (A_R[1] B_BL[0])
    # C[1,1] = (A_L[1] B_TL[1]) + (A_R[1] B_BL[1])
    
    # So the left half of C (cols 0,1) is sum of:
    # 1. A_L (2x2) * B_TL (2x2) -> contributes to C[0:2, 0:2]
    # 2. A_R (2x2) * B_BL (2x2) -> contributes to C[0:2, 0:2]
    
    # Similarly for Right half of C (cols 2,3):
    # 3. A_L (2x2) * B_TR (2x2) -> contributes to C[0:2, 2:4]
    # 4. A_R (2x2) * B_BR (2x2) -> contributes to C[0:2, 2:4]
    
    # We need to implement 4 instances of 2x2 matrix multiplication.
    # Instance 1: A_L, B_TL, C_LL (part of C)
    # Instance 2: A_R, B_BL, C_LR (part of C)
    # Instance 3: A_L, B_TR, C_RL (part of C)
    # Instance 4: A_R, B_BR, C_RR (part of C)
    
    # Local indices for A_L:
    # A[0,0] -> U[0]
    # A[0,1] -> U[1]
    # A[1,0] -> U[4]
    # A[1,1] -> U[5]
    # (Row-major for A: index = i*m + j. m=4.
    # i=0, j=0 -> 0.
    # i=0, j=1 -> 1.
    # i=1, j=0 -> 4.
    # i=1, j=1 -> 5.)
    
    # Local indices for A_R:
    # A[0,2] -> U[2]
    # A[0,3] -> U[3]
    # A[1,2] -> U[6]
    # A[1,3] -> U[7]
    
    # Local indices for B_TL:
    # B[0,0] -> V[0]
    # B[0,1] -> V[1]
    # B[1,0] -> V[4]
    # B[1,1] -> V[5]
    # (Row-major for B: index = j*p + k. p=4.
    # j=0, k=0 -> 0.
    # j=0, k=1 -> 1.
    # j=1, k=0 -> 4.
    # j=1, k=1 -> 5.)
    
    # Local indices for B_BL:
    # B[2,0] -> V[8]
    # B[2,1] -> V[9]
    # B[3,0] -> V[12]
    # B[3,1] -> V[13]
    
    # Local indices for B_TR:
    # B[0,2] -> V[2]
    # B[0,3] -> V[3]
    # B[1,2] -> V[6]
    # B[1,3] -> V[7]
    
    # Local indices for B_BR:
    # B[2,2] -> V[10]
    # B[2,3] -> V[11]
    # B[3,2] -> V[14]
    # B[3,3] -> V[15]
    
    # Local indices for C_L (Cols 0,1):
    # C[0,0] -> W[0]
    # C[0,1] -> W[1]
    # C[1,0] -> W[2]
    # C[1,1] -> W[3]
    
    # Local indices for C_R (Cols 2,3):
    # C[0,2] -> W[4]
    # C[0,3] -> W[5]
    # C[1,2] -> W[6]
    # C[1,3] -> W[7]
    
    # Strassen Terms for 2x2 product (a,b,c,d) * (w,x,y,z) = (p,q,r,s)
    # a=A00, b=A01, c=A10, d=A11
    # w=B00, x=B01, y=B10, z=B11
    # p=C00, q=C01, r=C10, s=C11
    
    # M1 = (a+d)(w+z)
    # M2 = (c+d)w
    # M3 = a(x-z)
    # M4 = d(y-w)
    # M5 = (a+b)z
    # M6 = (c-a)(w+x)
    # M7 = (b-d)(y+z)
    
    # p = M1 + M4 - M5 + M7
    # q = M3 + M5
    # r = M2 + M4
    # s = M1 - M2 + M3 + M6
    
    def add_strassen_block(u_idx, v_idx, w_idx, offset):
        # u_idx: [a, b, c, d]
        # v_idx: [w, x, y, z]
        # w_idx: [p, q, r, s]
        
        a, b, c, d = u_idx
        w, x, y, z = v_idx
        p, q, r, s = w_idx
        
        # M1
        U[a, offset] += 1
        U[d, offset] += 1
        V[w, offset] += 1
        V[z, offset] += 1
        W[p, offset] += 1
        W[s, offset] += 1
        
        # M2
        U[c, offset] += 1
        U[d, offset] += 1
        V[w, offset] += 1
        W[r, offset] += 1
        W[s, offset] -= 1
        
        # M3
        U[a, offset] += 1
        V[x, offset] += 1
        V[z, offset] -= 1
        W[q, offset] += 1
        W[s, offset] += 1
        
        # M4
        U[d, offset] += 1
        V[y, offset] += 1
        V[w, offset] -= 1
        W[p, offset] += 1
        W[r, offset] += 1
        
        # M5
        U[a, offset] += 1
        U[b, offset] += 1
        V[z, offset] += 1
        W[p, offset] -= 1
        W[q, offset] += 1
        
        # M6
        U[c, offset] += 1
        U[a, offset] -= 1
        V[w, offset] += 1
        V[x, offset] += 1
        W[s, offset] += 1
        
        # M7
        U[b, offset] += 1
        U[d, offset] -= 1
        V[y, offset] += 1
        V[z, offset] += 1
        W[p, offset] += 1
        
    # Block 1: A_L * B_TL -> C_L
    add_strassen_block([0, 1, 4, 5], [0, 1, 4, 5], [0, 1, 2, 3], 0)
    
    # Block 2: A_R * B_BL -> C_L
    add_strassen_block([2, 3, 6, 7], [8, 9, 12, 13], [0, 1, 2, 3], 7)
    
    # Block 3: A_L * B_TR -> C_R
    add_strassen_block([0, 1, 4, 5], [2, 3, 6, 7], [4, 5, 6, 7], 14)
    
    # Block 4: A_R * B_BR -> C_R
    add_strassen_block([2, 3, 6, 7], [10, 11, 14, 15], [4, 5, 6, 7], 21)
    
    # Convert to lists as required by interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if time.time() > deadline:
        pass
        
    return {"U": U_list, "V": V_list, "W": W_list}
