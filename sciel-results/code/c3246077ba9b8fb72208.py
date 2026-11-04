"""Program for matrix multiplication 3,4,4 tensor decomposition."""
DIMENSIONS = [3, 4, 4]

import time
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    """
    Low-rank decomposition for 3x4x4 matrix multiplication.
    
    Uses Strassen's rank-7 algorithm for 2x2x2 sub-blocks to achieve rank 44.
    
    Decomposition strategy:
    - Split A (3x4) = [A_top (2x4); A_bottom (1x4)]
    - A_top * B: (2x4)*(4x4) -> 2x4, decomposed via Strassen on 2x2x2 blocks
    - A_bottom * B: (1x4)*(4x4) -> 1x4, schoolbook rank 16
    
    For (2x4)*(4x4):
    - Split B = [B1 (4x2); B2 (4x2)]  (two column blocks)
    - C_left (2x2) = A(2x4) * B1(4x2): split A into two 2x2 column groups
      C_left = A[:,0:2]*(2x2) * B1[0:2,:](2x2) + A[:,2:4](2x2) * B1[2:4,:](2x2)
      Each (2x2)*(2x2) is rank 7 (Strassen), so 14 for C_left
    - Similarly C_right = 14
    - Total for 2x4x4: 28
    
    Total rank: 28 + 16 = 44
    """
    n, m, p = 3, 4, 4
    
    # Build Strassen 2x2x2 decomposition
    def strassen_2x2x2():
        """Rank-7 decomposition for (2x2)*(2x2) -> (2x2).
        U: 4 rows (A entries), V: 4 rows (B entries), W: 4 rows (C entries)
        C[k,i] uses W index k*2+i
        """
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        
        # M1 = (a00+a11)(b00+b11): C00 += M1, C11 -= M1
        U[0][0]=1; U[3][0]=1
        V[0][0]=1; V[3][0]=1
        W[0][0]=1; W[3][0]=-1
        
        # M2 = (a10+a11)(b00): C10 += M2, C11 += M2  
        # Actually: M2 = (a10+a11)*b00 -> C10 += M2, C11 += M2
        # Wait, let me use the standard formulation:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        #
        # C00 = M1 - M3 + M5 - M6
        # C01 = M4 + M5 - M7
        # C10 = M2 - M6 + M7  
        # C11 = M1 - M2 + M4 + M7
        
        # a00=U[0], a01=U[1], a10=U[2], a11=U[3]
        # b00=V[0], b01=V[1], b10=V[2], b11=V[3]
        # C00=W[0], C01=W[1], C10=W[2], C11=W[3]
        
        # M1: (a00+a11)(b00+b11)
        U[0][0]=1; U[3][0]=1
        V[0][0]=1; V[3][0]=1
        # C00 += 1, C11 += -1
        W[0][0]=1; W[3][0]=-1
        
        # M2: (a10+a11)b00
        U[2][1]=1; U[3][1]=1
        V[0][1]=1
        # C10 += 1, C11 += 1
        W[2][1]=1; W[3][1]=1
        
        # M3: a00(b01-b11)
        U[0][2]=1
        V[1][2]=1; V[3][2]=-1
        # C00 += -1, C01 += 1
        W[0][2]=-1; W[1][2]=1
        
        # M4: a11(b10-b00)
        U[3][3]=1
        V[2][3]=1; V[0][3]=-1
        # C01 += 1, C11 += 1
        W[1][3]=1; W[3][3]=1
        
        # M5: (a00+a01)b11
        U[0][4]=1; U[1][4]=1
        V[3][4]=1
        # C00 += 1, C01 += -1
        W[0][4]=1; W[1][4]=-1
        
        # M6: (a10-a00)(b00+b01)
        U[2][5]=1; U[0][5]=-1
        V[0][5]=1; V[1][5]=1
        # C00 += -1, C10 += -1
        W[0][5]=-1; W[2][5]=-1
        
        # M7: (a01-a11)(b10+b11)
        U[1][6]=1; U[3][6]=-1
        V[2][6]=1; V[3][6]=1
        # C01 += 1, C10 += 1
        W[1][6]=1; W[2][6]=1
        
        return U, V, W
    
    su, sv, sw = strassen_2x2x2()
    
    # Now build the full 3x4x4 decomposition
    # Total rank = 4*7 (four 2x2x2 Strassen blocks) + 16 (1x4x4 schoolbook) = 28+16 = 44
    
    total_rank = 44
    U = [[0]*total_rank for _ in range(12)]  # 3*4 = 12
    V = [[0]*total_rank for _ in range(16)]  # 4*4 = 16
    W = [[0]*total_rank for _ in range(12)]  # 4*3 = 12
    
    # Part 1: top 2 rows of A (rows 0,1 of 3x4)
    # A_top is 2x4, B is 4x4, result is 2x4
    # Split into left 2 cols and right 2 cols of result
    # C_left (2x2) = A_top[:,0:2] * B[0:2,:]
    #   = A_top[:,0](col0) * B[0,:] (row0 of B) + A_top[:,1](col1) * B[1,:] (row1 of B)
    # Hmm, this isn't cleanly 2x2x2.
    
    # Better approach for 2x4x4:
    # C (2x4) = A (2x4) * B (4x4)
    # C[:, 0:2] = A[:,0:2] * B[0:2, 0:2] + A[:,2:4] * B[2:4, 0:2]
    # C[:, 2:4] = A[:,0:2] * B[0:2, 2:4] + A[:,2:4] * B[2:4, 2:4]
    #
    # So we have four 2x2x2 multiplications:
    # 1. A[:,:2] * B[:2,:2] -> top-left 2x2 of C
    # 2. A[:,:2] * B[:2,2:4] -> top-right 2x2 of C  
    # 3. A[:,2:] * B[2:,:2] -> bottom-left... wait, C only has 2 rows.
    
    # For 2x4x4: A is 2x4, B is 4x4, C is 2x4.
    # C[:,0:2] = A[:,:2]*B[:2,0:2] + A[:,2:]*B[2:,0:2]
    # C[:,2:4] = A[:,:2]*B[:2,2:4] + A[:,2:]*B[2:,2:4]
    #
    # Four 2x2x2 blocks:
    # Block 1: A[:,:2] (2x2) * B[:2,0:2] (2x2) -> adds to C[:,:2]
    # Block 2: A[:,:2] (2x2) * B[:2,2:4] (2x2) -> adds to C[:,2:4]
    # Block 3: A[:,2:] (2x2) * B[2:,0:2] (2x2) -> adds to C[:,:2]
    # Block 4: A[:,2:] (2x2) * B[2:,2:4] (2x2) -> adds to C[:,2:4]
    
    # For the top 2 rows of A (global rows 0,1), col groups:
    # A rows 0,1; A cols 0,1 -> U indices 0,1 (i=0,j=0; i=0,j=1)
    # A rows 0,1; A cols 2,3 -> U indices 2,3 (i=0,j=2; i=0,j=3)
    # A rows 0,1; B rows 0,1 -> V indices 0,1 (j=0,k=0; j=0,k=1) for B[0,0],B[0,1]
    #   Wait, B is 4x4, V index = j*4+k
    # B[0,0]=V[0], B[0,1]=V[1], B[0,2]=V[2], B[0,3]=V[3]
    # B[1,0]=V[4], B[1,1]=V[5], B[1,2]=V[6], B[1,3]=V[7]
    # B[2,0]=V[8], B[2,1]=V[9], B[2,2]=V[10], B[2,3]=V[11]
    # B[3,0]=V[12], B[3,1]=V[13], B[3,2]=V[14], B[3,3]=V[15]
    
    # C for top 2 rows: C[0,0]=W[0*3+0]=W[0], C[0,1]=W[1*3+0]=W[3], C[0,2]=W[2*3+0]=W[6], C[0,3]=W[3*3+0]=W[9]
    # C[1,0]=W[0*3+1]=W[1], C[1,1]=W[1*3+1]=W[4], C[1,2]=W[2*3+1]=W[7], C[1,3]=W[3*3+1]=W[10]
    
    def add_strassen_block(U_global, V_global, W_global, 
                           u_rows, v_rows, w_cols,
                           su, sv, sw):
        """Add a Strassen block to global decomposition.
        u_rows: list of 2 global U row indices
        v_rows: list of 2 global V row indices  
        w_cols: list of 2 global W col indices (for the 2 output columns of the 2x2 C block)
        Actually for 2x2 C block, we need 4 W entries.
        """
        for c in range(7):
            for ri in range(2):
                if su[ri][c] != 0:
                    U_global[u_rows[ri]][c + 0] += su[ri][c]  # This won't work with offset
            
            # Actually each block gets 7 columns of rank
        pass
    
    # This is getting complex. Let me use a cleaner indexing scheme.
    # Assign 7 rank columns per Strassen block, 4 blocks = 28 columns
    # Then 16 columns for the bottom row schoolbook = 44 total
    
    col_offset = 0
    
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # U rows: A[0,0]=U[0], A[1,0]=U[4] (i=0,j=0 -> 0; i=1,j=0 -> 1*4+0=4)
    # V rows: B[0,0]=V[0], B[1,0]=V[4] (j=0,k=0 -> 0; j=1,k=0 -> 1*4+0=4)
    # W: C[0,0]=W[0], C[1,0]=W[1], C[0,1]=W[3], C[1,1]=W[4]
    # For 2x2 C: W[k*n+i] where n=3 (total), but for the sub-block:
    # The sub-C is 2x2, entry (kk, ii) where kk in {0,1}, ii in {0,1}
    # Global: C[kk, ii] -> W[kk*3+ii]
    
    # Strassen sub-indexing: local (i,j) -> local U[i*2+j]
    # A[0,0] global U[0*4+0]=U[0] -> local U[0]
    # A[0,1] global U[0*4+1]=U[1] -> local U[1]  
    # A[1,0] global U[1*4+0]=U[4] -> local U[2]
    # A[1,1] global U[1*4+1]=U[5] -> local U[3]
    
    u_map1 = [0, 5]   # A[0,0], A[1,1]  -- wait I need A[0,0],A[0,1],A[1,0],A[1,1]
    # local U[0]=A[0,0]=global U[0]
    # local U[1]=A[0,1]=global U[1]
    # local U[2]=A[1,0]=global U[4]
    # local U[3]=A[1,1]=global U[5]
    u1 = [0, 1, 4, 5]
    v1 = [0, 4]  # B[0,0],B[1,0] -- local V[0]=B[0,0]=V[0], V[1]=B[0,1]=V[1], V[2]=B[1,0]=V[4], V[3]=B[1,1]=V[5]
    # Actually for the (2x2)*(2x2) where B sub-block is rows 0,1 cols 0,1:
    # B[0,0]=V[0*4+0]=V[0], B[0,1]=V[0*4+1]=V[1], B[1,0]=V[1*4+0]=V[4], B[1,1]=V[1*4+1]=V[5]
    v1 = [0, 1, 4, 5]
    # C sub-block: rows 0,1 of C (which are global C rows 0,1), cols 0,1
    # C[0,0]=W[0*3+0]=W[0], C[0,1]=W[1*3+0]=W[3]
    # C[1,0]=W[0*3+1]=W[1], C[1,1]=W[1*3+1]=W[4]
    # Local C: C[0,0]=W[0], C[0,1]=W[3], C[1,0]=W[1], C[1,1]=W[4]
    w1 = [0, 3, 1, 4]  # local W[0],W[1],W[2],W[3] -> global W[0],W[3],W[1],W[4]
    
    for c in range(7):
        for li in range(4):
            U[u1[li]][col_offset + c] += su[li][c]
            V[v1[li]][col_offset + c] += sv[li][c]
            W[w1[li]][col_offset + c] += sw[li][c]
    col_offset += 7
    
    # Block 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    u2 = [0, 1, 4, 5]
    # B rows 0,1, cols 2,3: B[0,2]=V[2], B[0,3]=V[3], B[1,2]=V[6], B[1,3]=V[7]
    v2 = [2, 3, 6, 7]
    # C rows 0,1, cols 2,3: C[0,2]=W[6], C[0,3]=W[9], C[1,2]=W[7], C[1,3]=W[10]
    w2 = [6, 9, 7, 10]
    
    for c in range(7):
        for li in range(4):
            U[u2[li]][col_offset + c] += su[li][c]
            V[v2[li]][col_offset + c] += sv[li][c]
            W[w2[li]][col_offset + c] += sw[li][c]
    col_offset += 7
    
    # Block 3: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    # A rows 0,1, cols 2,3: A[0,2]=U[2], A[0,3]=U[3], A[1,2]=U[6], A[1,3]=U[7]
    u3 = [2, 3, 6, 7]
    # B rows 2,3, cols 0,1: B[2,0]=V[8], B[2,1]=V[9], B[3,0]=V[12], B[3,1]=V[13]
    v3 = [8, 9, 12, 13]
    w3 = [0, 3, 1, 4]  # same C sub-block as block 1
    
    for c in range(7):
        for li in range(4):
            U[u3[li]][col_offset + c] += su[li][c]
            V[v3[li]][col_offset + c] += sv[li][c]
            W[w3[li]][col_offset + c] += sw[li][c]
    col_offset += 7
    
    # Block 4: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    u4 = [2, 3, 6, 7]
    # B rows 2,3, cols 2,3: B[2,2]=V[10], B[2,3]=V[11], B[3,2]=V[14], B[3,3]=V[15]
    v4 = [10, 11, 14, 15]
    w4 = [6, 9, 7, 10]  # same C sub-block as block 2
    
    for c in range(7):
        for li in range(4):
            U[u4[li]][col_offset + c] += su[li][c]
            V[v4[li]][col_offset + c] += sv[li][c]
            W[w4[li]][col_offset + c] += sw[li][c]
    col_offset += 7
    
    # Part 2: bottom row of A (row 2 of 3x4)
    # A[2,:] (1x4) * B (4x4) -> 1x4 result
    # Schoolbook: for each k in 0..3, C[2,k] += A[2,k]*B[k,:] ... no
    # C[2, col] = sum_{k=0}^{3} A[2,k] * B[k,col]
    # Tensor entries: (i=2, j=k, col) -> W[col*3+2]
    # 4*4 = 16 terms
    
    for k in range(4):
        for col in range(4):
            U[2*4 + k][col_offset] = 1  # A[2,k] -> U[8+k]
            V[k*4 + col][col_offset] = 1  # B[k,col] -> V[4k+col]
            W[col*3 + 2][col_offset] = 1  # C[col, 2] -> W[3*col+2]
            col_offset += 1
    
    assert col_offset == total_rank, f"col_offset={col_offset}, expected {total_rank}"
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
