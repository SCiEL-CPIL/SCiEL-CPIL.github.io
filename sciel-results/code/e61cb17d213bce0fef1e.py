"""
Matrix multiplication 2x2 by 2x15: rank-minimal exact tensor decomposition.
Uses Strassen for 2x2 blocks to reduce rank from 60 to 53.
"""
import time

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 15
    # We'll use Strassen's algorithm on 2x2 blocks.
    # Split the 2x15 matrix into 2x2 and 2x1 blocks? No, let's think differently.
    
    # Actually, let's partition the 15 columns into groups.
    # Strassen works on 2x2 by 2x2. We can group columns.
    # 15 = 2*7 + 1, so we have 7 groups of 2 columns and 1 group of 1 column.
    # For each 2x2 block: use Strassen (rank 7 instead of 8)
    # For the 2x1 block: use schoolbook (rank 2 instead of 2)
    
    # Total rank: 7*7 + 1*2 = 49 + 2 = 51? Wait, let me recalculate.
    # Each 2x2 by 2x2 multiplication has rank 7 with Strassen.
    # Each 2x1 by 2x1 multiplication has rank 2 with schoolbook.
    # 7 blocks of 2x2: 7 * 7 = 49
    # 1 block of 2x1: 1 * 2 = 2
    # Total: 51
    
    # But wait, the task says 2x2 matrix times 2x15 matrix.
    # A is 2x2, B is 2x15.
    # Partition B into 7 blocks of size 2x2 and 1 block of size 2x1.
    # A stays as 2x2.
    
    # For each 2x2 block of B: compute A * B_block using Strassen (rank 7)
    # For the 2x1 block: compute A * B_block using schoolbook (rank 2)
    
    rank_total = 7 * 7 + 2  # 51
    U = [[0] * rank_total for _ in range(n * m)]  # 4 x 51
    V = [[0] * rank_total for _ in range(m * p)]  # 30 x 51
    W = [[0] * rank_total for _ in range(p * n)]  # 30 x 51
    
    col = 0
    
    # Strassen coefficients for 2x2 multiplication
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = A * B
    # Strassen uses 7 products:
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each 2x2 block (7 blocks):
    for block in range(7):
        # Block covers columns 2*block and 2*block+1
        for s in range(7):  # 7 Strassen terms
            U_row_idx = 0  # will be filled per term
            V_row_idx = 0
            W_row_idx = 0
            
            if s == 0:  # M1 = (a11+a22)*(b11+b22)
                # U contributes to a11 and a22
                U[0][col] = 1  # a11
                U[3][col] = 1  # a22
                # V contributes to b11 and b22
                V[2*block][col] = 1  # b11 (row 0, col 2*block)
                V[2*block+1+2][col] = 1  # b22 (row 1, col 2*block+1) -> row 2, col 2*block+1
                # Wait, V is indexed as [j*p+k], j in 0..1, k in 0..14
                # For block b, columns are 2b and 2b+1
                # b11: j=0, k=2b -> V[0*15+2b] = V[2b]
                # b22: j=1, k=2b+1 -> V[1*15+2b+1] = V[15+2b+1] = V[16+2b]
                # Let me redo this.
                pass
            
            # Let me restart with proper indexing
            pass
    
    # Let me redo this more carefully.
    # A is 2x2: A[i,j] where i,j in 0,1
    # U has shape (n*m, rank) = (4, rank)
    # U[i*m+j, c] corresponds to A[i,j] coefficient
    
    # B is 2x15: B[j,k] where j in 0,1, k in 0..14
    # V has shape (m*p, rank) = (30, rank)
    # V[j*p+k, c] corresponds to B[j,k] coefficient
    
    # C is 2x15: C[i,k]
    # W has shape (p*n, rank) = (30, rank)
    # W[k*n+i, c] corresponds to C[i,k] coefficient
    
    # For Strassen on a 2x2 block:
    # A is 2x2, B_block is 2x2 (columns 2b, 2b+1)
    
    U = [[0] * rank_total for _ in range(4)]
    V = [[0] * rank_total for _ in range(30)]
    W = [[0] * rank_total for _ in range(30)]
    
    col = 0
    
    # 7 blocks of 2x2
    for b in range(7):
        # Strassen 7 terms
        # M1 = (A00+A11)*(B00+B11) where B00=B[0,2b], B11=B[1,2b+1]
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11) where B01=B[0,2b+1], B11=B[1,2b+1]
        # M4 = A11*(B10-B00) where B10=B[1,2b]
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1
        U[0][col] = 1; U[3][col] = 1  # A00 + A11
        V[0*15+2*b][col] = 1; V[1*15+2*b+1][col] = 1  # B[0,2b] + B[1,2b+1]
        W[(2*b)*2+0][col] = 1; W[(2*b+1)*2+1][col] = 1  # C[0,2b] += 1, C[1,2b+1] += 1
        # Wait, W[k*n+i, c]: C[i,k] -> W[k*2+i]
        # C[0,2b]: W[2b*2+0] = W[4b]
        # C[1,2b+1]: W[(2b+1)*2+1] = W[4b+3]
        W[4*b][col] = 1; W[4*b+3][col] = 1
        col += 1
        
        # M2
        U[1][col] = 1; U[3][col] = 1  # A10 + A11
        V[0*15+2*b][col] = 1  # B[0,2b]
        W[(2*b+1)*2+1][col] = 1  # C[1,2b+1] += 1 -> W[4b+3]
        # C[1,2b+1]: W[(2b+1)*2+1] = W[4b+3]
        W[4*b+3][col] = 1
        col += 1
        
        # M3
        U[0][col] = 1  # A00
        V[0*15+2*b+1][col] = 1; V[1*15+2*b+1][col] = -1  # B[0,2b+1] - B[1,2b+1]
        W[(2*b+1)*2+0][col] = 1  # C[0,2b+1] += 1 -> W[(2b+1)*2+0] = W[4b+2]
        W[4*b+2][col] = 1
        col += 1
        
        # M4
        U[3][col] = 1  # A11
        V[1*15+2*b][col] = 1; V[0*15+2*b][col] = -1  # B[1,2b] - B[0,2b]
        W[(2*b)*2+0][col] = 1; W[(2*b+1)*2+1][col] = 1  # C[0,2b] += 1, C[1,2b+1] += 1
        W[4*b][col] = 1; W[4*b+3][col] = 1
        col += 1
        
        # M5
        U[0][col] = 1; U[1][col] = 1  # A00 + A01
        V[1*15+2*b+1][col] = 1  # B[1,2b+1]
        W[(2*b)*2+0][col] = -1; W[(2*b+1)*2+0][col] = 1  # C[0,2b] -= 1, C[0,2b+1] += 1
        W[4*b][col] = -1; W[4*b+2][col] = 1
        col += 1
        
        # M6
        U[1][col] = 1; U[0][col] = -1  # A10 - A00
        V[0*15+2*b][col] = 1; V[0*15+2*b+1][col] = 1  # B[0,2b] + B[0,2b+1]
        W[(2*b+1)*2+1][col] = 1  # C[1,2b+1] += 1
        W[4*b+3][col] = 1
        col += 1
        
        # M7
        U[1][col] = 1; U[3][col] = -1  # A01 - A11
        V[1*15+2*b][col] = 1; V[1*15+2*b+1][col] = 1  # B[1,2b] + B[1,2b+1]
        W[(2*b)*2+0][col] = 1  # C[0,2b] += 1
        W[4*b][col] = 1
        col += 1
    
    # 1 block of 2x1 (column 14)
    # Schoolbook: 2 terms
    # C[0,14] = A[0,0]*B[0,14] + A[0,1]*B[1,14]
    # C[1,14] = A[1,0]*B[0,14] + A[1,1]*B[1,14]
    
    # Term 1: A[0,0]*B[0,14]
    U[0][col] = 1
    V[0*15+14][col] = 1
    W[14*2+0][col] = 1  # C[0,14]
    col += 1
    
    # Term 2: A[0,1]*B[1,14]
    U[1][col] = 1
    V[1*15+14][col] = 1
    W[14*2+0][col] = 1  # C[0,14]
    col += 1
    
    # Term 3: A[1,0]*B[0,14]
    U[2][col] = 1
    V[0*15+14][col] = 1
    W[14*2+1][col] = 1  # C[1,14]
    col += 1
    
    # Term 4: A[1,1]*B[1,14]
    U[3][col] = 1
    V[1*15+14][col] = 1
    W[14*2+1][col] = 1  # C[1,14]
    col += 1
    
    # Wait, that's 4 terms for the 2x1 block, not 2.
    # For 2x1 by 2x1: C[i,k] = sum_j A[i,j]*B[j,k]
    # C[0,14] = A[0,0]*B[0,14] + A[0,1]*B[1,14]
    # C[1,14] = A[1,0]*B[0,14] + A[1,1]*B[1,14]
    # That's 4 terms total (2 for each output entry).
    # So rank for 2x1 block is 4, not 2.
    
    # Total rank: 7*7 + 4 = 49 + 4 = 53
    
    # Let me fix the last part
    U = [[0] * 53 for _ in range(4)]
    V = [[0] * 53 for _ in range(30)]
    W = [[0] * 53 for _ in range(30)]
    
    col = 0
    
    for b in range(7):
        # M1
        U[0][col] = 1; U[3][col] = 1
        V[2*b][col] = 1; V[15+2*b+1][col] = 1
        W[4*b][col] = 1; W[4*b+3][col] = 1
        col += 1
        
        # M2
        U[1][col] = 1; U[3][col] = 1
        V[2*b][col] = 1
        W[4*b+3][col] = 1
        col += 1
        
        # M3
        U[0][col] = 1
        V[2*b+1][col] = 1; V[15+2*b+1][col] = -1
        W[4*b+2][col] = 1
        col += 1
        
        # M4
        U[3][col] = 1
        V[15+2*b][col] = 1; V[2*b][col] = -1
        W[4*b][col] = 1; W[4*b+3][col] = 1
        col += 1
        
        # M5
        U[0][col] = 1; U[1][col] = 1
        V[15+2*b+1][col] = 1
        W[4*b][col] = -1; W[4*b+2][col] = 1
        col += 1
        
        # M6
        U[1][col] = 1; U[0][col] = -1
        V[2*b][col] = 1; V[2*b+1][col] = 1
        W[4*b+3][col] = 1
        col += 1
        
        # M7
        U[1][col] = 1; U[3][col] = -1
        V[15+2*b][col] = 1; V[15+2*b+1][col] = 1
        W[4*b][col] = 1
        col += 1
    
    # 2x1 block: 4 terms
    # C[0,14] = A[0,0]*B[0,14] + A[0,1]*B[1,14]
    # C[1,14] = A[1,0]*B[0,14] + A[1,1]*B[1,14]
    
    # A[0,0]*B[0,14]
    U[0][col] = 1; V[14][col] = 1; W[28][col] = 1
    col += 1
    
    # A[0,1]*B[1,14]
    U[1][col] = 1; V[29][col] = 1; W[28][col] = 1
    col += 1
    
    # A[1,0]*B[0,14]
    U[2][col] = 1; V[14][col] = 1; W[29][col] = 1
    col += 1
    
    # A[1,1]*B[1,14]
    U[3][col] = 1; V[29][col] = 1; W[29][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
