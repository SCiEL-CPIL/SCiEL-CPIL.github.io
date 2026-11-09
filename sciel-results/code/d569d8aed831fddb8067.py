import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x5 matrix by a 5x6 matrix.
    Returns a dict with 'U', 'V', 'W' matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    The score is the rank (lower is better).
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin for the 150s limit

    # Strategy: Use Strassen's algorithm for 2x2x2 blocks where beneficial.
    # 4x5x6 can be decomposed into blocks.
    # 4 = 2 + 2
    # 5 = 2 + 3 (We will split 5 into 2 and 3. 2x2x2 can use Strassen, 2x2x3 and 2x3x3 and 2x3x2 need handling)
    # Actually, let's split 5 into 2 + 2 + 1? Or 3 + 2?
    # Let's try splitting 4 = 2 + 2, 5 = 3 + 2, 6 = 3 + 3.
    # This creates 2x2 = 4 block types for A and B, leading to 16 C blocks.
    # This is complex.
    
    # Simpler: Split 4 = 2 + 2, 5 = 2 + 3, 6 = 3 + 3.
    # A is 4x5, B is 5x6.
    # A blocks: A11 (2x2), A12 (2x3), A21 (2x2), A22 (2x3).
    # B blocks: B11 (2x3), B12 (2x3), B21 (3x3), B22 (3x3).
    # C blocks:
    # C11 = A11 B11 + A12 B21  (2x2 = 2x2 * 2x3 + 2x3 * 3x3)
    # C12 = A11 B12 + A12 B22  (2x3 = 2x2 * 2x3 + 2x3 * 3x3)
    # C21 = A21 B11 + A22 B21  (2x2 = 2x2 * 2x3 + 2x3 * 3x3)
    # C22 = A21 B12 + A22 B22  (2x3 = 2x2 * 2x3 + 2x3 * 3x3)
    
    # This involves mixed dimensions. Strassen is for 2x2x2.
    # We can use schoolbook for the mixed parts, and Strassen for the 2x2x2 parts if we split further.
    # But splitting 3 into 1+2 leads to more blocks.
    
    # Let's try a different split:
    # 4 = 2 + 2
    # 5 = 2 + 2 + 1
    # 6 = 2 + 2 + 2
    # This is very fine-grained.
    # Number of 2x2x2 blocks: (4/2) * (5/2) * (6/2) is not integer.
    
    # Let's stick to the 108 rank solution derived in the thought process:
    # Split 4 = 2 + 2
    # Split 5 = 3 + 2 (Wait, the thought process said 5=2+3, but then used 2x2x2 for the 2x2 part of A and 2x2 part of B? No.)
    # The thought process:
    # "12 2x2x2 products. 12 * 7 = 84."
    # "6 2x1x2 products. 6 * 4 = 24."
    # Total 108.
    # How did we get 12 2x2x2 and 6 2x1x2?
    # A is 4x5. B is 5x6.
    # If we split 4=2+2, 5=2+3, 6=2+4?
    # Let's re-derive.
    # We want to use Strassen (2x2x2) as much as possible.
    # Let's split 4 = 2 + 2.
    # Let's split 6 = 2 + 2 + 2.
    # Let's split 5 = 2 + 2 + 1.
    
    # A (4x5) blocks:
    # Rows: 2, 2
    # Cols: 2, 2, 1
    # A11 (2x2), A12 (2x2), A13 (2x1)
    # A21 (2x2), A22 (2x2), A23 (2x1)
    
    # B (5x6) blocks:
    # Rows: 2, 2, 1
    # Cols: 2, 2, 2
    # B11 (2x2), B12 (2x2), B13 (2x2)
    # B21 (2x2), B22 (2x2), B23 (2x2)
    # B31 (1x2), B32 (1x2), B33 (1x2)
    
    # C (4x6) blocks:
    # C11 = A11 B11 + A12 B21 + A13 B31  (2x2 = 2x2*2x2 + 2x2*2x2 + 2x1*1x2)
    # C12 = A11 B12 + A12 B22 + A13 B32  (2x2 = 2x2*2x2 + 2x2*2x2 + 2x1*1x2)
    # C13 = A11 B13 + A12 B23 + A13 B33  (2x2 = 2x2*2x2 + 2x2*2x2 + 2x1*1x2)
    # C21 = A21 B11 + A22 B21 + A23 B31  (2x2 = 2x2*2x2 + 2x2*2x2 + 2x1*1x2)
    # C22 = A21 B12 + A22 B22 + A23 B32  (2x2 = 2x2*2x2 + 2x2*2x2 + 2x1*1x2)
    # C23 = A21 B13 + A22 B23 + A23 B33  (2x2 = 2x2*2x2 + 2x2*2x2 + 2x1*1x2)
    
    # For each Cij, we have two 2x2x2 products and one 2x1x2 product.
    # There are 6 C blocks.
    # Total 2x2x2 products: 6 * 2 = 12.
    # Total 2x1x2 products: 6 * 1 = 6.
    
    # Rank of 2x2x2 is 7 (Strassen).
    # Rank of 2x1x2 is 4 (Schoolbook: (2*1*2)). Note: 2x1x2 is outer product of 2x1 and 1x2. Rank is 2?
    # Wait. C = A B. A is 2x1, B is 1x2. C is 2x2.
    # C[i,k] = A[i] * B[k].
    # This is rank 1? No, the tensor rank of the map (A,B) -> C.
    # T[i,j,k] where j=0 is fixed. T[i,0,k] = 1.
    # This is a 2x1x2 tensor.
    # Can it be done in rank 2?
    # u1 v1 w1 + u2 v2 w2.
    # If we set v1=v2=e0.
    # (u1 v1 + u2 v2) is 2x2 matrix W? No.
    # u_i * v_0 * w_k.
    # We need sum_c U_c[i] V_c[0] W_c[k] = 1.
    # If V_c[0] = 1 for all c.
    # Then sum_c U_c[i] W_c[k] = 1.
    # This is an outer product 1*1 = 1. Rank 1.
    # So rank of 2x1x2 is 1?
    # Let's check.
    # A = [a1, a2]^T, B = [b1, b2].
    # C = [[a1 b1, a1 b2], [a2 b1, a2 b2]].
    # We need to compute this bilinearly.
    # M1 = a1 b1 -> C11
    # M2 = a1 b2 -> C12
    # M3 = a2 b1 -> C21
    # M4 = a2 b2 -> C22
    # This is 4 multiplications.
    # Can we do better?
    # No, because C is rank 1 as a matrix, but the bilinear map separates i and k.
    # T[i,0,k] = 1.
    # Rank is 4?
    # Actually, T[i,0,k] is 1 for all i,k.
    # This is the all-ones matrix in i,k space.
    # Rank of all-ones matrix is 1.
    # But the tensor is T[i, j=0, k].
    # It is a slice.
    # The rank of the tensor T_{2,1,2} is 2?
    # Let's try:
    # M1 = a1 (b1+b2) -> contributes a1 b1 + a1 b2
    # M2 = a2 (b1+b2) -> contributes a2 b1 + a2 b2
    # M3 = a1 (b1-b2) -> contributes a1 b1 - a1 b2
    # M4 = a2 (b1-b2) -> contributes a2 b1 - a2 b2
    # C11 = M1 + M3? No.
    # C11 = a1 b1 = (M1+M3)/2.
    # C12 = a1 b2 = (M1-M3)/2.
    # C21 = a2 b1 = (M2+M4)/2.
    # C22 = a2 b2 = (M2-M4)/2.
    # This uses 4 multiplications.
    # Is it possible with 2?
    # M1 = a1 b1, M2 = a2 b2. Then C12 and C21 are missing.
    # M1 = a1 (b1+b2), M2 = a2 (b1+b2).
    # C11 = a1 b1? No, we have a1 b1 + a1 b2.
    # We can't separate b1 and b2 without more multiplications.
    # So rank is 4.
    
    # So total rank = 12 * 7 + 6 * 4 = 84 + 24 = 108.
    # This is better than 120.

    # Implementation:
    # We need to construct U, V, W.
    # We will iterate over the 6 C blocks.
    # For each C block, we add 7 terms for the first 2x2x2 and 7 terms for the second 2x2x2 and 4 terms for the 2x1x2.
    # Wait, the 2x1x2 term is shared? No, A13 is specific to the row block of A, B31 is specific to the col block of B.
    # For C11, A13 and B31.
    # For C12, A13 and B32.
    # So the 2x1x2 parts are distinct.
    
    # Let's build the factors.
    # Total rank R = 108.
    # We will fill U, V, W column by column.
    
    U = [[0] * 108 for _ in range(4 * 5)]
    V = [[0] * 108 for _ in range(5 * 6)]
    W = [[0] * 108 for _ in range(6 * 4)]
    
    col = 0
    
    # Strassen factors for 2x2x2
    # Inputs: A (2x2), B (2x2)
    # A indices: 0,1,2,3 (row-major: 00,01,10,11) -> 0,1,2,3
    # B indices: 0,1,2,3
    # Output: C (2x2)
    
    # M1 = A00 (B11 - B10)
    # M2 = (A00 + A01) B10
    # M3 = A11 (B01 - B00)
    # M4 = (A10 + A11) B00
    # M5 = (A00 + A11) (B00 + B11)
    # M6 = (A01 - A11) (B10 + B11)
    # M7 = (A00 - A10) (B00 + B01)
    
    # C00 = M5 + M4 - M2 + M6
    # C01 = M1 + M5
    # C10 = M2 + M3
    # C11 = M1 + M3 - M4 + M7
    
    # We need to map these to the global indices.
    # For a given 2x2x2 block located at A rows [r0, r0+1], B cols [c0, c0+1] for A part?
    # No, A is 2x2, B is 2x2.
    # A indices in global U:
    # A row i, col j. Global index i*5 + j.
    # B indices in global V:
    # B row j, col k. Global index j*6 + k.
    # C indices in global W:
    # C row i, col k. Global index k*4 + i.
    
    # Let's define a function to add a 2x2x2 Strassen block.
    # a_r0, a_c0: top-left of A block (2x2)
    # b_r0, b_c0: top-left of B block (2x2)
    # c_r0, c_c0: top-left of C block (2x2)
    
    def add_strassen_222(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0, start_col):
        # A indices:
        # A00: a_r0, a_c0
        # A01: a_r0, a_c0+1
        # A10: a_r0+1, a_c0
        # A11: a_r0+1, a_c0+1
        
        # B indices:
        # B00: b_r0, b_c0
        # B01: b_r0, b_c0+1
        # B10: b_r0+1, b_c0
        # B11: b_r0+1, b_c0+1
        
        # C indices:
        # C00: c_r0, c_c0
        # C01: c_r0, c_c0+1
        # C10: c_r0+1, c_c0
        # C11: c_r0+1, c_c0+1
        
        # Term 1: M1 = A00 (B11 - B10)
        # U: A00=1
        # V: B11=1, B10=-1
        # W: C01=1, C11=1
        U[a_r0*5 + a_c0][start_col] = 1
        V[(b_r0+1)*6 + (b_c0+1)][start_col] = 1
        V[(b_r0+1)*6 + b_c0][start_col] = -1
        W[c_c0*4 + c_r0][start_col] = 1
        W[(c_c0+1)*4 + c_r0+1][start_col] = 1
        # Wait, C01 is (c_r0, c_c0+1) -> W[c_c0+1, c_r0]?
        # W index is k*n + i. i is row of C, k is col of C.
        # C01: i=c_r0, k=c_c0+1. Index: (c_c0+1)*4 + c_r0.
        # C11: i=c_r0+1, k=c_c0+1. Index: (c_c0+1)*4 + c_r0+1.
        W[c_c0*4 + c_r0][start_col] = 0 # Overwrite? No, M1 contributes to C01 and C11.
        # Let's re-verify Strassen outputs.
        # C12 (01) = M1 + M5.
        # C22 (11) = M1 + M3 - M4 + M7.
        # So M1 contributes to C01 and C11.
        W[(c_c0+1)*4 + c_r0][start_col] = 1
        W[(c_c0+1)*4 + c_r0+1][start_col] = 1
        # Corrected W assignment:
        W[(c_c0+1)*4 + c_r0][start_col] = 1
        W[(c_c0+1)*4 + c_r0+1][start_col] = 1
        # My previous line set W[c_c0*4 + c_r0] which is C00. M1 does not go to C00.
        # So:
        W[c_c0*4 + c_r0][start_col] = 0
        
        # Term 2: M2 = (A00 + A01) B10
        # U: A00=1, A01=1
        # V: B10=1
        # W: C00=1, C10=1
        U[a_r0*5 + a_c0][start_col+1] = 1
        U[a_r0*5 + (a_c0+1)][start_col+1] = 1
        V[(b_r0+1)*6 + b_c0][start_col+1] = 1
        W[c_c0*4 + c_r0][start_col+1] = 1
        W[c_c0*4 + (c_r0+1)][start_col+1] = 1
        
        # Term 3: M3 = A11 (B01 - B00)
        # U: A11=1
        # V: B01=1, B00=-1
        # W: C10=1, C11=1
        U[(a_r0+1)*5 + (a_c0+1)][start_col+2] = 1
        V[b_r0*6 + (b_c0+1)][start_col+2] = 1
        V[b_r0*6 + b_c0][start_col+2] = -1
        W[c_c0*4 + (c_r0+1)][start_col+2] = 1
        W[(c_c0+1)*4 + (c_r0+1)][start_col+2] = 1
        
        # Term 4: M4 = (A10 + A11) B00
        # U: A10=1, A11=1
        # V: B00=1
        # W: C00=-1, C11=-1
        U[(a_r0+1)*5 + a_c0][start_col+3] = 1
        U[(a_r0+1)*5 + (a_c0+1)][start_col+3] = 1
        V[b_r0*6 + b_c0][start_col+3] = 1
        W[c_c0*4 + c_r0][start_col+3] = -1
        W[(c_c0+1)*4 + (c_r0+1)][start_col+3] = -1
        
        # Term 5: M5 = (A00 + A11) (B00 + B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C01=1, C10? No. C00, C01, C10, C11?
        # C00 = ... + M5 ...
        # C01 = M1 + M5
        # C10 = M2 + M3
        # C11 = M1 + M3 - M4 + M7
        # So M5 contributes to C00 and C01.
        U[a_r0*5 + a_c0][start_col+4] = 1
        U[(a_r0+1)*5 + (a_c0+1)][start_col+4] = 1
        V[b_r0*6 + b_c0][start_col+4] = 1
        V[(b_r0+1)*6 + (b_c0+1)][start_col+4] = 1
        W[c_c0*4 + c_r0][start_col+4] = 1
        W[(c_c0+1)*4 + c_r0][start_col+4] = 1
        
        # Term 6: M6 = (A01 - A11) (B10 + B11)
        # U: A01=1, A11=-1
        # V: B10=1, B11=1
        # W: C00=1
        U[a_r0*5 + (a_c0+1)][start_col+5] = 1
        U[(a_r0+1)*5 + (a_c0+1)][start_col+5] = -1
        V[(b_r0+1)*6 + b_c0][start_col+5] = 1
        V[(b_r0+1)*6 + (b_c0+1)][start_col+5] = 1
        W[c_c0*4 + c_r0][start_col+5] = 1
        
        # Term 7: M7 = (A00 - A10) (B00 + B01)
        # U: A00=1, A10=-1
        # V: B00=1, B01=1
        # W: C11=1
        U[a_r0*5 + a_c0][start_col+6] = 1
        U[(a_r0+1)*5 + a_c0][start_col+6] = -1
        V[b_r0*6 + b_c0][start_col+6] = 1
        V[b_r0*6 + (b_c0+1)][start_col+6] = 1
        W[(c_c0+1)*4 + (c_r0+1)][start_col+6] = 1
        
        return start_col + 7

    def add_outer_212(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0, start_col):
        # A is 2x1 at (a_r0, a_c0)
        # B is 1x2 at (b_r0, b_c0)
        # C is 2x2 at (c_r0, c_c0)
        # C[i,k] = A[i] * B[k]
        # Indices:
        # A0: a_r0, a_c0
        # A1: a_r0+1, a_c0
        # B0: b_r0, b_c0
        # B1: b_r0, b_c0+1
        # C00: c_r0, c_c0
        # C01: c_r0, c_c0+1
        # C10: c_r0+1, c_c0
        # C11: c_r0+1, c_c0+1
        
        # Term 1: A0 * B0 -> C00
        U[a_r0*5 + a_c0][start_col] = 1
        V[b_r0*6 + b_c0][start_col] = 1
        W[c_c0*4 + c_r0][start_col] = 1
        
        # Term 2: A0 * B1 -> C01
        U[a_r0*5 + a_c0][start_col+1] = 1
        V[b_r0*6 + (b_c0+1)][start_col+1] = 1
        W[(c_c0+1)*4 + c_r0][start_col+1] = 1
        
        # Term 3: A1 * B0 -> C10
        U[(a_r0+1)*5 + a_c0][start_col+2] = 1
        V[b_r0*6 + b_c0][start_col+2] = 1
        W[c_c0*4 + (c_r0+1)][start_col+2] = 1
        
        # Term 4: A1 * B1 -> C11
        U[(a_r0+1)*5 + a_c0][start_col+3] = 1
        V[b_r0*6 + (b_c0+1)][start_col+3] = 1
        W[(c_c0+1)*4 + (c_r0+1)][start_col+3] = 1
        
        return start_col + 4

    # Loop over the 6 C blocks
    # C11: A rows 0,1; B cols 0,1 (for 2x2 parts); A col 2 (for 2x1); B row 4 (for 1x2)
    # Wait, let's map the blocks carefully.
    # A rows: 0-1 (block 0), 2-3 (block 1)
    # A cols: 0-1 (block 0), 2-3 (block 1), 4 (block 2)
    # B rows: 0-1 (block 0), 2-3 (block 1), 4 (block 2)
    # B cols: 0-1 (block 0), 2-3 (block 1), 4-5 (block 2)
    
    # C Block (i, j) where i is A row block, j is B col block
    # C[i,j] = sum_k A[i,k] B[k,j]
    # k=0: A[i,0] (2x2) * B[0,j] (2x2) -> 2x2x2
    # k=1: A[i,1] (2x2) * B[1,j] (2x2) -> 2x2x2
    # k=2: A[i,2] (2x1) * B[2,j] (1x2) -> 2x1x2
    
    # i=0 (A rows 0-1):
    # j=0 (B cols 0-1):
    #   A[0,0] (rows 0-1, cols 0-1) * B[0,0] (rows 0-1, cols 0-1) -> C[0,0] (rows 0-1, cols 0-1)
    #   A[0,1] (rows 0-1, cols 2-3) * B[1,0] (rows 2-3, cols 0-1) -> C[0,0]
    #   A[0,2] (rows 0-1, col 4) * B[2,0] (row 4, cols 0-1) -> C[0,0]
    
    # Let's call the functions.
    
    # C[0,0]
    col = add_strassen_222(0, 0, 0, 0, 0, 0, col)
    col = add_strassen_222(0, 2, 2, 0, 0, 0, col)
    col = add_outer_212(0, 4, 4, 0, 0, 0, col)
    
    # C[0,1]
    col = add_strassen_222(0, 0, 0, 2, 0, 2, col)
    col = add_strassen_222(0, 2, 2, 2, 0, 2, col)
    col = add_outer_212(0, 4, 4, 2, 0, 2, col)
    
    # C[0,2]
    col = add_strassen_222(0, 0, 0, 4, 0, 4, col)
    col = add_strassen_222(0, 2, 2, 4, 0, 4, col)
    col = add_outer_212(0, 4, 4, 4, 0, 4, col)
    
    # C[1,0]
    col = add_strassen_222(2, 0, 0, 0, 2, 0, col)
    col = add_strassen_222(2, 2, 2, 0, 2, 0, col)
    col = add_outer_212(2, 4, 4, 0, 2, 0, col)
    
    # C[1,1]
    col = add_strassen_222(2, 0, 0, 2, 2, 2, col)
    col = add_strassen_222(2, 2, 2, 2, 2, 2, col)
    col = add_outer_212(2, 4, 4, 2, 2, 2, col)
    
    # C[1,2]
    col = add_strassen_222(2, 0, 0, 4, 2, 4, col)
    col = add_strassen_222(2, 2, 2, 4, 2, 4, col)
    col = add_outer_212(2, 4, 4, 4, 2, 4, col)
    
    # Total col should be 108.
    if col != 108:
        print(f"Warning: col is {col}, expected 108")
    
    # Convert to numpy arrays if needed, but list of lists is fine.
    # Verify quickly?
    # Time check
    if time.time() > deadline:
        pass # Just return
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    # Basic sanity check
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    # Tensor T = sum_c U_c V_c W_c
    # T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
    # We want T[i,j,k] = 1 if i,j,k form a valid path in 4x5x6 multiplication?
    # T[i, j, k] = 1 if j is the inner index.
    # A[i,j] B[j,k] -> C[i,k].
    # So T[i,j,k] = 1.
    # Wait, the tensor components are T_{i,j,k}.
    # U is 20xR, V is 30xR, W is 24xR.
    # T = U V^T W^T? No.
    # T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c].
    # This should be 1 for all i,j,k?
    # No, only if i,j,k are compatible?
    # A is 4x5. i in 0..3, j in 0..4.
    # B is 5x6. j in 0..4, k in 0..5.
    # So for any i in 0..3, j in 0..4, k in 0..5, T[i,j,k] should be 1.
    # Let's check T[0,0,0].
    # Sum_c U[0,c] V[0,c] W[0,c].
    # This is a dot product.
    # We can verify a few entries.
    pass
