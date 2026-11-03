"""Matrix multiplication 2x4x7 - rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x7 matrix.
    
    We use a block-structured approach:
    - Split the 4x4 middle dimension into 2x2 blocks
    - Apply Strassen's algorithm to each 2x2x2 block (rank 7 instead of 8)
    - Handle remainders with schoolbook
    
    The 2x4x7 matrix multiplication can be decomposed as:
    - Split A (2x4) into two 2x2 blocks: A = [A1, A2]
    - Split B (4x7) into two 4x7 blocks: B = [B1; B2] where B1 is 2x7, B2 is 2x7
    - Actually, let's think more carefully.
    
    A is 2x4, B is 4x7, C = A*B is 2x7.
    
    Let's split the middle dimension (4) into two parts of size 2:
    A = [A_left, A_right] where A_left is 2x2, A_right is 2x2
    B = [B_top; B_bottom] where B_top is 2x7, B_bottom is 2x7
    
    Then C = A_left * B_top + A_right * B_bottom
    
    Each 2x2 times 2x7 multiplication:
    - A_left (2x2) * B_top (2x7): This is 2*2*7 = 28 scalar multiplications schoolbook
    - But we can view this as: for each of the 7 columns of B_top, do a 2x2 * 2x1 multiplication
    - A 2x2 * 2x1 can be done with Strassen in 7 multiplications (treating it as 2x2 * 2x2 with one column zero, but that doesn't help directly)
    
    Actually, let me reconsider. The tensor for 2x2x2 matrix mult has rank 7 (Strassen).
    For 2x2x7 (i.e., 2x2 matrix times 2x7 matrix), we can do 7 multiplications per column of output? No, that's not right either.
    
    Let me think about this differently. The matrix multiplication tensor for m x n * n x p has a known structure.
    
    For 2x4x7:
    - We can split the 4 into 2+2
    - This gives us two 2x2x7 sub-problems
    - Each 2x2x7: multiply a 2x2 matrix by a 2x7 matrix to get a 2x7 result
    
    For 2x2x7: Think of it as 7 independent 2x2x1 multiplications. But 2x2x1 is just a linear combination, rank 2.
    So 2x2x7 would be 7*2 = 14 with schoolbook.
    
    But we can do better. A 2x2 matrix times a 2x7 matrix:
    Let A = [[a,b],[c,d]], B = [b11..b17; b21..b27]
    C = [a*b11+b*b21, ..., a*b17+b*b27; c*b11+d*b21, ..., c*b17+d*b27]
    
    This is 4*7 = 28 terms, but each output element is a sum of 2 products.
    
    Actually, I think the cleanest approach is:
    
    Split 4 = 2 + 2.
    A (2x4) = [A1 | A2] where A1, A2 are 2x2
    B (4x7) = [B1; B2] where B1, B2 are 2x7
    
    C = A1*B1 + A2*B2
    
    Now, A1*B1: 2x2 * 2x7. We can use Strassen-like decomposition.
    Treat B1 as 7 columns, each a 2x1 vector. But that's not efficient.
    
    Alternative: Pad B1 to 2x2 by adding a zero column? No, that changes the problem.
    
    Let me just use the known result: the rank of the 2x2x2 matrix multiplication tensor is 7.
    
    For 2x4x7, one approach:
    - Split the 4 into 2+2
    - Each 2x2x7 block: this is equivalent to doing 7 instances of 2x2x1, but we can batch them.
    
    Actually, I'll use a different strategy. Let me construct an explicit decomposition.
    
    Schoolbook rank: 2*4*7 = 56.
    
    Using Strassen on the 2x2 blocks:
    Split A = [A11 A12; A21 A22] but A is only 2x4, so A = [A1 A2] with A1, A2 being 2x2.
    B = [B1; B2] with B1, B2 being 2x7.
    
    C = A1*B1 + A2*B2
    
    For each 2x2 * 2x7 multiplication, we can use 7 multiplications per "Strassen block".
    But 2x7 is not 2x2.
    
    Let me think of 2x2 * 2x7 as: for each of the 7 columns, we need 2 output values.
    Each column computation is a 2x2 * 2x1, which has rank 2 (just 2 dot products of length 2).
    So 2x2*2x7 has rank at most 7*2 = 14 schoolbook, but maybe less.
    
    Actually, I think I should just try to find a good decomposition empirically or use known results.
    
    Let me try a different approach: use the fact that we can share computations.
    
    For 2x4x7, I'll use the following block decomposition:
    - Split the 4 into 2+2
    - Use Strassen for the 2x2x2 core, and handle the extra columns of B with linear combinations.
    
    Let me just code up a reasonable decomposition and verify it.
    """
    n, m, p = 2, 4, 7  # A is n x m, B is m x p, C is n x p
    
    # We'll build the decomposition using a combination of Strassen and schoolbook.
    # Split m=4 into two parts of 2.
    
    # A = [A1 | A2], each 2x2
    # B = [B1; B2], each 2x7
    # C = A1*B1 + A2*B2
    
    # For A1*B1 (2x2 * 2x7):
    #   Let A1 = [[a, b], [c, d]], B1 has columns (x1,y1), (x2,y2), ..., (x7,y7)
    #   Output row 1: a*x1 + b*y1, a*x2 + b*y2, ..., a*x7 + b*y7
    #   Output row 2: c*x1 + d*y1, c*x2 + d*y2, ..., c*x7 + d*y7
    #   
    #   Using Strassen-style: define 7 products:
    #   M1 = (a+b)*(x+y)  -- but x,y vary per column, so this doesn't work directly.
    
    # The issue is that Strassen works for square matrices. For 2x2 * 2x7, we can't directly apply it.
    
    # Alternative: Just use schoolbook for now and see if we can do better.
    # Schoolbook: 56.
    
    # Let me try a different split. What if we split p=7 into 2+2+2+1?
    # Then we have 2x4x2 (three times) and 2x4x1.
    # 2x4x2: split m=4 into 2+2, get 2x2x2 + 2x2x2, each rank 7 with Strassen, total 14.
    # 2x4x1: split m=4 into 2+2, get 2x2x1 + 2x2x1, each rank 2, total 4.
    # Total: 3*14 + 4 = 46.
    
    # Or split p=7 into 3+4?
    # 2x4x3 and 2x4x4.
    # 2x4x4: split m=4 into 2+2, get two 2x2x4. Each 2x2x4: split p=4 into 2+2, get two 2x2x2, each rank 7, so 2x2x4 has rank 14. Two of them: 28.
    # 2x4x3: split m=4 into 2+2, get two 2x2x3. 2x2x3: split p=3 into 2+1, get 2x2x2 (rank 7) + 2x2x1 (rank 2) = 9. Two of them: 18.
    # Total: 28 + 18 = 46.
    
    # Let me try to implement the p=7 -> 2+2+2+1 split with Strassen on the 2x2x2 blocks.
    
    # Actually, let me just implement a clean decomposition and verify correctness.
    
    # I'll use the approach: split m=4 into 2+2, and for each 2x2x7 subproblem, use schoolbook (rank 14 each), total 28.
    # That's better than 56.
    
    # But can we do better on 2x2x7?
    # 2x2 * 2x7: The tensor has 2*2*7 = 28 entries. Rank is at most 28.
    # With Strassen on the 2x2 part... it's not straightforward.
    
    # Let me just go with a known-good decomposition. I'll use the block Strassen approach.
    
    # Split: m=4 = 2+2, p=7 = 2+2+2+1
    # This gives 4 subproblems: three 2x2x2 and one 2x2x1, times 2 (for the m split) = 6 subproblems.
    # Wait, no. Let me be more careful.
    
    # A (2x4) = [A1 | A2], A1, A2 are 2x2
    # B (4x7) = [B1; B2], B1, B2 are 2x7
    # C (2x7) = A1*B1 + A2*B2
    
    # Now split B1 (2x7) into columns: B1 = [b11..b17; b21..b27]
    # Split into 7 columns, each 2x1.
    # A1 * column_j = 2x2 * 2x1, rank 2.
    # So A1*B1 has rank at most 7*2 = 14.
    # Similarly A2*B2 has rank at most 14.
    # Total rank: 28.
    
    # That's a 2x improvement over schoolbook (56 -> 28).
    
    # Can we do better? What if we group columns?
    # A1 * [col1 col2] = 2x2 * 2x2, rank 7 with Strassen.
    # So A1 * B1 where B1 is 2x7: split into 2x2, 2x2, 2x1.
    # Rank: 7 + 7 + 2 = 16.
    # Similarly A2*B2: 16.
    # Total: 32.
    
    # Hmm, 28 is better than 32. So the column-by-column approach is better for 2x2x7.
    
    # Let me implement the rank-28 decomposition.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 2x4 = 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 4x7 = 28 rows
    W = [[0] * rank for _ in range(p * n)]  # 7x2 = 14 rows
    
    # U[i*m + j, c] where i in [0,2), j in [0,4), c in [0,28)
    # V[j*p + k, c] where j in [0,4), k in [0,7), c in [0,28)
    # W[k*n + i, c] where k in [0,7), i in [0,2), c in [0,28)
    
    # Decomposition:
    # For each of the 2 "left" A-blocks (A1, A2) and each of the 7 columns:
    #   We have 2 multiplications per column (one for each row of A-block).
    #   Total: 2 * 7 * 2 = 28.
    
    col_idx = 0
    for a_block in range(2):  # A1 or A2
        for col in range(7):  # column of B
            for a_row in range(2):  # row within the 2x2 A-block
                # U: A-block row a_row, column j = a_block*2 + a_col
                # We need to identify which element of A is used.
                # A1 corresponds to columns 0,1 of A; A2 corresponds to columns 2,3.
                # A_block = A[:, a_block*2 : a_block*2+2]
                # So A[a_row, a_block*2 + a_col]
                
                # For this multiplication, U picks out A[a_row, a_block*2 + a_col]
                # But we have 2 a_col values (0 and 1) for each a_row.
                # So we need 2 multiplications per (a_block, col, a_row)? No.
                
                # Let me reconsider. For A1 * B1_col:
                # A1 = [[a00, a01], [a10, a11]]
                # B1_col = [x, y]
                # Result = [a00*x + a01*y, a10*x + a11*y]
                # This is 4 products: a00*x, a01*y, a10*x, a11*y.
                # Rank 4 for 2x2*2x1? No, rank is 2 because:
                # result[0] = a00*x + a01*y
                # result[1] = a10*x + a11*y
                # These are 2 independent bilinear forms, so rank 2.
                # The decomposition uses 2 "multiplication slots":
                # Slot 1: multiply (a00*x + a01*y) -- but that's not a single product.
                
                # I think I'm overcomplicating this. Let me use the standard rank-2 decomposition for 2x2*2x1:
                # Actually, 2x2 * 2x1: the output is 2x1. Each output element is a dot product of length 2.
                # output[0] = A[0,0]*B[0] + A[0,1]*B[1]
                # output[1] = A[1,0]*B[0] + A[1,1]*B[1]
                # This is 4 scalar products, but the tensor rank is 4 (since each product is independent).
                # Wait, no. The tensor is sum_{i,j} A[i,j] * B[j] * e_i.
                # This is a sum of 4 rank-1 tensors, so rank <= 4.
                # Is rank exactly 4? For 2x2*2x1, yes, I believe rank is 4.
                
                # So 2x2*2x7 has rank 7*4 = 28? No, that's schoolbook.
                # 2x2*2x7: sum over 7 columns, each contributing rank 4, total rank 28.
                # But maybe we can share? No, the columns are independent.
                
                # So my rank-28 estimate was actually schoolbook for the split version.
                # Original schoolbook: 2*4*7 = 56.
                # Split m into 2+2: 2 * (2*2*7) = 2 * 28 = 56. Same!
                
                # I made an error. Splitting m doesn't reduce the count.
                
                # Let me go back to basics. The schoolbook rank for m x n * n x p is m*n*p.
                # For 2x4x7, that's 56.
                
                # To beat 56, we need a non-trivial decomposition.
                # Strassen works for 2x2x2 (rank 7 vs 8).
                # For 2x4x7, we can try to embed Strassen.
                
                # Split m=4 into 2+2, p=7 into 2+2+2+1:
                # Subproblems: 2 * (2x2x2 + 2x2x2 + 2x2x1) = 2 * (8 + 8 + 4) = 40 schoolbook.
                # With Strassen on the 2x2x2 parts: 2 * (7 + 7 + 4) = 36.
                
                # Let me implement this.
                pass
    
    # Let me restart with a cleaner implementation.
    
    # Strategy: Split m=4 into 2+2, p=7 into 2+2+2+1.
    # This gives 2 * 4 = 8 subproblems:
    #   For each of the 2 m-blocks and 4 p-blocks:
    #     2x2x2 (three times per m-block) and 2x2x1 (once per m-block)
    #   Total: 2 * (3 * 2x2x2 + 1 * 2x2x1)
    #   Schoolbook: 2 * (3*8 + 1*4) = 2 * 28 = 56
    #   With Strassen: 2 * (3*7 + 1*4) = 2 * 25 = 50
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 28 rows
    W = [[0] * rank for _ in range(p * n)]  # 14 rows
    
    col = 0
    for m_block in range(2):  # 0 or 1, corresponding to columns 0-1 or 2-3 of A
        for p_block in range(4):  # 0,1,2 are 2-wide, 3 is 1-wide
            p_start = p_block * 2 if p_block < 3 else 6
            p_width = 2 if p_block < 3 else 1
            
            if p_width == 2:
                # 2x2x2 Strassen: 7 multiplications
                # A_block = A[:, m_block*2 : m_block*2+2]
                # B_block = B[m_block*2 : m_block*2+2, p_start : p_start+2]
                # C_block = C[:, p_start : p_start+2]
                
                # Strassen:
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
                
                # U indices: A[i, j] where i in {0,1}, j in {m_block*2, m_block*2+1}
                # V indices: B[j, k] where j in {m_block*2, m_block*2+1}, k in {p_start, p_start+1}
                # W indices: C[k, i] where i in {0,1}, k in {p_start, p_start+1}
                
                # M1: (A00 + A11) * (B00 + B11)
                U[m_block*2 + 0][col] = 1  # A[0, m_block*2]
                U[m_block*2 + 1][col] = 1  # A[1, m_block*2+1]
                V[m_block*2 + 0][col] = 1  # B[m_block*2, p_start]
                V[m_block*2 + 1][col] = 1  # B[m_block*2+1, p_start+1]
                W[p_start * n + 0][col] = 1   # C[p_start, 0]
                W[(p_start+1) * n + 0][col] = -1  # C[p_start+1, 0]
                W[p_start * n + 1][col] = 1   # C[p_start, 1]
                W[(p_start+1) * n + 1][col] = -1  # C[p_start+1, 1]
                col += 1
                
                # M2: (A10 + A11) * B00
                U[m_block*2 + 0][col] = 1  # A[1, m_block*2]
                U[m_block*2 + 1][col] = 1  # A[1, m_block*2+1]
                V[m_block*2 + 0][col] = 1  # B[m_block*2, p_start]
                W[p_start * n + 1][col] = 1   # C[p_start, 1]
                W[(p_start+1) * n + 1][col] = 1  # C[p_start+1, 1]
                col += 1
                
                # M3: A00 * (B01 - B11)
                U[m_block*2 + 0][col] = 1  # A[0, m_block*2]
                V[m_block*2 + 0][col] = 1  # B[m_block*2, p_start+1]
                V[m_block*2 + 1][col] = -1 # B[m_block*2+1, p_start+1]
                W[p_start * n + 0][col] = 1   # C[p_start, 0]
                W[(p_start+1) * n + 0][col] = 1  # C[p_start+1, 0]
                col += 1
                
                # M4: A11 * (B10 - B00)
                U[m_block*2 + 1][col] = 1  # A[1, m_block*2+1]
                V[m_block*2 + 1][col] = 1  # B[m_block*2+1, p_start]
                V[m_block*2 + 0][col] = -1 # B[m_block*2, p_start]
                W[p_start * n + 0][col] = 1   # C[p_start, 0]
                W[(p_start+1) * n + 0][col] = 1  # C[p_start+1, 0]
                W[p_start * n + 1][col] = 1   # C[p_start, 1]
                W[(p_start+1) * n + 1][col] = 1  # C[p_start+1, 1]
                col += 1
                
                # M5: (A00 + A01) * B11
                U[m_block*2 + 0][col] = 1  # A[0, m_block*2]
                U[m_block*2 + 1][col] = 1  # A[0, m_block*2+1]
                V[m_block*2 + 1][col] = 1  # B[m_block*2+1, p_start+1]
                W[p_start * n + 0][col] = -1  # C[p_start, 0]
                W[(p_start+1) * n + 0][col] = -1  # C[p_start+1, 0]
                W[p_start * n + 1][col] = 1   # C[p_start, 1]
                W[(p_start+1) * n + 1][col] = 1  # C[p_start+1, 1]
                col += 1
                
                # M6: (A10 - A00) * (B00 + B01)
                U[m_block*2 + 0][col] = -1 # A[1, m_block*2]
                U[m_block*2 + 1][col] = 1  # A[0, m_block*2]
                V[m_block*2 + 0][col] = 1  # B[m_block*2, p_start]
                V[m_block*2 + 0][col] += 1 # B[m_block*2, p_start+1]
                W[(p_start+1) * n + 1][col] = 1  # C[p_start+1, 1]
                col += 1
                
                # M7: (A01 - A11) * (B10 + B11)
                U[m_block*2 + 1][col] = 1  # A[0, m_block*2+1]
                U[m_block*2 + 1][col] -= 1 # A[1, m_block*2+1]
                V[m_block*2 + 1][col] = 1  # B[m_block*2+1, p_start]
                V[m_block*2 + 1][col] += 1 # B[m_block*2+1, p_start+1]
                W[p_start * n + 0][col] = 1   # C[p_start, 0]
                W[(p_start+1) * n + 0][col] = 1  # C[p_start+1, 0]
                col += 1
                
            else:
                # 2x2x1 schoolbook: 4 multiplications
                # A_block = A[:, m_block*2 : m_block*2+2]
                # B_col = B[m_block*2 : m_block*2+2, p_start]
                # C_col = C[:, p_start]
                
                for a_row in range(2):
                    for b_row in range(2):
                        U[a_row * m + m_block*2 + b_row][col] = 1
                        V[(m_block*2 + b_row) * p + p_start][col] = 1
                        W[p_start * n + a_row][col] = 1
                        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
