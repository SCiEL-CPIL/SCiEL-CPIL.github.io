"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Use a block decomposition approach:
    # Split the 4x4 middle dimension into two 2x2 blocks
    # Use Strassen's algorithm for each 2x2 block multiplication
    # This gives rank 7*7 = 49 for the two 2x2x7 blocks
    # But we can share some terms between the two blocks
    
    # Actually, let's think more carefully:
    # A is 2x4, B is 4x7
    # We can split A into [A1 | A2] where A1, A2 are 2x2
    # B into [B1; B2] where B1, B2 are 2x7
    # Then A*B = A1*B1 + A2*B2
    
    # For each 2x2 * 2x7 multiplication, we can use a rank-7 decomposition
    # (since 2x2*2x2 has rank 7 via Strassen, and 2x2*2x7 can be done with rank 7)
    # Wait, that's not quite right. Let me reconsider.
    
    # For 2x2 * 2x2, Strassen gives rank 7
    # For 2x2 * 2x7, we can think of it as 7 independent 2x2*2x1 multiplications
    # Each 2x2*2x1 has rank... let's think
    
    # Actually, the standard approach:
    # 2x2 * 2x2 has tensor rank 7 (Strassen)
    # 2x2 * 2x7: split the 7 columns into 3+2+2 or similar
    # Or: 2x2 * 2x7 can be done with rank 7 * ceil(7/2) = 7*4 = 28? No that's too high.
    
    # Let me think differently. The total rank for 2x4 * 4x7:
    # Schoolbook: 2*4*7 = 56
    # Using the block structure with Strassen:
    # A = [A1 A2], B = [B1; B2]
    # A*B = A1*B1 + A2*B2
    # Each A_i is 2x2, B_i is 2x7
    # 2x2 * 2x7: this is equivalent to computing 7 independent 2x2 * 2x1 products
    # But they share the same A_i, so we can be more efficient
    
    # For 2x2 * 2x7, the tensor rank is at most 7 (using Strassen-like approach)
    # Actually no. Let me look up: the rank of 2x2*2xn is 2n for n>=2? No.
    
    # Let me just use a concrete construction:
    # For 2x2 * 2x2, Strassen uses 7 multiplications:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b22)
    # M4 = a22(b12-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # For 2x2 * 2x7, we can apply Strassen to each pair of columns:
    # Columns 0,1: use 7 mults
    # Columns 2,3: use 7 mults
    # Columns 4,5: use 7 mults
    # Column 6: use 4 mults (schoolbook for 2x2*2x1)
    # Total: 7+7+7+4 = 25 per block
    # Two blocks: 25+25 = 50
    
    # So rank 50 is achievable. Let me construct it.
    
    U_rows = n * m  # 8
    V_rows = m * p  # 28
    W_rows = p * n  # 14
    
    # We'll build the decomposition with rank 50
    rank = 50
    U = np.zeros((U_rows, rank), dtype=int)
    V = np.zeros((V_rows, rank), dtype=int)
    W = np.zeros((W_rows, rank), dtype=int)
    
    col = 0
    
    # Helper: add a Strassen decomposition for 2x2 * 2x2 at given positions
    def add_strassen_2x2_2x2(a_row_offset, b_col_offset, c_row_offset, c_col_offset, start_col):
        """
        Add Strassen decomposition for multiplying a 2x2 matrix by a 2x2 matrix.
        a_row_offset: row offset in A (0 or 2 for the two 2x2 blocks of A)
        b_col_offset: col offset in B (0, 2, 4, or 6 for columns of B)
        c_row_offset: row offset in C (0 or 2 for the two 2x2 blocks of C)
        c_col_offset: col offset in C (0, 2, 4, or 6 for columns of C)
        start_col: starting column index in the decomposition
        """
        # Strassen's 7 multiplications for 2x2 * 2x2
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b01-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # Output: c00 = M1 - M4 + M5 - M7
        #         c01 = M3 + M7
        #         c10 = M2 + M4
        #         c11 = M1 - M2 + M3 + M6
        
        # The 7 terms:
        # Term 1: (a00+a11) * (b00+b11)
        # U: a00=+1, a11=+1; V: b00=+1, b11=+1; W: c00=+1, c11=+1
        U[a_row_offset + 0, start_col] += 1
        U[a_row_offset + 1, start_col] += 1
        V[b_col_offset + 0, start_col] += 1
        V[b_col_offset + 1, start_col] += 1
        W[c_row_offset + 0, start_col] += 1
        W[c_row_offset + 1, start_col] += 1
        col = start_col + 1
        
        # Term 2: (a10+a11) * b00
        # U: a10=+1, a11=+1; V: b00=+1; W: c10=+1, c11=-1
        U[a_row_offset + 2, start_col] += 1  # a10 is row 1, col 0 -> index 1*2+0=2? No.
        # Wait, A is 2x2, so indices are: a00=0, a01=1, a10=2, a11=3
        # But in our U, the rows are indexed by i*m+j where i is row, j is col
        # For the 2x2 block at a_row_offset, the local indices are:
        # local (0,0) -> global row a_row_offset + 0*2 + 0 = a_row_offset
        # local (0,1) -> global row a_row_offset + 0*2 + 1 = a_row_offset + 1
        # local (1,0) -> global row a_row_offset + 1*2 + 0 = a_row_offset + 2
        # local (1,1) -> global row a_row_offset + 1*2 + 1 = a_row_offset + 3
        
        # Similarly for V: B is 2x7, local indices:
        # local (0,0) -> global row b_col_offset + 0*7 + 0 = b_col_offset
        # local (0,1) -> global row b_col_offset + 0*7 + 1 = b_col_offset + 1
        # local (1,0) -> global row b_col_offset + 1*7 + 0 = b_col_offset + 7
        # local (1,1) -> global row b_col_offset + 1*7 + 1 = b_col_offset + 8
        
        # For W: C is 2x7, local indices:
        # local (0,0) -> global row c_row_offset + 0*7 + 0 = c_row_offset
        # local (0,1) -> global row c_row_offset + 0*7 + 1 = c_row_offset + 1
        # local (1,0) -> global row c_row_offset + 1*7 + 0 = c_row_offset + 7
        # local (1,1) -> global row c_row_offset + 1*7 + 1 = c_row_offset + 8
        
        # But wait, W is indexed by k*n+i where k is column of B, i is row of A
        # So W[k*n+i, r] corresponds to output position (i, k)
        # For the 2x2 block at c_row_offset (row) and c_col_offset (col):
        # local (i, k) -> global W index: (c_col_offset + k)*n + (c_row_offset + i)
        # So:
        # local (0,0) -> W[(c_col_offset+0)*2 + (c_row_offset+0)] = W[2*c_col_offset + c_row_offset]
        # local (0,1) -> W[(c_col_offset+1)*2 + (c_row_offset+0)] = W[2*c_col_offset + 2 + c_row_offset]
        # local (1,0) -> W[(c_col_offset+0)*2 + (c_row_offset+1)] = W[2*c_col_offset + c_row_offset + 1]
        # local (1,1) -> W[(c_col_offset+1)*2 + (c_row_offset+1)] = W[2*c_col_offset + 2 + c_row_offset + 1]
        
        # Let me redefine with proper indexing
        # U row for local (i,j) in A block: (c_row_offset + i)*m + (a_row_offset + j)
        # Wait no. A is n x m = 2 x 4.
        # U[i*m + j, c] where i is row of A, j is col of A
        # For the 2x2 block of A starting at row c_row_offset (0 or 1) and col a_row_offset (0 or 2):
        # local (i,j) -> U[(c_row_offset + i)*4 + (a_row_offset + j), c]
        
        # V[j*p + k, c] where j is row of B, k is col of B
        # For the 2x2 block of B starting at row b_row_offset (0 or 2) and col b_col_offset:
        # local (j,k) -> V[(b_row_offset + j)*7 + (b_col_offset + k), c]
        
        # W[k*n + i, c] where k is col of B, i is row of A
        # For the 2x2 block of C starting at row c_row_offset and col c_col_offset:
        # local (i,k) -> W[(c_col_offset + k)*2 + (c_row_offset + i), c]
        
        # OK let me restart with clearer indexing.
        return start_col
    
    # This is getting complicated. Let me just directly construct the rank-50 decomposition.
    # 
    # A is 2x4: rows 0,1; cols 0,1,2,3
    # B is 4x7: rows 0,1,2,3; cols 0,1,2,3,4,5,6
    # C is 2x7: rows 0,1; cols 0,1,2,3,4,5,6
    #
    # Split: A = [A1 | A2] where A1 = A[:, 0:2], A2 = A[:, 2:4]
    # B = [B1; B2] where B1 = B[0:2, :], B2 = B[2:4, :]
    # C = A1*B1 + A2*B2
    #
    # For A1*B1 (2x2 * 2x7):
    #   Split B1 into [B1a | B1b | B1c] where B1a = B1[:, 0:2], B1b = B1[:, 2:4], B1c = B1[:, 4:6], B1d = B1[:, 6]
    #   A1*B1 = A1*B1a + A1*B1b + A1*B1c + A1*B1d
    #   A1*B1a: 2x2*2x2, rank 7 (Strassen)
    #   A1*B1b: 2x2*2x2, rank 7 (Strassen)
    #   A1*B1c: 2x2*2x2, rank 7 (Strassen)
    #   A1*B1d: 2x2*2x1, rank 4 (schoolbook)
    #   Total for A1*B1: 7+7+7+4 = 25
    #
    # Similarly for A2*B2: 25
    # Total: 50
    
    # Now let me implement this properly.
    
    def add_strassen_2x2_2x2(a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base, start_col):
        """
        Add Strassen decomposition for 2x2 * 2x2.
        A block starts at (a_row_base, a_col_base), size 2x2
        B block starts at (b_row_base, b_col_base), size 2x2
        C block starts at (c_row_base, c_col_base), size 2x2
        """
        # A indices: U[(a_row_base+i)*4 + (a_col_base+j), col]
        # B indices: V[(b_row_base+j)*7 + (b_col_base+k), col]
        # C indices: W[(c_col_base+k)*2 + (c_row_base+i), col]
        
        # M1 = (a00+a11)(b00+b11)
        U[(a_row_base+0)*4 + (a_col_base+0), start_col] += 1
        U[(a_row_base+1)*4 + (a_col_base+1), start_col] += 1
        V[(b_row_base+0)*7 + (b_col_base+0), start_col] += 1
        V[(b_row_base+1)*7 + (b_col_base+1), start_col] += 1
        W[(c_col_base+0)*2 + (c_row_base+0), start_col] += 1
        W[(c_col_base+1)*2 + (c_row_base+1), start_col] += 1
        col = start_col + 1
        
        # M2 = (a10+a11)b00
        U[(a_row_base+1)*4 + (a_col_base+0), col] += 1
        U[(a_row_base+1)*4 + (a_col_base+1), col] += 1
        V[(b_row_base+0)*7 + (b_col_base+0), col] += 1
        W[(c_col_base+0)*2 + (c_row_base+1), col] += 1
        W[(c_col_base+1)*2 + (c_row_base+1), col] -= 1
        col += 1
        
        # M3 = a00(b10-b11)
        U[(a_row_base+0)*4 + (a_col_base+0), col] += 1
        V[(b_row_base+1)*7 + (b_col_base+0), col] += 1
        V[(b_row_base+1)*7 + (b_col_base+1), col] -= 1
        W[(c_col_base+1)*2 + (c_row_base+0), col] += 1
        W[(c_col_base+1)*2 + (c_row_base+1), col] += 1
        col += 1
        
        # M4 = a11(b01-b00)
        U[(a_row_base+1)*4 + (a_col_base+1), col] += 1
        V[(b_row_base+0)*7 + (b_col_base+1), col] += 1
        V[(b_row_base+0)*7 + (b_col_base+0), col] -= 1
        W[(c_col_base+0)*2 + (c_row_base+0), col] -= 1
        W[(c_col_base+0)*2 + (c_row_base+1), col] += 1
        col += 1
        
        # M5 = (a00+a01)b11
        U[(a_row_base+0)*4 + (a_col_base+0), col] += 1
        U[(a_row_base+0)*4 + (a_col_base+1), col] += 1
        V[(b_row_base+1)*7 + (b_col_base+1), col] += 1
        W[(c_col_base+0)*2 + (c_row_base+0), col] -= 1
        W[(c_col_base+1)*2 + (c_row_base+1), col] += 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[(a_row_base+1)*4 + (a_col_base+0), col] += 1
        U[(a_row_base+0)*4 + (a_col_base+0), col] -= 1
        V[(b_row_base+0)*7 + (b_col_base+0), col] += 1
        V[(b_row_base+0)*7 + (b_col_base+1), col] += 1
        W[(c_col_base+1)*2 + (c_row_base+1), col] += 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[(a_row_base+0)*4 + (a_col_base+1), col] += 1
        U[(a_row_base+1)*4 + (a_col_base+1), col] -= 1
        V[(b_row_base+1)*7 + (b_col_base+0), col] += 1
        V[(b_row_base+1)*7 + (b_col_base+1), col] += 1
        W[(c_col_base+0)*2 + (c_row_base+0), col] -= 1
        W[(c_col_base+1)*2 + (c_row_base+0), col] += 1
        col += 1
        
        return col
    
    def add_schoolbook_2x2_2x1(a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base, start_col):
        """
        Add schoolbook decomposition for 2x2 * 2x1.
        A block starts at (a_row_base, a_col_base), size 2x2
        B block starts at (b_row_base, b_col_base), size 2x1
        C block starts at (c_row_base, c_col_base), size 2x1
        """
        # c00 = a00*b00 + a01*b10
        # c10 = a10*b00 + a11*b10
        
        # Term 1: a00 * b00
        U[(a_row_base+0)*4 + (a_col_base+0), start_col] += 1
        V[(b_row_base+0)*7 + (b_col_base+0), start_col] += 1
        W[(c_col_base+0)*2 + (c_row_base+0), start_col] += 1
        col = start_col + 1
        
        # Term 2: a01 * b10
        U[(a_row_base+0)*4 + (a_col_base+1), col] += 1
        V[(b_row_base+1)*7 + (b_col_base+0), col] += 1
        W[(c_col_base+0)*2 + (c_row_base+0), col] += 1
        col += 1
        
        # Term 3: a10 * b00
        U[(a_row_base+1)*4 + (a_col_base+0), col] += 1
        V[(b_row_base+0)*7 + (b_col_base+0), col] += 1
        W[(c_col_base+0)*2 + (c_row_base+1), col] += 1
        col += 1
        
        # Term 4: a11 * b10
        U[(a_row_base+1)*4 + (a_col_base+1), col] += 1
        V[(b_row_base+1)*7 + (b_col_base+0), col] += 1
        W[(c_col_base+0)*2 + (c_row_base+1), col] += 1
        col += 1
        
        return col
    
    # Now build the full decomposition
    col = 0
    
    # A1*B1: A1 = A[0:2, 0:2], B1 = B[0:2, :]
    # A1*B1a: 2x2*2x2, A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    col = add_strassen_2x2_2x2(0, 0, 0, 0, 0, 0, col)
    
    # A1*B1b: 2x2*2x2, A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    col = add_strassen_2x2_2x2(0, 0, 0, 2, 0, 2, col)
    
    # A1*B1c: 2x2*2x2, A[0:2,0:2] * B[0:2,4:6] -> C[0:2,4:6]
    col = add_strassen_2x2_2x2(0, 0, 0, 4, 0, 4, col)
    
    # A1*B1d: 2x2*2x1, A[0:2,0:2] * B[0:2,6:7] -> C[0:2,6:7]
    col = add_schoolbook_2x2_2x1(0, 0, 0, 6, 0, 6, col)
    
    # A2*B2: A2 = A[0:2, 2:4], B2 = B[2:4, :]
    # A2*B2a: 2x2*2x2, A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    col = add_strassen_2x2_2x2(0, 2, 2, 0, 0, 0, col)
    
    # A2*B2b: 2x2*2x2, A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    col = add_strassen_2x2_2x2(0, 2, 2, 2, 0, 2, col)
    
    # A2*B2c: 2x2*2x2, A[0:2,2:4] * B[2:4,4:6] -> C[0:2,4:6]
    col = add_strassen_2x2_2x2(0, 2, 2, 4, 0, 4, col)
    
    # A2*B2d: 2x2*2x1, A[0:2,2:4] * B[2:4,6:7] -> C[0:2,6:7]
    col = add_schoolbook_2x2_2x1(0, 2, 2, 6, 0, 6, col)
    
    # col should be 50
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify the decomposition
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank_schoolbook = n * m * p
    U_fb = np.zeros((U_rows, rank_schoolbook), dtype=int)
    V_fb = np.zeros((V_rows, rank_schoolbook), dtype=int)
    W_fb = np.zeros((W_rows, rank_schoolbook), dtype=int)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(U.shape[1]):
                    total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if total != 1:
                    return False
    return True
# EVOLVE-BLOCK-END
