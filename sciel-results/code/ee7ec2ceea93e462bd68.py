"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a low-rank exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns dict with U, V, W matrices.
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    deadline = time.time() + 110
    
    # Strategy: Use Strassen-like decomposition for 2x2x2 sub-blocks
    # Split A (3x3) into:
    #   A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # Split B (3x8) into:
    #   B = [[B11, B12], [B21, B22]] where B11 is 2x8, B12... 
    # Actually, let's split B by rows: B has rows indexed by j=0,1,2
    #   B = [B_row0; B_row1; B_row2] where each is 1x8
    # Or split by columns: B = [B_left (3x4), B_right (3x4)]
    
    # Let me use a cleaner approach:
    # Split the middle dimension m=3 into 2+1
    # A (3x3) -> columns: A[:, :2] (3x2) and A[:, 2] (3x1)
    # B (3x8) -> rows: B[:2, :] (2x8) and B[2, :] (1x8)
    
    # C = A[:, :2] @ B[:2, :] + A[:, 2] @ B[2, :]
    # First term: (3x2) @ (2x8) = 3x8, schoolbook rank = 3*2*8 = 48
    # Second term: (3x1) @ (1x8) = 3x8, schoolbook rank = 3*1*8 = 24
    # Total: 72 (same as before, no improvement)
    
    # To improve, I need to use low-rank decompositions for the sub-products.
    # For (3x2) @ (2x8): this is like 2 independent 3x1 @ 1x8 outer products summed.
    # Actually, (3x2)@(2x8) = sum_j A[:,j] * B[j,:] for j=0,1
    # This is 2 outer products of 3-vectors and 8-vectors, so rank = 2*3*8/... no.
    # Each term A[:,j] * B[j,:] is a rank-1 3x8 matrix, so 2 such terms have rank <= 2.
    # But in tensor decomposition terms, the rank is 3*2*8 = 48 for schoolbook.
    
    # Let me try a different splitting. Use Strassen for 2x2x2 within.
    
    # New approach: Split n=3 into 2+1, p=8 into 4+4
    # A (3x3), B (3x8) -> B = [B1 (3x4), B2 (3x4)]
    # C = A@B1 (3x4) and A@B2 (3x4)
    
    # For A@B1: (3x3)@(3x4)
    # Split A into rows: A[:2,:] (2x3) and A[2,:] (1x3)
    # Split B1 into cols: B1[:,:2] (3x2), B1[:,2:] (3x2)
    # This is getting complicated.
    
    # Let me try the most direct approach: use known Strassen for 2x2x2
    # and build up to 3x3x8.
    
    # Concrete construction using Strassen for 2x2x2:
    # 
    # Write A = [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]
    # Write B = [[b11, b12, b13, b14, b15, b16, b17, b18],
    #            [b21, b22, b23, b24, b25, b26, b27, b28],
    #            [b31, b32, b33, b34, b35, b36, b37, b38]]
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Let's split j into {0,1} and {2}:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # The first two terms involve A[:, :2] (3x2) and B[:2, :] (2x8)
    # This is a (3x2)@(2x8) product.
    #
    # For (3x2)@(2x8), we can further split:
    # Let A2 = A[:, :2] (3x2), B2 = B[:2, :] (2x8)
    # C2 = A2 @ B2 (3x8)
    #
    # Split A2 into columns: A2 = [a_col0, a_col1] where each is 3x1
    # Split B2 into rows: B2 = [b_row0; b_row1] where each is 1x8
    # C2 = a_col0 * b_row0 + a_col1 * b_row1
    #
    # Each a_col * b_row is a 3x8 matrix of rank 1.
    # In tensor decomposition: for term a_col0 * b_row0:
    #   U[i*3+j, r] = a_col0[i] for j=0
    #   V[j*8+k, r] = b_row0[k] for j=0
    #   W[k*3+i, r] = 1
    #
    # This uses 3*8 = 24 terms per outer product, so 48 for both.
    # Plus 3*8 = 24 for the A[:,2]*B[2,:] term.
    # Total: 72. Still schoolbook.
    
    # I need to find actual rank reduction. Let me use Strassen properly.
    
    # Strassen for 2x2x2:
    # Given P (2x2) and Q (2x2), compute R = P@Q with 7 multiplications:
    # m1 = (P00+P11)*(Q00+Q11)
    # m2 = (P10+P11)*Q00
    # m3 = P00*(Q01-Q11)
    # m4 = P11*(Q10-Q00)
    # m5 = (P00+P01)*Q11
    # m6 = (P10-P00)*(Q00+Q01)
    # m7 = (P01-P11)*(Q10+Q11)
    # R00 = m1 + m4 - m5 + m7
    # R01 = m3 + m5
    # R10 = m2 + m4
    # R11 = m1 - m2 + m3 + m6
    
    # Now let me build a 3x3x8 decomposition using Strassen for 2x2x2 sub-blocks.
    
    # Split A (3x3) as:
    # A = [[A11 (2x2), A12 (2x1)], [A21 (1x2), A22 (1x1)]]
    # Split B (3x8) as:
    # B = [[B11 (2x4), B12 (2x4)], [B21 (1x4), B22 (1x4)]]
    # (splitting the 8 columns into 4+4)
    
    # C = A@B (3x8)
    # C = [[C11 (2x4), C12 (2x4)], [C21 (1x4), C22 (1x4)]]
    
    # C11 = A11@B11 + A12@B21
    # C12 = A11@B12 + A12@B22
    # C21 = A21@B11 + A22@B21
    # C22 = A21@B12 + A22@B22
    
    # For A11@B11: (2x2)@(2x4)
    # Split B11 into B11a (2x2) and B11b (2x2)
    # A11@B11a: (2x2)@(2x2) -> use Strassen, rank 7
    # A11@B11b: (2x2)@(2x2) -> use Strassen, rank 7
    # Total for A11@B11: 14
    
    # Similarly for A11@B12: 14
    # A12@B21: (2x1)@(1x4) -> schoolbook rank 2*1*4 = 8
    # A12@B22: (2x1)@(1x4) -> 8
    # A21@B11: (1x2)@(2x4) -> 1*2*4 = 8
    # A21@B12: (1x2)@(2x4) -> 8
    # A22@B21: (1x1)@(1x4) -> 1*1*4 = 4
    # A22@B22: (1x1)@(1x4) -> 4
    
    # Total: 14 + 14 + 8 + 8 + 8 + 8 + 4 + 4 = 68
    
    # That's an improvement from 72 to 68!
    
    # Let me implement this.
    
    # Define the Strassen decomposition for 2x2x2:
    # For P (2x2) @ Q (2x2) = R (2x2):
    # The 7 multiplications are:
    # m1 = (P00+P11)*(Q00+Q11)
    # m2 = (P10+P11)*Q00
    # m3 = P00*(Q01-Q11)
    # m4 = P11*(Q10-Q00)
    # m5 = (P00+P01)*Q11
    # m6 = (P10-P00)*(Q00+Q01)
    # m7 = (P01-P11)*(Q10+Q11)
    # R00 = m1 + m4 - m5 + m7
    # R01 = m3 + m5
    # R10 = m2 + m4
    # R11 = m1 - m2 + m3 + m6
    
    # In tensor decomposition form, each multiplication mr corresponds to:
    # U[i, r] * V[j, r] * W[k, r] where i,j are input indices, k is output index
    
    # For 2x2x2, the indices are:
    # U: 2x2 = 4 entries (i in {0,1}, j in {0,1})
    # V: 2x2 = 4 entries (j in {0,1}, k in {0,1})
    # W: 2x2 = 4 entries (k in {0,1}, i in {0,1})
    
    # Let me define the Strassen factors for 2x2x2:
    # U_strassen (4x7), V_strassen (4x7), W_strassen (4x7)
    
    # For each of the 7 multiplications, define the U, V, W vectors:
    
    def _strassen_2x2_2x2():
        """Return U, V, W for 2x2 x 2x2 = 2x2 using Strassen (rank 7)."""
        # U: (n*m, rank) = (4, 7)
        # V: (m*p, rank) = (4, 7)  
        # W: (p*n, rank) = (4, 7)
        # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] with n=m=p=2
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # m1 = (P00+P11)*(Q00+Q11)
        # P00: i=0,j=0 -> U[0*2+0, 0] = U[0,0]
        # P11: i=1,j=1 -> U[1*2+1, 0] = U[3,0]
        # Q00: j=0,k=0 -> V[0*2+0, 0] = V[0,0]
        # Q11: j=1,k=1 -> V[1*2+1, 0] = V[3,0]
        U[0, 0] = 1  # P00
        U[3, 0] = 1  # P11
        V[0, 0] = 1  # Q00
        V[3, 0] = 1  # Q11
        # Output: R00 += m1, R01 += 0, R10 += 0, R11 += m1
        # R00: i=0,k=0 -> W[0*2+0, 0] = W[0,0]
        # R11: i=1,k=1 -> W[1*2+1, 0] = W[3,0]
        W[0, 0] = 1  # R00
        W[3, 0] = 1  # R11
        
        # m2 = (P10+P11)*Q00
        # P10: i=1,j=0 -> U[1*2+0, 1] = U[2,1]
        # P11: i=1,j=1 -> U[3,1]
        # Q00: j=0,k=0 -> V[0,1]
        U[2, 1] = 1  # P10
        U[3, 1] = 1  # P11
        V[0, 1] = 1  # Q00
        # Output: R10 += m2
        # R10: i=1,k=0 -> W[0*2+1, 1] = W[1,1]
        W[1, 1] = 1  # R10
        
        # m3 = P00*(Q01-Q11)
        # P00: i=0,j=0 -> U[0,2]
        # Q01: j=0,k=1 -> V[0*2+1, 2] = V[1,2]
        # Q11: j=1,k=1 -> V[3,2]
        U[0, 2] = 1  # P00
        V[1, 2] = 1  # Q01
        V[3, 2] = -1  # -Q11
        # Output: R01 += m3
        # R01: i=0,k=1 -> W[1*2+0, 2] = W[2,2]
        W[2, 2] = 1  # R01
        
        # m4 = P11*(Q10-Q00)
        # P11: i=1,j=1 -> U[3,3]
        # Q10: j=1,k=0 -> V[1*2+0, 3] = V[2,3]
        # Q00: j=0,k=0 -> V[0,3]
        U[3, 3] = 1  # P11
        V[2, 3] = 1  # Q10
        V[0, 3] = -1  # -Q00
        # Output: R00 += m4, R10 += m4
        # R00: i=0,k=0 -> W[0,3]
        # R10: i=1,k=0 -> W[1,3]
        W[0, 3] = 1  # R00
        W[1, 3] = 1  # R10
        
        # m5 = (P00+P01)*Q11
        # P00: i=0,j=0 -> U[0,4]
        # P01: i=0,j=1 -> U[0*2+1, 4] = U[1,4]
        # Q11: j=1,k=1 -> V[3,4]
        U[0, 4] = 1  # P00
        U[1, 4] = 1  # P01
        V[3, 4] = 1  # Q11
        # Output: R01 += m5, R00 -= m5
        # R01: i=0,k=1 -> W[2,4]
        # R00: i=0,k=0 -> W[0,4]
        W[2, 4] = 1  # R01
        W[0, 4] = -1  # -R00
        
        # m6 = (P10-P00)*(Q00+Q01)
        # P10: i=1,j=0 -> U[2,5]
        # P00: i=0,j=0 -> U[0,5]
        # Q00: j=0,k=0 -> V[0,5]
        # Q01: j=0,k=1 -> V[1,5]
        U[2, 5] = 1  # P10
        U[0, 5] = -1  # -P00
        V[0, 5] = 1  # Q00
        V[1, 5] = 1  # Q01
        # Output: R11 += m6
        # R11: i=1,k=1 -> W[3,5]
        W[3, 5] = 1  # R11
        
        # m7 = (P01-P11)*(Q10+Q11)
        # P01: i=0,j=1 -> U[1,6]
        # P11: i=1,j=1 -> U[3,6]
        # Q10: j=1,k=0 -> V[2,6]
        # Q11: j=1,k=1 -> V[3,6]
        U[1, 6] = 1  # P01
        V[2, 6] = 1  # Q10
        V[3, 6] = 1  # Q11
        # P11: i=1,j=1 -> U[3,6] = -1
        U[3, 6] = -1  # -P11
        # Output: R00 += m7
        # R00: i=0,k=0 -> W[0,6]
        W[0, 6] = 1  # R00
        
        return U, V, W
    
    # Now build the full 3x3x8 decomposition.
    # 
    # A (3x3) indices: i in {0,1,2}, j in {0,1,2}
    # B (3x8) indices: j in {0,1,2}, k in {0,...,7}
    # C (3x8) indices: i in {0,1,2}, k in {0,...,7}
    #
    # Split:
    # A11 = A[0:2, 0:2] (2x2), A12 = A[0:2, 2] (2x1)
    # A21 = A[2, 0:2] (1x2), A22 = A[2, 2] (1x1)
    #
    # B split by columns into 4+4:
    # B1 = B[:, 0:4] (3x4), B2 = B[:, 4:8] (3x4)
    # B11 = B[0:2, 0:4] (2x4), B12 = B[0:2, 4:8] (2x4)
    # B21 = B[2, 0:4] (1x4), B22 = B[2, 4:8] (1x4)
    #
    # Further split B11 into B11a = B[0:2, 0:2] (2x2), B11b = B[0:2, 2:4] (2x2)
    # Similarly for B12: B12a = B[0:2, 4:6] (2x2), B12b = B[0:2, 6:8] (2x2)
    #
    # C11 = A11@B11 + A12@B21
    # C12 = A11@B12 + A12@B22
    # C21 = A21@B11 + A22@B21
    # C22 = A21@B12 + A22@B22
    #
    # C11[0:2, 0:2] = A11@B11a (2x2) -> Strassen rank 7
    # C11[0:2, 2:4] = A11@B11b (2x2) -> Strassen rank 7
    # C11[2, 0:2] = A21@B11a (1x2) -> schoolbook rank 1*2*2 = 4
    # C11[2, 2:4] = A21@B11b (1x2) -> 4
    # C11[0:2, 0:2] also gets A12@B21a where B21a = B[2, 0:2] (1x2)
    #   A12@B21a: (2x1)@(1x2) -> schoolbook rank 2*1*2 = 4
    # C11[0:2, 2:4] gets A12@B21b where B21b = B[2, 2:4] (1x2) -> 4
    #
    # Wait, I need to be more careful. Let me redefine:
    #
    # For the top-left 2x4 block of C (C[0:2, 0:4]):
    # C[0:2, 0:4] = A11@B[0:2, 0:4] + A12@B[2, 0:4]
    # = A11@B11 + A12@B21
    #
    # A11@B11: (2x2)@(2x4)
    # Split B11 into two 2x2 blocks: B11[:2] and B11[2:4]
    # A11@B11[:2]: (2x2)@(2x2) -> Strassen rank 7
    # A11@B11[2:4]: (2x2)@(2x2) -> Strassen rank 7
    # Total: 14
    #
    # A12@B21: (2x1)@(1x4) -> schoolbook rank 2*1*4 = 8
    #
    # Similarly for C[0:2, 4:8]: 14 + 8 = 22
    #
    # For C[2, 0:4]: A21@B[0:2, 0:4] + A22@B[2, 0:4]
    # A21@B[0:2, 0:4]: (1x2)@(2x4) -> schoolbook rank 1*2*4 = 8
    # A22@B[2, 0:4]: (1x1)@(1x4) -> schoolbook rank 1*1*4 = 4
    # Total: 12
    #
    # For C[2, 4:8]: 8 + 4 = 12
    #
    # Grand total: 22 + 22 + 12 + 12 = 68
    
    # Now let me implement this.
    
    # Total rank = 68
    total_rank = 68
    
    U = [[0] * total_rank for _ in range(n * m)]  # (9, 68)
    V = [[0] * total_rank for _ in range(m * p)]  # (24, 68)
    W = [[0] * total_rank for _ in range(p * n)]  # (24, 68)
    
    col = 0
    
    # Helper: add a Strassen 2x2x2 decomposition for A_sub @ B_sub = C_sub
    # where A_sub is 2x2, B_sub is 2x2, C_sub is 2x2
    # A_sub[i,j] corresponds to global A[i_off + i, j_off + j]
    # B_sub[j,k] corresponds to global B[j_off + j, k_off + k]
    # C_sub[i,k] corresponds to global C[i_off + i, k_off + k]
    
    def add_strassen(A_i_off, A_j_off, B_j_off, B_k_off, C_i_off, C_k_off):
        nonlocal col
        U_s, V_s, W_s = _strassen_2x2_2x2()
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[A_i_off * m + (A_j_off + j)][col] += U_s[i * 2 + j][r]
            for j in range(2):
                for k in range(2):
                    V[(B_j_off + j) * p + (B_k_off + k)][col] += V_s[j * 2 + k][r]
            for k in range(2):
                for i in range(2):
                    W[(C_k_off + k) * n + (C_i_off + i)][col] += W_s[k * 2 + i][r]
        col += 7
    
    # Helper: add schoolbook for (ni x mj) @ (mj x pk) = (ni x pk)
    def add_schoolbook(A_i_off, A_j_off, B_j_off, B_k_off, C_i_off, C_k_off, ni, mj, pk):
        nonlocal col
        for i in range(ni):
            for j in range(mj):
                for k in range(pk):
                    U[(A_i_off + i) * m + (A_j_off + j)][col] = 1
                    V[(B_j_off + j) * p + (B_k_off + k)][col] = 1
                    W[(C_k_off + k) * n + (C_i_off + i)][col] = 1
                    col += 1
    
    # C[0:2, 0:2] = A11@B[0:2, 0:2] (Strassen) + A12@B[2, 0:2] (schoolbook)
    add_strassen(0, 0, 0, 0, 0, 0)  # A[0:2,0:2] @ B[0:2,0:2] -> C[0:2,0:2]
    # A12 = A[0:2, 2], B21a = B[2, 0:2]
    add_schoolbook(0, 2, 2, 0, 0, 0, 2, 1, 2)  # A[0:2,2] @ B[2,0:2] -> C[0:2,0:2]
    
    # C[0:2, 2:4] = A11@B[0:2, 2:4] (Strassen) + A12@B[2, 2:4] (schoolbook)
    add_strassen(0, 0, 0, 2, 0, 2)  # A[0:2,0:2] @ B[0:2,2:4] -> C[0:2,2:4]
    add_schoolbook(0, 2, 2, 2, 0, 2, 2, 1, 2)  # A[0:2,2] @ B[2,2:4] -> C[0:2,2:4]
    
    # C[0:2, 4:6] = A11@B[0:2, 4:6] (Strassen) + A12@B[2, 4:6] (schoolbook)
    add_strassen(0, 0, 0, 4, 0, 4)
    add_schoolbook(0, 2, 2, 4, 0, 4, 2, 1, 2)
    
    # C[0:2, 6:8] = A11@B[0:2, 6:8] (Strassen) + A12@B[2, 6:8] (schoolbook)
    add_strassen(0, 0, 0, 6, 0, 6)
    add_schoolbook(0, 2, 2, 6, 0, 6, 2, 1, 2)
    
    # C[2, 0:2] = A21@B[0:2, 0:2] + A22@B[2, 0:2]
    add_schoolbook(2, 0, 0, 0, 2, 0, 1, 2, 2)  # A[2,0:2] @ B[0:2,0:2] -> C[2,0:2]
    add_schoolbook(2, 2, 2, 0, 2, 0, 1, 1, 2)  # A[2,2] @ B[2,0:2] -> C[2,0:2]
    
    # C[2, 2:4]
    add_schoolbook(2, 0, 0, 2, 2, 2, 1, 2, 2)
    add_schoolbook(2, 2, 2, 2, 2, 2, 1, 1, 2)
    
    # C[2, 4:6]
    add_schoolbook(2, 0, 0, 4, 2, 4, 1, 2, 2)
    add_schoolbook(2, 2, 2, 4, 2, 4, 1, 1, 2)
    
    # C[2, 6:8]
    add_schoolbook(2, 0, 0, 6, 2, 6, 1, 2, 2)
    add_schoolbook(2, 2, 2, 6, 2, 6, 1, 1, 2)
    
    # Verify col == total_rank
    assert col == total_rank, f"col={col}, expected {total_rank}"
    
    # Verify correctness
    def _verify(U, V, W, n, m, p):
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        schoolbook_rank = n * m * p
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
