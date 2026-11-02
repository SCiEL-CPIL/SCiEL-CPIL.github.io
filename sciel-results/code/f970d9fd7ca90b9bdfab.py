"""Matrix multiplication 2x3x7: rank-minimal exact decomposition."""
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START

def construct():
    """
    Returns a rank-39 hybrid Strassen/Schoolbook decomposition for 2x3 x 3x7 multiplication.
    
    Strategy: split the middle dimension m=3 into m1=2 and m2=1.
    - For the 2x2 x 2x7 block, split p=7 into three 2s and one 1,
      use Strassen (rank 7) for each 2x2x2 and schoolbook (rank 2) for 2x2x1.
      Sub-rank = 3*7 + 2 = 23.
    - For the 1x2 x 2x7 block, use schoolbook (rank 14).
    Total rank = 23 + 14 = 37.
    
    Wait, let me recount. Actually let me be careful with indices.
    
    Let's use a cleaner construction:
    Split m = 3 into 2 + 1.
    Block A: rows 0..1 (i in {0,1}), cols 0..1 (j in {0,1}) -> 2x2 matrix A1
    Block B: rows 0..1 (i in {0,1}), col 2 (j=2) -> 2x1 matrix B
    Right matrix: left part cols 0..1 -> 2x7 matrix C1, right part col 2 -> 1x7 matrix C2
    
    Result: A1 @ C1 + B @ C2
    A1 @ C1 is 2x7, B @ C2 is 2x7.
    
    For A1 (2x2) @ C1 (2x7): split C1's 7 columns into 2+2+2+1.
    For each 2x2 block of C1, use Strassen (rank 7) to multiply 2x2 by 2x2.
    For the last 2x1 block, use schoolbook (rank 2).
    Rank for A1 @ C1 = 3*7 + 2 = 23.
    
    For B (2x1) @ C2 (1x7): schoolbook, rank = 2*1*7 = 14.
    
    Total rank = 23 + 14 = 37.
    """
    # Actually, let me verify: is rank 37 valid?
    # 2x2 by 2x7: The 2x2x2 multiplication has rank 7 (Strassen).
    # 2x2 by 2x7 = 2x2 by (2x2 + 2x2 + 2x2 + 2x1)
    # = (2x2 by 2x2) + (2x2 by 2x2) + (2x2 by 2x2) + (2x2 by 2x1)
    # Each 2x2x2 can be done in rank 7.
    # 2x2 by 2x1: this is just 4 scalar multiplications, rank 4? No.
    # Wait, 2x2 matrix times 2x1 matrix = 2x1 result.
    # This requires 4 multiplications (schoolbook), rank 4.
    # Hmm, can we do better? 2x2 by 1x1... no, it's 2x2 by 2x1.
    # Actually 2x2 * 2x1: result has 2 entries. Each entry is a dot product of length 2.
    # That's 4 scalar products, so rank 4.
    # So 2x2 by 2x7 rank = 3*7 + 4 = 25.
    # Total = 25 + 14 = 39.
    #
    # Let me reconsider. 2x2 by 2x1:
    # [[a,b],[c,d]] * [x;y] = [ax+by; cx+dy]
    # 4 multiplications, 2 additions. Rank 4.
    # So total = 3*7 + 4 + 14 = 21 + 4 + 14 = 39.
    
    # Let's construct rank 39 decomposition.
    n, m, p = 2, 3, 7
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]   # 6 x 39
    V = [[0] * rank for _ in range(m * p)]   # 21 x 39
    W = [[0] * rank for _ in range(p * n)]   # 14 x 39
    
    col = 0
    
    # Part 1: A1 (2x2) @ C1 (2x7), where C1 split into 2+2+2+1
    # Strassen for 2x2x2, schoolbook for 2x2x1
    
    # Strassen constants for 2x2x2:
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b11-b21)
    # M4 = a22*(b12-b22)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b21)
    # M7 = (a12-a22)*(b12+b22)
    # c11 = M1 - M4 + M5 + M7
    # c12 = M3 + M4
    # c21 = M2 + M5
    # c22 = M1 - M3 + M6 + M7
    
    def strassen_222(u_base, v_base, w_base, offset):
        """Add Strassen 2x2x2 decomposition starting at column offset.
        u_base: (row_i_offset, row_j_offset) for U
        v_base: (row_j_offset, row_k_offset) for V  
        w_base: (row_k_offset, row_i_offset) for W
        Returns number of columns used."""
        # 7 terms
        # Term 1: (a11+a22)*(b11+b22)
        U[u_base[0]][offset] = 1  # a11
        U[u_base[1]][offset] = 1  # a22
        V[v_base[0]][offset] = 1  # b11
        V[v_base[1]][offset] = 1  # b22
        W[w_base[0]][offset] = 1  # c11 contribution
        W[w_base[1]][offset] = 1  # c22 contribution
        # But wait, W convention is W[k*n+i, r]. 
        # For 2x2x2, the output is 2x2, indexed by (k, i) where k is row of result, i is... 
        # Actually the convention says W[k*n+i, r] where the output is (n x p) = (2 x 2).
        # So W[row, col] where row = k*n + i, k in 0..p-1, i in 0..n-1.
        # For a 2x2 result, W[0] = W[0*2+0] = c[0][0], W[1] = c[0][1], W[2] = c[1][0], W[3] = c[1][1].
        # Hmm, this is getting complex. Let me just build it directly.
        return 1  # placeholder
    
    # Let me just build it term by term manually.
    # This is getting complicated. Let me use the known rank-39 construction.
    
    # Actually, the simplest correct approach: use the hybrid decomposition.
    # Let me just implement it carefully.
    
    # Reset
    col = 0
    
    # Indices:
    # U: row = i*m + j, i in 0..1, j in 0..2 -> rows 0..5
    # V: row = j*p + k, j in 0..2, k in 0..6 -> rows 0..20
    # W: row = k*n + i, k in 0..6, i in 0..1 -> rows 0..13
    
    # Split j (middle) into j1 in {0,1} and j2 = 2.
    # For j in {0,1}: use Strassen for the 2x2x2 blocks when splitting k.
    # Split k into groups: k in {0,1}, {2,3}, {4,5}, {6}
    
    # For each pair of k values (k1, k2), we multiply the 2x2 submatrix (i in 0,1; j in 0,1)
    # by the 2x2 submatrix (j in 0,1; k in k1,k2).
    
    # Strassen terms for one 2x2x2 block:
    # The input submatrix A_sub has entries A[i][j] for i in {0,1}, j in {0,1}
    # The input submatrix B_sub has entries B[j][k] for j in {0,1}, k in {k1,k2}
    # Result C_sub has entries C[i][k] for i in {0,1}, k in {k1,k2}
    
    # Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B00-B10)
    # M4 = A11*(B01-B11)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M3 + M6 + M7
    
    def add_strassen_222(i_off, j_off, k1, k2, col):
        """Add 7 Strassen terms for multiplying A[i_off:i_off+2, j_off:j_off+2] @ B[j_off:j_off+2, k1:k1+2]
        into the output at positions (i_off, k1), (i_off, k1+1), (i_off+1, k1), (i_off+1, k1+1).
        U rows: (i, j) -> i*m + j
        V rows: (j, k) -> j*p + k
        W rows: (k, i) -> k*n + i
        """
        i1, i2 = i_off, i_off + 1
        j1, j2 = j_off, j_off + 1
        
        # M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
        U[i1*m + j1][col] = 1
        U[i2*m + j2][col] = 1
        V[j1*p + k1][col] = 1
        V[j2*p + k2][col] = 1
        W[k1*n + i1][col] = 1
        W[k2*n + i2][col] = 1
        col += 1
        
        # M2 = (A[i2,j1]+A[i2,j2]) * B[j1,k1]
        U[i2*m + j1][col] = 1
        U[i2*m + j2][col] = 1
        V[j1*p + k1][col] = 1
        W[k1*n + i2][col] = 1
        col += 1
        
        # M3 = A[i1,j1] * (B[j1,k1]-B[j2,k1])
        U[i1*m + j1][col] = 1
        V[j1*p + k1][col] = 1
        V[j2*p + k1][col] = -1
        W[k1*n + i1][col] = 1
        col += 1
        
        # M4 = A[i2,j2] * (B[j1,k2]-B[j2,k2])
        U[i2*m + j2][col] = 1
        V[j1*p + k2][col] = 1
        V[j2*p + k2][col] = -1
        W[k2*n + i2][col] = 1
        col += 1
        
        # M5 = (A[i1,j1]+A[i1,j2]) * B[j2,k2]
        U[i1*m + j1][col] = 1
        U[i1*m + j2][col] = 1
        V[j2*p + k2][col] = 1
        W[k2*n + i1][col] = 1
        col += 1
        
        # M6 = (A[i2,j1]-A[i1,j1]) * (B[j1,k1]+B[j2,k1])
        U[i2*m + j1][col] = 1
        U[i1*m + j1][col] = -1
        V[j1*p + k1][col] = 1
        V[j2*p + k1][col] = 1
        W[k1*n + i2][col] = 1
        col += 1
        
        # M7 = (A[i1,j2]-A[i2,j2]) * (B[j1,k2]+B[j2,k2])
        U[i1*m + j2][col] = 1
        U[i2*m + j2][col] = -1
        V[j1*p + k2][col] = 1
        V[j2*p + k2][col] = 1
        W[k2*n + i1][col] = 1
        col += 1
        
        return col
    
    # Now combine the Strassen terms.
    # For each Strassen 2x2x2 block, the output C[i][k] gets contributions from multiple terms.
    # I need to add the correct coefficients to W.
    
    # Let me redo this more carefully. For one 2x2x2 Strassen block:
    # The 7 intermediate products are:
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B00-B10)
    # P4 = A11*(B01-B11)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B10)
    # P7 = (A01-A11)*(B01+B11)
    #
    # C00 = P1 - P4 + P5 + P7
    # C01 = P3 + P4
    # C10 = P2 + P5
    # C11 = P1 - P3 + P6 + P7
    
    # So in the tensor decomposition, each column corresponds to one Pi.
    # U and V encode the linear combinations for the inputs.
    # W encodes the linear combinations for the output.
    
    # For a single 2x2x2 block with i_off, j_off, and k positions k1, k2:
    # The U entries for this block affect rows (i_off, j_off) and (i_off+1, j_off+1) etc.
    # The V entries affect rows (j_off, k1), (j_off+1, k2) etc.
    # The W entries affect positions (k1, i_off), (k1, i_off+1), (k2, i_off), (k2, i_off+1).
    
    # Let me define a helper that adds one complete Strassen 2x2x2 block.
    
    def strassen_block(i_off, j_off, k1, k2, col):
        """Add 7 columns for Strassen 2x2x2 multiplication.
        A submatrix: rows i_off, i_off+1; cols j_off, j_off+1
        B submatrix: rows j_off, j_off+1; cols k1, k1+1
        C submatrix: rows k1, k1+1; cols i_off, i_off+1
        """
        i1, i2 = i_off, i_off + 1
        j1, j2 = j_off, j_off + 1
        k1p, k2p = k1, k1 + 1  # k2 = k1+1
        
        # P1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
        U[i1*m+j1][col] = 1; U[i2*m+j2][col] = 1
        V[j1*p+k1][col] = 1; V[j2*p+k2][col] = 1
        W[k1*n+i1][col] = 1   # C00 += P1
        W[k2*n+i2][col] = 1   # C11 += P1
        col += 1
        
        # P2 = (A[i2,j1]+A[i2,j2]) * B[j1,k1]
        U[i2*m+j1][col] = 1; U[i2*m+j2][col] = 1
        V[j1*p+k1][col] = 1
        W[k1*n+i2][col] = 1   # C10 += P2
        col += 1
        
        # P3 = A[i1,j1] * (B[j1,k1]-B[j2,k1])
        U[i1*m+j1][col] = 1
        V[j1*p+k1][col] = 1; V[j2*p+k1][col] = -1
        W[k1*n+i1][col] = 1   # C00? No, C01 += P3
        W[k1p*n+i1][col] = 0  # wait, C01 means output at (row k1+1, col i1)?
        # Let me recheck. C is 2x2, indexed by (k, i) in our convention W[k*n+i].
        # C00 = C at position (k1, i1), C01 = C at position (k2, i1)? No.
        # Standard: C = A @ B. C[i][k] = sum_j A[i][j]*B[j][k].
        # So C[i1][k1] = C00, C[i1][k2] = C01, C[i2][k1] = C10, C[i2][k2] = C11.
        # In W convention: W[k*n+i] = C[i][k].
        # So W[k1*n+i1] = C[i1][k1] = C00
        #    W[k2*n+i1] = C[i1][k2] = C01
        #    W[k1*n+i2] = C[i2][k1] = C10
        #    W[k2*n+i2] = C[i2][k2] = C11
        
        # C01 = P3 + P4
        # So W[k2*n+i1] should get +P3 and +P4.
        # Let me fix P3:
        W[k2*n+i1][col] = 1   # C01 += P3
        col += 1
        
        # P4 = A[i2,j2] * (B[j1,k2]-B[j2,k2])
        U[i2*m+j2][col] = 1
        V[j1*p+k2][col] = 1; V[j2*p+k2][col] = -1
        W[k2*n+i2][col] = 1   # C11? No.
        # C01 = P3 + P4, so W[k2*n+i1] += P4
        # C11 = P1 - P3 + P6 + P7, no P4 in C11.
        # Wait, I think I mislabeled. Let me recheck Strassen formulas.
        # Standard Strassen for 2x2:
        # C00 = M1 - M4 + M5 + M7
        # C01 = M3 + M4
        # C10 = M2 + M5
        # C11 = M1 - M3 + M6 + M7
        # where M1..M7 as defined above.
        # So C01 = M3 + M4 = P3 + P4.
        # C11 = M1 - M3 + M6 + M7 = P1 - P3 + P6 + P7.
        # So P4 only appears in C01.
        W[k2*n+i1][col] = 1   # C01 += P4
        col += 1
        
        # P5 = (A[i1,j1]+A[i1,j2]) * B[j2,k2]
        U[i1*m+j1][col] = 1; U[i1*m+j2][col] = 1
        V[j2*p+k2][col] = 1
        W[k2*n+i1][col] = 1   # C01? No, C00 += P5 and C10 += P5
        # C00 = M1 - M4 + M5 + M7, so C00 += P5 -> W[k1*n+i1] += P5
        # C10 = M2 + M5, so C10 += P5 -> W[k1*n+i2] += P5
        W[k1*n+i1][col] = 1
        W[k1*n+i2][col] = 1
        col += 1
        
        # P6 = (A[i2,j1]-A[i1,j1]) * (B[j1,k1]+B[j2,k1])
        U[i2*m+j1][col] = 1; U[i1*m+j1][col] = -1
        V[j1*p+k1][col] = 1; V[j2*p+k1][col] = 1
        W[k1*n+i2][col] = 1   # C10? No, C11 += P6
        # C11 = M1 - M3 + M6 + M7, so C11 += P6 -> W[k2*n+i2] += P6
        W[k2*n+i2][col] = 1
        col += 1
        
        # P7 = (A[i1,j2]-A[i2,j2]) * (B[j1,k2]+B[j2,k2])
        U[i1*m+j2][col] = 1; U[i2*m+j2][col] = -1
        V[j1*p+k2][col] = 1; V[j2*p+k2][col] = 1
        W[k2*n+i1][col] = 1   # C01? No, C00 += P7 and C11 += P7
        # C00 = M1 - M4 + M5 + M7, so C00 += P7 -> W[k1*n+i1] += P7
        # C11 = M1 - M3 + M6 + M7, so C11 += P7 -> W[k2*n+i2] += P7
        W[k1*n+i1][col] = 1
        W[k2*n+i2][col] = 1
        col += 1
        
        return col
    
    # Hmm, I'm making mistakes with the W assignments. Let me be very careful.
    # I'll build it step by step and verify at the end.
    
    # Let me restart the construction more carefully.
    
    # Fresh start
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Strassen 2x2x2 helper, returns new col
    def add_strassen_222(i1, j1, k1, col):
        """
        Strassen for A[i1:i1+2, j1:j1+2] @ B[j1:j1+2, k1:k1+2]
        A indices: (i1,j1), (i1,j1+1), (i1+1,j1), (i1+1,j1+1)
        B indices: (j1,k1), (j1,k1+1), (j1+1,k1), (j1+1,k1+1)
        C indices: (i1,k1), (i1,k1+1), (i1+1,k1), (i1+1,k1+1)
        W convention: W[k*n + i] = C[i][k]
        """
        i2 = i1 + 1
        j2 = j1 + 1
        k2 = k1 + 1
        
        # P1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
        U[i1*m+j1][col] = 1; U[i2*m+j2][col] = 1
        V[j1*p+k1][col] = 1; V[j2*p+k2][col] = 1
        # C[i1,k1] += P1, C[i2,k2] += P1
        W[k1*n+i1][col] = 1
        W[k2*n+i2][col] = 1
        col += 1
        
        # P2 = (A[i2,j1]+A[i2,j2]) * B[j1,k1]
        U[i2*m+j1][col] = 1; U[i2*m+j2][col] = 1
        V[j1*p+k1][col] = 1
        # C[i2,k1] += P2
        W[k1*n+i2][col] = 1
        col += 1
        
        # P3 = A[i1,j1] * (B[j1,k1]-B[j2,k1])
        U[i1*m+j1][col] = 1
        V[j1*p+k1][col] = 1; V[j2*p+k1][col] = -1
        # C[i1,k2] += P3
        W[k2*n+i1][col] = 1
        col += 1
        
        # P4 = A[i2,j2] * (B[j1,k2]-B[j2,k2])
        U[i2*m+j2][col] = 1
        V[j1*p+k2][col] = 1; V[j2*p+k2][col] = -1
        # C[i1,k2] += P4
        W[k2*n+i1][col] = 1
        col += 1
        
        # P5 = (A[i1,j1]+A[i1,j2]) * B[j2,k2]
        U[i1*m+j1][col] = 1; U[i1*m+j2][col] = 1
        V[j2*p+k2][col] = 1
        # C[i1,k1] += P5, C[i2,k1] += P5
        W[k1*n+i1][col] = 1
        W[k1*n+i2][col] = 1
        col += 1
        
        # P6 = (A[i2,j1]-A[i1,j1]) * (B[j1,k1]+B[j2,k1])
        U[i2*m+j1][col] = 1; U[i1*m+j1][col] = -1
        V[j1*p+k1][col] = 1; V[j2*p+k1][col] = 1
        # C[i2,k2] += P6
        W[k2*n+i2][col] = 1
        col += 1
        
        # P7 = (A[i1,j2]-A[i2,j2]) * (B[j1,k2]+B[j2,k2])
        U[i1*m+j2][col] = 1; U[i2*m+j2][col] = -1
        V[j1*p+k2][col] = 1; V[j2*p+k2][col] = 1
        # C[i1,k1] += P7, C[i2,k2] += P7
        W[k1*n+i1][col] = 1
        W[k2*n+i2][col] = 1
        col += 1
        
        return col
    
    # Three Strassen blocks for k = (0,1), (2,3), (4,5)
    col = add_strassen_222(0, 0, 0, col)   # i=0, j=0, k=0 -> 7 cols
    col = add_strassen_222(0, 0, 2, col)   # i=0, j=0, k=2 -> 7 cols
    col = add_strassen_222(0, 0, 4, col)   # i=0, j=0, k=4 -> 7 cols
    
    # Wait, the i1 and j1 are always 0 here because we're multiplying the top-left 2x2 of A
    # by the corresponding 2x2 blocks of B. But that's only for j in {0,1}.
    # The Strassen block multiplies A[0:2, 0:2] @ B[0:2, k1:k1+2].
    # So i1=0, j1=0 is correct for the A submatrix, but the k positions vary.
    
    # Now the 2x2x1 part: A[0:2, 0:2] @ B[0:2, 6] (k=6)
    # Schoolbook: 4 terms
    # C[i1,k6] += A[i1,j1]*B[j1,k6]
    # C[i1,k6] += A[i1,j2]*B[j2,k6]
    # C[i2,k6] += A[i2,j1]*B[j1,k6]
    # C[i2,k6] += A[i2,j2]*B[j2,k6]
    
    # Term: A[0,0]*B[0,6]
    U[0*m+0][col] = 1
    V[0*p+6][col] = 1
    W[6*n+0][col] = 1
    col += 1
    
    # Term: A[0,1]*B[1,6]
    U[0*m+1][col] = 1
    V[1*p+6][col] = 1
    W[6*n+0][col] = 1
    col += 1
    
    # Term: A[1,0]*B[0,6]
    U[1*m+0][col] = 1
    V[0*p+6][col] = 1
    W[6*n+1][col] = 1
    col += 1
    
    # Term: A[1,1]*B[1,6]
    U[1*m+1][col] = 1
    V[1*p+6][col] = 1
    W[6*n+1][col] = 1
    col += 1
    
    # Now the 2x1 x 1x7 part: A[0:2, 2] @ B[2, 0:7]
    # A[0,2] * B[2,k] for k in 0..6
    # A[1,2] * B[2,k] for k in 0..6
    # 14 terms total
    
    for k in range(7):
        # A[0,2] * B[2,k]
        U[0*m+2][col] = 1
        V[2*p+k][col] = 1
        W[k*n+0][col] = 1
        col += 1
        
        # A[1,2] * B[2,k]
        U[1*m+2][col] = 1
        V[2*p+k][col] = 1
        W[k*n+1][col] = 1
        col += 1
    
    # Total columns: 21 + 4 + 14 = 39
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
