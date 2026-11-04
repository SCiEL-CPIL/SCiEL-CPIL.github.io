"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import sys

def _validate(U, V, W, n, m, p):
    """Check that the tensor decomposition is exact (within tolerance)."""
    rank = len(U[0])
    # Build the tensor T[i,j,k] from U, V, W
    # T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need this to equal 1 if i,j,k are all in range, else 0.
    # Actually, the matrix multiplication tensor M_{i,j,k} is 1 if the indices are valid.
    # For multiplying A (n x m) by B (m x p) to get C (n x p),
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 (it's the identity tensor for the bilinear form)
    # Wait, the decomposition is: C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    # No, the standard form is:
    # C[i,k] = sum_c U_{i, c} * W_{k, c} * V_{j, c} ... no.
    # Let's use the definition from the task:
    # "exactly decompose the matrix multiplication tensor"
    # The tensor M_{i,j,k} is 1 if the indices correspond to a valid multiplication term.
    # Actually, for matrix multiplication, the tensor is simply 1 for all i,j,k in range.
    # The decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # Let's verify this.
    tol = 1e-10
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1) > tol:
                    return False
    # Also check that it's 0 for "invalid" indices? 
    # No, the tensor is only defined for valid i,j,k. The decomposition must match the tensor on its support.
    # But wait, the matrices U, V, W have more rows than just the valid indices.
    # U has n*m rows, V has m*p rows, W has p*n rows.
    # The tensor T is of size (n*m) x (m*p) x (p*n).
    # T[a, b, c] = sum_k U[a, k] * V[b, k] * W[c, k]
    # This must equal 1 if a, b, c correspond to a valid (i, j, k) triple, i.e., 
    # a = i*m+j, b = j*p+k, c = k*n+i.
    # And 0 otherwise.
    # Let's re-verify.
    for a in range(n*m):
        i, j = divmod(a, m)
        for b in range(m*p):
            j2, k = divmod(b, p)
            # For the tensor to be non-zero, we need j == j2
            for c in range(p*n):
                k2, i2 = divmod(c, n)
                # For the tensor to be non-zero, we need k == k2 and i == i2
                expected = 1 if (j == j2 and k == k2 and i == i2) else 0
                val = 0
                for r in range(rank):
                    val += U[a][r] * V[b][r] * W[c][r]
                if abs(val - expected) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition."""
    # n=m=p=2
    # U size 4x7, V size 4x7, W size 4x7
    # Rows of U: i*2+j for i,j in 0..1 -> 0,1,2,3
    # Rows of V: j*2+k for j,k in 0..1 -> 0,1,2,3
    # Rows of W: k*2+i for k,i in 0..1 -> 0,1,2,3
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # U[i,j] = coefficient of A[i,j] in M_c
    # V[j,k] = coefficient of B[j,k] in M_c
    # W[k,i] = coefficient of M_c in C[k,i] (wait, W is indexed by k*n+i)
    #
    # Let's build the 7 vectors.
    # c=0 (M1):
    # U: A00=1, A11=1, A01=0, A10=0 -> [1, 0, 0, 1] (i=0,j=0; i=0,j=1; i=1,j=0; i=1,j=1)
    # Wait, row index is i*m+j. So row 0=(0,0), row 1=(0,1), row 2=(1,0), row 3=(1,1).
    # U_col0 = [1, 0, 0, 1]
    # V_col0: B00=1, B11=1, B01=0, B10=0 -> row 0=(0,0), row 1=(0,1), row 2=(1,0), row 3=(1,1) -> [1, 0, 0, 1]
    # W_col0: C00=1, C01=0, C10=0, C11=1 -> row 0=(0,0) k=0,i=0, row 1=(0,1) k=0,i=1, row 2=(1,0) k=1,i=0, row 3=(1,1) k=1,i=1
    # C00 is row 0, C11 is row 3. So W_col0 = [1, 0, 0, 1]
    
    # Let's do this systematically.
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C10
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C01
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1  # C00, C10
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1  # C00, C01
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1  # C00, C11
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Strategy: Use Strassen for the 4x4x4 part (rank 49) and schoolbook for the 4x4x3 part (rank 48)
    # Total rank: 97
    # 
    # But wait, the tensor is for 4x4x7. We can't just split it into 4x4x4 and 4x4x3 and add the ranks
    # because the decomposition must be for the full tensor.
    #
    # Actually, we can! The tensor for 4x4x7 has entries T[i,j,k] for i in 0..3, j in 0..3, k in 0..6.
    # We can decompose it as the sum of the tensor for 4x4x4 (k in 0..3) and the tensor for 4x4x3 (k in 4..6).
    # The 4x4x4 tensor can be decomposed with rank 49 (Strassen).
    # The 4x4x3 tensor can be decomposed with rank 48 (schoolbook).
    # The total rank is 49 + 48 = 97.
    #
    # Let's implement this.
    
    # First, generate the Strassen decomposition for 4x4x4.
    # This is a recursive application of Strassen 2x2x2.
    # U_4 size 16x49, V_4 size 16x49, W_4 size 16x49.
    
    # We can build this by applying Strassen 2x2x2 to the 2x2 blocks of 2x2 matrices.
    # There are 4 blocks in A, 4 blocks in B, 4 blocks in C.
    # The 7 Strassen multiplications are between 2x2 matrices.
    # Each 2x2 multiplication is decomposed into 7 scalar multiplications.
    # So total 7*7 = 49 scalar multiplications.
    
    # Let's build the 4x4x4 Strassen decomposition.
    # U4[i,j,c] = sum over block indices...
    # This is getting complex. Let's use a different approach.
    # 
    # Alternative: Use the known result that R(4,4,4) <= 49.
    # But I need the actual U, V, W matrices.
    # 
    # Let's build it step by step.
    # The 4x4 matrix can be divided into four 2x2 blocks:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # Strassen gives:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each M_i is a 2x2 matrix product.
    # So M1 = (A00+A11) * (B00+B11) is a 2x2 matrix product.
    # We can decompose each M_i using the 2x2x2 Strassen decomposition.
    #
    # Let's define the 2x2x2 Strassen decomposition first.
    U2, V2, W2 = _strassen_2x2x2()
    
    # Now build U4, V4, W4 for 4x4x4.
    # U4 has 16 rows (i*4+j for i,j in 0..3) and 49 columns.
    # V4 has 16 rows (j*4+k for j,k in 0..3) and 49 columns.
    # W4 has 16 rows (k*4+i for k,i in 0..3) and 49 columns.
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    # For each of the 7 Strassen multiplications (c2 in 0..6),
    # and for each of the 7 sub-multiplications (c1 in 0..6),
    # we have a scalar multiplication.
    # The column index in U4 is c2*7 + c1.
    
    # For a given c2 (Strassen multiplication index) and c1 (sub-multiplication index):
    # The A-part is a linear combination of the A blocks.
    # The B-part is a linear combination of the B blocks.
    # The C-part is a linear combination of the C blocks.
    
    # Let's define the coefficients for each c2.
    # For c2=0 (M1): A00+A11, B00+B11, C00+=1, C11+=1
    # For c2=1 (M2): A10+A11, B00, C10+=1
    # For c2=2 (M3): A00, B01-B11, C01+=1
    # For c2=3 (M4): A11, B10-B00, C00+=1, C10+=1
    # For c2=4 (M5): A00+A01, B11, C00+=-1, C01+=1
    # For c2=5 (M6): A10-A00, B00+B01, C11+=1
    # For c2=6 (M7): A01-A11, B10+B11, C00+=1, C11+=1
    
    # For each c2, we need to express the A-part and B-part in terms of the 4 blocks.
    # Each block is a 2x2 matrix, so it has 4 entries.
    # The U4 row index is i*4+j.
    # For block A00 (i=0,j=0), the entries are at rows 0,1,4,5 (i=0,j=0; i=0,j=1; i=1,j=0; i=1,j=1)
    # Wait, row index is i*4+j. So for block A00 (i_block=0, j_block=0), the rows are 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5.
    # For block A01 (i_block=0, j_block=1), the rows are 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7.
    # For block A10 (i_block=1, j_block=0), the rows are 1*4+0=4, 1*4+1=5, 2*4+0=8, 2*4+1=9.
    # For block A11 (i_block=1, j_block=1), the rows are 1*4+2=6, 1*4+3=7, 2*4+2=10, 2*4+3=11.
    
    # Let's define a helper function to get the row indices for a block.
    def block_rows(block_i, block_j, size=4):
        rows = []
        for di in range(2):
            for dj in range(2):
                i = block_i * 2 + di
                j = block_j * 2 + dj
                rows.append(i * size + j)
        return rows
    
    # For each c2, define the coefficients for A blocks and B blocks.
    # a_coeffs[c2] is a dict: {(block_i, block_j): coeff}
    # b_coeffs[c2] is a dict: {(block_j, block_k): coeff}
    # c_coeffs[c2] is a dict: {(block_i, block_k): coeff}
    
    a_coeffs = [
        {(0,0): 1, (1,1): 1},  # M1
        {(1,0): 1, (1,1): 1},  # M2
        {(0,0): 1},            # M3
        {(1,1): 1},            # M4
        {(0,0): 1, (0,1): 1},  # M5
        {(1,0): 1, (0,0): -1}, # M6
        {(0,1): 1, (1,1): -1}, # M7
    ]
    
    b_coeffs = [
        {(0,0): 1, (1,1): 1},  # M1
        {(0,0): 1},            # M2
        {(0,1): 1, (1,1): -1}, # M3
        {(1,0): 1, (0,0): -1}, # M4
        {(1,1): 1},            # M5
        {(0,0): 1, (0,1): 1},  # M6
        {(1,0): 1, (1,1): 1},  # M7
    ]
    
    c_coeffs = [
        {(0,0): 1, (1,1): 1},  # M1
        {(1,0): 1},            # M2
        {(0,1): 1},            # M3
        {(0,0): 1, (1,0): 1},  # M4
        {(0,0): -1, (0,1): 1}, # M5
        {(1,1): 1},            # M6
        {(0,0): 1, (1,1): 1},  # M7
    ]
    
    # Now fill U4, V4, W4.
    for c2 in range(7):
        for c1 in range(7):
            col = c2 * 7 + c1
            # For U4: for each A block (block_i, block_j) with coeff a,
            # the 2x2 block has 4 entries. For each entry (di, dj),
            # the row is (block_i*2+di)*4 + (block_j*2+dj).
            # The coefficient is a * U2[di*2+dj][c1]
            for (bi, bj), a in a_coeffs[c2].items():
                for di in range(2):
                    for dj in range(2):
                        row = (bi*2+di)*4 + (bj*2+dj)
                        U4[row][col] = a * U2[di*2+dj][c1]
            
            # For V4: similar for B blocks
            for (bj, bk), b in b_coeffs[c2].items():
                for dj in range(2):
                    for dk in range(2):
                        row = (bj*2+dj)*4 + (bk*2+dk)
                        V4[row][col] = b * V2[dj*2+dk][c1]
            
            # For W4: similar for C blocks
            for (bi, bk), c in c_coeffs[c2].items():
                for dk in range(2):
                    for di in range(2):
                        row = (bk*2+dk)*4 + (bi*2+di)
                        W4[row][col] = c * W2[dk*2+di][c1]
    
    # Now build the schoolbook decomposition for 4x4x3 (rank 48).
    # U3 size 16x48, V3 size 12x48, W3 size 12x48.
    # Wait, the dimensions are n=4, m=4, p=3.
    # U size n*m = 16, V size m*p = 12, W size p*n = 12.
    # But we need to combine this with the 4x4x4 part.
    # The full tensor is 4x4x7, so U size 16, V size 28, W size 28.
    # The 4x4x3 part corresponds to k in 3..6 (0-indexed, so k=3,4,5,6).
    # Wait, p=7, so k in 0..6. The 4x4x4 part is k in 0..3, and the 4x4x3 part is k in 4..6.
    # No, 4+3=7, so k in 0..3 is 4 values, and k in 4..6 is 3 values.
    
    # Let's build the schoolbook decomposition for the 4x4x3 part.
    # This corresponds to k in 4..6.
    # U3 size 16x48, V3 size 12x48, W3 size 12x48.
    # But we need to place this into the full U, V, W matrices.
    # U has 16 rows, V has 28 rows, W has 28 rows.
    # The 4x4x3 part uses V rows corresponding to j*7+k for k in 4..6.
    # And W rows corresponding to k*4+i for k in 4..6.
    
    U3 = [[0]*48 for _ in range(16)]
    V3 = [[0]*48 for _ in range(12)]
    W3 = [[0]*48 for _ in range(12)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):  # k in 4,5,6
                U3[i*4+j][col] = 1
                V3[j*7+k][col] = 1  # Wait, V row index is j*p+k = j*7+k
                W3[k*4+i][col] = 1  # W row index is k*n+i = k*4+i
                col += 1
    
    # Now combine U4 and U3 into U, V4 and V3 into V, W4 and W3 into W.
    # U has 16 rows, 49+48=97 columns.
    # V has 28 rows, 49+48=97 columns.
    # W has 28 rows, 49+48=97 columns.
    
    U = [[0]*97 for _ in range(16)]
    V = [[0]*97 for _ in range(28)]
    W = [[0]*97 for _ in range(28)]
    
    # Copy U4 into U (columns 0..48)
    for r in range(16):
        for c in range(49):
            U[r][c] = U4[r][c]
    
    # Copy V4 into V (columns 0..48)
    # V4 has 16 rows, corresponding to j*4+k for k in 0..3.
    # But in the full V, the row index is j*7+k.
    # So V4 row (j*4+k) corresponds to V row (j*7+k).
    for r in range(16):
        j, k = divmod(r, 4)
        v_row = j*7 + k
        for c in range(49):
            V[v_row][c] = V4[r][c]
    
    # Copy W4 into W (columns 0..48)
    # W4 has 16 rows, corresponding to k*4+i for k in 0..3.
    # But in the full W, the row index is k*7+i.
    # So W4 row (k*4+i) corresponds to W row (k*7+i).
    for r in range(16):
        k, i = divmod(r, 4)
        w_row = k*7 + i
        for c in range(49):
            W[w_row][c] = W4[r][c]
    
    # Copy U3 into U (columns 49..96)
    for r in range(16):
        for c in range(48):
            U[r][49+c] = U3[r][c]
    
    # Copy V3 into V (columns 49..96)
    # V3 has 12 rows, corresponding to j*3+k for k in 0..2 (in the 4x4x3 decomposition).
    # But in the full V, the row index is j*7+k for k in 4..6.
    # So V3 row (j*3+k) corresponds to V row (j*7 + (4+k)).
    for r in range(12):
        j, k = divmod(r, 3)
        v_row = j*7 + (4 + k)
        for c in range(48):
            V[v_row][49+c] = V3[r][c]
    
    # Copy W3 into W (columns 49..96)
    # W3 has 12 rows, corresponding to k*4+i for k in 0..2 (in the 4x4x3 decomposition).
    # But in the full W, the row index is k*7+i for k in 4..6.
    # So W3 row (k*4+i) corresponds to W row ((4+k)*7 + i).
    for r in range(12):
        k, i = divmod(r, 4)
        w_row = (4 + k)*7 + i
        for c in range(48):
            W[w_row][49+c] = W3[r][c]
    
    # Verify the decomposition
    if _validate(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
