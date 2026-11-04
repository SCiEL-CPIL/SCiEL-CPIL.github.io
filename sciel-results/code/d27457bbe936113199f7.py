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
    
    # 1. Construct 4x4x4 Strassen decomposition (rank 49)
    U2, V2, W2 = _strassen_2x2x2()
    
    # Define coefficients for the 4x4x4 Strassen decomposition
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
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for c2 in range(7):
        for c1 in range(7):
            col = c2 * 7 + c1
            for (bi, bj), a in a_coeffs[c2].items():
                for di in range(2):
                    for dj in range(2):
                        row = (bi*2+di)*4 + (bj*2+dj)
                        U4[row][col] = a * U2[di*2+dj][c1]
            for (bj, bk), b in b_coeffs[c2].items():
                for dj in range(2):
                    for dk in range(2):
                        row = (bj*2+dj)*4 + (bk*2+dk)
                        V4[row][col] = b * V2[dj*2+dk][c1]
            for (bi, bk), c in c_coeffs[c2].items():
                for dk in range(2):
                    for di in range(2):
                        row = (bk*2+dk)*4 + (bi*2+di)
                        W4[row][col] = c * W2[dk*2+di][c1]
    
    # 2. Construct 4x4x3 schoolbook decomposition (rank 48)
    # This handles k in 4..6 of the full 4x4x7 problem
    U3 = [[0]*48 for _ in range(16)]
    V3 = [[0]*48 for _ in range(12)] # Local p=3, so m*p = 4*3=12
    W3 = [[0]*48 for _ in range(12)] # Local p=3, so p*n = 3*4=12
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k_local in range(3): # 0,1,2 corresponding to global k=4,5,6
                U3[i*4+j][col] = 1
                V3[j*3+k_local][col] = 1
                W3[k_local*4+i][col] = 1
                col += 1
    
    # 3. Combine into full 4x4x7 matrices
    total_rank = 49 + 48
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(28)] # m*p = 4*7
    W = [[0]*total_rank for _ in range(28)] # p*n = 7*4
    
    # Copy 4x4x4 part (columns 0..48)
    for r in range(16):
        for c in range(49):
            U[r][c] = U4[r][c]
            # Map V4 row (j*4+k) to global V row (j*7+k)
            j, k = divmod(r, 4)
            V[j*7+k][c] = V4[r][c]
            # Map W4 row (k*4+i) to global W row (k*4+i) -- Note: k here is local 0..3, global is same
            k_w, i_w = divmod(r, 4)
            W[k_w*4+i_w][c] = W4[r][c]
            
    # Copy 4x4x3 part (columns 49..96)
    for r in range(16):
        for c in range(48):
            U[r][49+c] = U3[r][c]
    
    for r in range(12):
        j, k_local = divmod(r, 3)
        # Map V3 row (j*3+k_local) to global V row (j*7 + (4+k_local))
        v_row = j*7 + (4 + k_local)
        for c in range(48):
            V[v_row][49+c] = V3[r][c]
            
    for r in range(12):
        k_local, i = divmod(r, 4)
        # Map W3 row (k_local*4+i) to global W row ((4+k_local)*4 + i)
        w_row = (4 + k_local)*4 + i
        for c in range(48):
            W[w_row][49+c] = W3[r][c]
    
    # Verify
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
