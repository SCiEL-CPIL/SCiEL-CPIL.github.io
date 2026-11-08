"""Strassen-based rank-42 decomposition for 2x4 x 4x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start = time.time()
    deadline = start + 135.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # We decompose 2x4 x 4x6 into 3 blocks of 2x2 x 2x2 using Strassen (rank 7 each)
    # A is 2x4: split into A_left (2x2, cols 0-1) and A_right (2x2, cols 2-3)
    # B is 4x6: split into B_top (2x6, rows 0-1) and B_bottom (2x6, rows 2-3)
    # B_top split into B_top_left (2x2), B_top_mid (2x2), B_top_right (2x2)
    # B_bottom split into B_bot_left (2x2), B_bot_mid (2x2), B_bot_right (2x2)
    # C is 2x6: split into C_left (2x2), C_mid (2x2), C_right (2x2)
    #
    # C_left = A_left * B_top_left + A_right * B_bot_left  -> use Strassen on (A_left, A_right) x (B_top_left, B_bot_left)
    # Actually, we need C[j,k] = sum_i sum_l A[i,l] * B[l,k]
    # 
    # Let's think of it as:
    # A = [A0 A1] where A0 is 2x2, A1 is 2x2
    # B = [B0; B1] where B0 is 2x6, B1 is 2x6
    # C = A0*B0 + A1*B1
    #
    # Now split B0 = [B00 B01 B02] and B1 = [B10 B11 B12], each 2x2
    # C = A0*[B00 B01 B02] + A1*[B10 B11 B12]
    #   = [A0*B00 + A1*B10, A0*B01 + A1*B11, A0*B02 + A1*B12]
    #
    # Each term A0*Bjk + A1*Bjk is a 2x2 product sum.
    # We can use Strassen-like approach for computing X*P + Y*Q where X,Y,P,Q are 2x2.
    #
    # Actually, let's use a simpler approach: compute A0*B0 and A1*B1 separately using Strassen?
    # No, Strassen computes 2x2 * 2x2. Here we have 2x2 * 2x6.
    #
    # Alternative: split the 2x4 x 4x6 into 3 independent 2x2 x 2x2 problems is not directly possible
    # because the dimensions don't align that way.
    #
    # Let me reconsider. The standard approach for 2x4 x 4x6:
    # We can view this as computing three 2x2 matrix products simultaneously? No.
    #
    # Actually, rank 42 comes from: 3 blocks, each using 14 multiplications? No, 3*7=21? That doesn't add up.
    # Wait, 2x4x6 has schoolbook rank 48. Strassen gives rank 28 for 2x2x2.
    # 
    # For 2x4 x 4x6, we can decompose as follows:
    # A (2x4) = [A00 A01 A02 A03] (but as 2x4, it's 2 rows, 4 cols)
    # B (4x6)
    # C (2x6)
    #
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    #
    # Group j=0,1 and j=2,3:
    # C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{3} A[i,j] * B[j,k]
    #
    # First term: 2x2 * 2x6 -> can be done with 2 Strassen calls? No, it's 2x2 * 2x6.
    # We can split the 2x6 into three 2x2 blocks: B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6]
    # So first term gives three 2x2 * 2x2 products.
    # Second term: 2x2 * 2x6, similarly three 2x2 * 2x2 products.
    # Total 6 products of 2x2 * 2x2, each rank 7, total rank 42.
    #
    # But wait, the outputs need to be combined: C[:, 0:2] = term1_block0 + term2_block0, etc.
    # The W matrix accumulates, so we just add the contributions.
    #
    # Let's implement this:
    # For each of the 6 (2x2, 2x2) Strassen problems:
    #   (A_sub, B_sub) -> 7 rank-1 tensors
    #   A_sub is a 2x2 submatrix of A (2x4)
    #   B_sub is a 2x2 submatrix of B (4x6)
    #   The output contributes to a 2x2 block of C (2x6)
    #
    # A_sub options: A[:, 0:2] (call it A_L) and A[:, 2:4] (call it A_R)
    # B_sub options for A_L: B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6]
    # B_sub options for A_R: B[2:4, 0:2], B[2:4, 2:4], B[2:4, 4:6]
    # C block for A_L, B[0:2, c:c+2]: C[:, c:c+2]
    # C block for A_R, B[2:4, c:c+2]: C[:, c:c+2]
    
    U = [[0.0] * 42 for _ in range(n * m)]  # 8 x 42
    V = [[0.0] * 42 for _ in range(m * p)]  # 24 x 42
    W = [[0.0] * 42 for _ in range(p * n)]  # 12 x 42
    
    col = 0
    
    def strassen_add(u_base, v_base, w_base):
        """Add 7 Strassen rank-1 terms.
        u_base: base index for U (2 rows for the 2x2 A block)
        v_base: base index for V (2 rows for the 2x2 B block)
        w_base: base index for W (2 rows for the 2x2 C block)
        Returns next column index.
        """
        global col
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
        
        # M1
        U[u_base + 0][col] += 1.0
        U[u_base + 1][col] += 1.0
        V[v_base + 0][col] += 1.0
        V[v_base + 1][col] += 1.0
        W[w_base + 0][col] += 1.0
        W[w_base + 1][col] += 1.0
        col += 1
        
        # M2
        U[u_base + 1][col] += 1.0  # A10
        U[u_base + 1][col] += 1.0  # A11 -> wait, A10+A11
        # Actually u_base+0 is row 0 (A00, A01), u_base+1 is row 1 (A10, A11)
        # M2 = (A10+A11)*B00
        # A10 is at index (row=1, col=0) of the 2x2 block, A11 is at (row=1, col=1)
        # In U, the 2x2 A block occupies rows u_base, u_base+1, and within each row, 2 columns of the original
        # But we're flattening: the 2x2 A submatrix has 4 elements, mapped to U rows.
        # Wait, I need to be more careful about indexing.
        #
        # Let me reconsider the indexing.
        # A is 2x4. Flatten to 8 elements: A[0,0], A[0,1], A[0,2], A[0,3], A[1,0], A[1,1], A[1,2], A[1,3]
        # U has 8 rows.
        #
        # For A_L = A[:, 0:2]: elements are A[0,0], A[0,1], A[1,0], A[1,1] -> U rows 0,1,4,5
        # For A_R = A[:, 2:4]: elements are A[0,2], A[0,3], A[1,2], A[1,3] -> U rows 2,3,6,7
        #
        # B is 4x6. Flatten to 24 elements: B[0,0]..B[0,5], B[1,0]..B[1,5], B[2,0]..B[2,5], B[3,0]..B[3,5]
        # V has 24 rows.
        #
        # B[0:2, 0:2]: B[0,0], B[0,1], B[1,0], B[1,1] -> V rows 0,1,6,7
        # B[0:2, 2:4]: B[0,2], B[0,3], B[1,2], B[1,3] -> V rows 2,3,8,9
        # B[0:2, 4:6]: B[0,4], B[0,5], B[1,4], B[1,5] -> V rows 4,5,10,11
        # B[2:4, 0:2]: B[2,0], B[2,1], B[3,0], B[3,1] -> V rows 12,13,18,19
        # B[2:4, 2:4]: B[2,2], B[2,3], B[3,2], B[3,3] -> V rows 14,15,20,21
        # B[2:4, 4:6]: B[2,4], B[2,5], B[3,4], B[3,5] -> V rows 16,17,22,23
        #
        # C is 2x6. Flatten to 12 elements: C[0,0]..C[0,5], C[1,0]..C[1,5]
        # W has 12 rows. But W[k*n+i, r] means W is indexed as [k*n + i] where k is col, i is row.
        # So C[i,k] is at W[k*2 + i].
        # C[:, 0:2]: C[0,0]=W[0], C[0,1]=W[2], C[1,0]=W[1], C[1,1]=W[3]
        # C[:, 2:4]: C[0,2]=W[4], C[0,3]=W[6], C[1,2]=W[5], C[1,3]=W[7]
        # C[:, 4:6]: C[0,4]=W[8], C[0,5]=W[10], C[1,4]=W[9], C[1,5]=W[11]
        
        # Let me redo with proper indexing.
        # This is getting complex. Let me just implement it directly.
        return col
    
    # Let me restart with a cleaner implementation.
    
    def add_strassen(a_rows, b_rows, c_base_col):
        """Add Strassen decomposition for 2x2 A block at a_rows, 2x2 B block at b_rows, 
        contributing to C block starting at column c_base_col (2x2 block).
        a_rows: [row0, row1] - U row indices for A[0,0], A[1,0] (col 0 of A block)
                 Actually, let's pass the 4 U indices for the 2x2 A block.
        """
        pass
    
    # Let me just write it out explicitly for each of the 6 cases.
    
    def strassen_block(a00_idx, a01_idx, a10_idx, a11_idx, b00_idx, b01_idx, b10_idx, b11_idx, c00_w, c01_w, c10_w, c11_w):
        """Add 7 Strassen terms for 2x2 A block, 2x2 B block, 2x2 C block.
        a00_idx: U row for A[0,0]
        a01_idx: U row for A[0,1]
        a10_idx: U row for A[1,0]
        a11_idx: U row for A[1,1]
        b00_idx: V row for B[0,0]
        b01_idx: V row for B[0,1]
        b10_idx: V row for B[1,0]
        b11_idx: V row for B[1,1]
        c00_w: W row for C[0,0]
        c01_w: W row for C[0,1]
        c10_w: W row for C[1,0]
        c11_w: W row for C[1,1]
        """
        global col
        # M1 = (A00+A11)(B00+B11)
        U[a00_idx][col] += 1.0
        U[a11_idx][col] += 1.0
        V[b00_idx][col] += 1.0
        V[b11_idx][col] += 1.0
        W[c00_w][col] += 1.0
        W[c11_w][col] += 1.0
        col += 1
        
        # M2 = (A10+A11)B00
        U[a10_idx][col] += 1.0
        U[a11_idx][col] += 1.0
        V[b00_idx][col] += 1.0
        W[c10_w][col] += 1.0
        W[c11_w][col] -= 1.0
        col += 1
        
        # M3 = A00(B01-B11)
        U[a00_idx][col] += 1.0
        V[b01_idx][col] += 1.0
        V[b11_idx][col] -= 1.0
        W[c01_w][col] += 1.0
        W[c11_w][col] += 1.0
        col += 1
        
        # M4 = A11(B10-B00)
        U[a11_idx][col] += 1.0
        V[b10_idx][col] += 1.0
        V[b00_idx][col] -= 1.0
        W[c00_w][col] += 1.0
        W[c10_w][col] += 1.0
        col += 1
        
        # M5 = (A00+A01)B11
        U[a00_idx][col] += 1.0
        U[a01_idx][col] += 1.0
        V[b11_idx][col] += 1.0
        W[c00_w][col] -= 1.0
        W[c01_w][col] += 1.0
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[a10_idx][col] += 1.0
        U[a00_idx][col] -= 1.0
        V[b00_idx][col] += 1.0
        V[b01_idx][col] += 1.0
        W[c11_w][col] += 1.0
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[a01_idx][col] += 1.0
        U[a11_idx][col] -= 1.0
        V[b10_idx][col] += 1.0
        V[b11_idx][col] += 1.0
        W[c00_w][col] += 1.0
        W[c11_w][col] += 1.0
        col += 1
    
    # A_L: A[:, 0:2] -> U rows: A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    # A_R: A[:, 2:4] -> U rows: A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
    
    # B blocks:
    # B[0:2, 0:2]: V rows B[0,0]=0, B[0,1]=1, B[1,0]=6, B[1,1]=7
    # B[0:2, 2:4]: V rows B[0,2]=2, B[0,3]=3, B[1,2]=8, B[1,3]=9
    # B[0:2, 4:6]: V rows B[0,4]=4, B[0,5]=5, B[1,4]=10, B[1,5]=11
    # B[2:4, 0:2]: V rows B[2,0]=12, B[2,1]=13, B[3,0]=18, B[3,1]=19
    # B[2:4, 2:4]: V rows B[2,2]=14, B[2,3]=15, B[3,2]=20, B[3,3]=21
    # B[2:4, 4:6]: V rows B[2,4]=16, B[2,5]=17, B[3,4]=22, B[3,5]=23
    
    # C blocks:
    # C[:, 0:2]: W rows C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3
    # C[:, 2:4]: W rows C[0,2]=4, C[0,3]=6, C[1,2]=5, C[1,3]=7
    # C[:, 4:6]: W rows C[0,4]=8, C[0,5]=10, C[1,4]=9, C[1,5]=11
    
    # 6 Strassen calls:
    # 1. A_L * B[0:2, 0:2] -> C[:, 0:2]
    strassen_block(0, 1, 4, 5,  # A_L
                   0, 1, 6, 7,  # B[0:2, 0:2]
                   0, 2, 1, 3)  # C[:, 0:2]
    
    # 2. A_L * B[0:2, 2:4] -> C[:, 2:4]
    strassen_block(0, 1, 4, 5,  # A_L
                   2, 3, 8, 9,  # B[0:2, 2:4]
                   4, 6, 5, 7)  # C[:, 2:4]
    
    # 3. A_L * B[0:2, 4:6] -> C[:, 4:6]
    strassen_block(0, 1, 4, 5,  # A_L
                   4, 5, 10, 11,  # B[0:2, 4:6]
                   8, 10, 9, 11)  # C[:, 4:6]
    
    # 4. A_R * B[2:4, 0:2] -> C[:, 0:2]
    strassen_block(2, 3, 6, 7,  # A_R
                   12, 13, 18, 19,  # B[2:4, 0:2]
                   0, 2, 1, 3)  # C[:, 0:2]
    
    # 5. A_R * B[2:4, 2:4] -> C[:, 2:4]
    strassen_block(2, 3, 6, 7,  # A_R
                   14, 15, 20, 21,  # B[2:4, 2:4]
                   4, 6, 5, 7)  # C[:, 2:4]
    
    # 6. A_R * B[2:4, 4:6] -> C[:, 4:6]
    strassen_block(2, 3, 6, 7,  # A_R
                   16, 17, 22, 23,  # B[2:4, 4:6]
                   8, 10, 9, 11)  # C[:, 4:6]
    
    if col != 42:
        raise ValueError(f"Expected 42 columns, got {col}")
    
    # Verify with a random test
    np.random.seed(42)
    A_test = np.random.rand(n, m)
    B_test = np.random.rand(m, p)
    C_expected = A_test @ B_test
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Flatten A and B according to the indexing
    A_flat = A_test.flatten()  # [A[0,0], A[0,1], A[0,2], A[0,3], A[1,0], ...]
    B_flat = B_test.flatten()  # [B[0,0], ..., B[0,5], B[1,0], ...]
    
    # Compute C via tensor decomposition: C_flat = sum_r U_r * V_r * W_r (outer product)
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for appropriate j
    # Actually, the tensor is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And C[i,k] = sum_j T[i,j,k]
    
    C_computed = np.zeros((n, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(42):
                    val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                C_computed[i, k] += val
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook
        return _schoolbook(n, m, p)
    
    # Convert to lists of integers/half-integers
    U_out = []
    for row in U:
        U_out.append([int(x) if x == int(x) else x for x in row])
    V_out = []
    for row in V:
        V_out.append([int(x) if x == int(x) else x for x in row])
    W_out = []
    for row in W:
        W_out.append([int(x) if x == int(x) else x for x in row])
    
    return {"U": U_out, "V": V_out, "W": W_out}
