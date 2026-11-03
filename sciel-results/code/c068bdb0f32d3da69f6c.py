import time
import random
import numpy as np

def _verify(U, V, W, n, m, p, iters=10):
    """
    Verify that the decomposition is correct.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    # W is transposed in the tensor sum convention for the C matrix
    # C[i, k] = sum_j sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # Let's verify by generating random matrices
    for _ in range(iters):
        A = np.random.randint(0, 10, size=(n, m))
        B = np.random.randint(0, 10, size=(m, p))
        C_true = A @ B
        
        # Compute C_decomp
        C_decomp = np.zeros((n, p))
        # Efficiently: sum_c (U[:, c] @ V[:, c].T) * W[:, c]
        # U[:, c] is (n*m), V[:, c] is (m*p)
        # Product is (n, p)
        # Then scale by W[:, c] which is (p*n)
        # W[:, c][k*n + i] multiplies C_decomp[i, k]
        
        # Reshape W[:, c] to (n, p) where W[:, c][k*n + i] -> (i, k)
        # So W_row = W[:, c].reshape(n, p, order='F')?
        # Index is k*n + i.
        # In numpy, reshape(n, p) with default order 'C' maps flat index i*p + k.
        # We want flat index k*n + i to map to (i, k).
        # This is transpose.
        # W_mat = W[:, c].reshape(n, p).T ?
        # Let's check:
        # flat[k*n + i]. If we reshape to (n, p) with order 'C', element (i, k) is flat[i*p + k].
        # This is not the same.
        # If we reshape to (p, n) with order 'C', element (k, i) is flat[k*n + i].
        # So W_mat = W[:, c].reshape(p, n)
        # Then C_decomp += W_mat.T * (U_col @ V_col.T) ?
        # C_decomp[i, k] += W_mat[k, i] * (sum_j U[i, j] V[j, k])
        
        # Actually, let's just loop over rank if it's small, or use einsum
        # Rank is likely < 100.
        
        for c in range(len(U[0])):
            u_col = U[:, c]
            v_col = V[:, c]
            w_col = W[:, c]
            
            # A_mat from U: u_col is (n*m). reshape to (n, m)
            # Index i*m + j. Default reshape (n, m) works.
            A_u = u_col.reshape(n, m)
            
            # B_mat from V: v_col is (m*p). reshape to (m, p)
            # Index j*p + k. Default reshape (m, p) works.
            B_v = v_col.reshape(m, p)
            
            # Product
            prod = A_u @ B_v # (n, p)
            
            # W_col is (p*n). Index k*n + i.
            # We need to multiply prod[i, k] by W_col[k*n + i].
            # Reshape W_col to (p, n) so that W_reshaped[k, i] = W_col[k*n + i].
            W_mat = w_col.reshape(p, n)
            
            C_decomp += prod * W_mat.T
            
        if not np.allclose(C_decomp, C_true):
            return False
    return True

def _strassen_decompose(n, m, p):
    """
    Generate a Strassen-based decomposition if possible.
    Here n=2, m=4, p=6.
    We split M into 2x2 blocks.
    A: 2x4 -> 2x2 blocks (rows 0-1, cols 0-1 and 2-3)
    B: 4x6 -> 2x2 blocks (rows 0-1, 2-3) and (cols 0-1, 2-3, 4-5)
    C: 2x6 -> 2x2 blocks (rows 0-1, cols 0-1, 2-3, 4-5)
    
    The product is block-wise:
    C[i, k] = sum_j A[i, j] B[j, k]
    
    Let's use Strassen for 2x2 multiplication.
    Standard Strassen computes C = A * B for 2x2 A, B.
    M1 = (A11 + A22) * (B11 + B22)
    M2 = (A21 + A22) * B11
    M3 = A11 * (B12 - B22)
    M4 = A22 * (B21 - B11)
    M5 = (A11 + A12) * B22
    M6 = (A21 - A11) * (B11 + B12)
    M7 = (A12 - A22) * (B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    We have 6 such 2x2x2x2 subproblems.
    Rank contribution per subproblem: 7.
    Total rank: 6 * 7 = 42.
    
    Let's implement this.
    """
    if n != 2 or m != 4 or p != 6:
        return None
        
    rank = 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Map global indices to block indices
    # A: row i in {0,1}, col j in {0..3}
    # Block col index: j//2. Block row index: i.
    # A_block[block_row, block_col] = A[i, j]
    
    # B: row j in {0..3}, col k in {0..5}
    # Block row index: j//2. Block col index: k//2.
    # B_block[block_row, block_col] = B[j, k]
    
    # C: row i in {0,1}, col k in {0..5}
    # Block row index: i. Block col index: k//2.
    # C_block[block_row, block_col] = C[i, k]
    
    # The 6 subproblems correspond to pairs of (A_block_col, B_block_col, C_block_col)
    # A has 2 cols of blocks (0, 1).
    # B has 2 rows of blocks (0, 1) and 3 cols of blocks (0, 1, 2).
    # C has 3 cols of blocks (0, 1, 2).
    
    # C_block_col = c_idx.
    # C_block_col = sum_ab (A_block_ab * B_block_bc)
    # So C_block_col = A_block_col0 * B_block_col0 + A_block_col1 * B_block_col1
    # Wait, B is 4x6.
    # B rows are split into 2 blocks of 2.
    # A cols are split into 2 blocks of 2.
    # So A * B:
    # C[i, k] = sum_{j=0}^3 A[i, j] B[j, k]
    # j is in block 0 (0,1) or block 1 (2,3).
    # A[i, j] is in A_block[i, j_block] where j_block = j//2.
    # B[j, k] is in B_block[j_block, k_block] where k_block = k//2.
    # So C[i, k] = sum_{j_block=0}^1 ( A[i, j] * B[j, k] )
    # This is a sum of two matrix products:
    # C_block = A_block0 * B_block_col_k + A_block1 * B_block_col_k
    # Where A_block0 is 2x2 (rows 0-1, cols 0-1 of A).
    # B_block_col_k is 2x2 (rows 0-1, cols 2*k, 2*k+1 of B).
    # Wait, B has 4 rows.
    # A_block0 (cols 0,1 of A) multiplies B rows 0,1.
    # A_block1 (cols 2,3 of A) multiplies B rows 2,3.
    
    # So for each output block column k_block (0, 1, 2):
    # C_block_k = A_block0 @ B_block_top_k + A_block1 @ B_block_bot_k
    # Where B_block_top_k is B[0:2, 2*k:2*k+2]
    # B_block_bot_k is B[2:4, 2*k:2*k+2]
    
    # This doesn't simplify to a single 2x2 multiplication unless we combine.
    # C_block_k = A0 @ B_top + A1 @ B_bot
    # This is a sum of two 2x2 products.
    # We can use Strassen for each product?
    # Product 1: A0 (2x2) * B_top (2x2) -> 7 mults.
    # Product 2: A1 (2x2) * B_bot (2x2) -> 7 mults.
    # Total 14 mults per output block column.
    # 3 output block columns -> 42 mults.
    # This matches rank 42.
    
    # Let's construct the decomposition for one Strassen product.
    # Inputs:
    # A_mat (2x2), B_mat (2x2).
    # U_A: 2x2 block of U.
    # V_B: 2x2 block of V.
    # W_C: 2x2 block of W.
    
    def add_strassen_product(col_offset, A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_row_offset, C_col_offset, sign=1):
        """
        Adds 7 columns to U, V, W corresponding to Strassen product of A[A_row_offset...], B[B_row_offset...].
        A indices: (A_row_offset, A_col_offset) to (A_row_offset+1, A_col_offset+1)
        B indices: (B_row_offset, B_col_offset) to (B_row_offset+1, B_col_offset+1)
        C indices: (C_row_offset, C_col_offset) to (C_row_offset+1, C_col_offset+1)
        
        Note: U index is i*m + j. V index is j*p + k. W index is k*n + i.
        """
        # Strassen M1..M7
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Local indices in 2x2:
        # 00, 01, 10, 11
        
        # We need to define the coefficients for U, V, W for each of the 7 products.
        # U_col will have non-zeros at A entries.
        # V_col will have non-zeros at B entries.
        # W_col will have non-zeros at C entries.
        
        # Let's list the (A_idx, B_idx) pairs and coefficients for M1..M7.
        # A_idx is (local_r, local_c). B_idx is (local_r, local_c).
        
        mults = [
            # (A_coeffs, B_coeffs, W_coeffs)
            # A_coeffs: dict {(r,c): val}
            # B_coeffs: dict {(r,c): val}
            # W_coeffs: dict {(r,c): val} for C entries
            
            # M1
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):-1, (1,0):-1, (0,1):1}), # Wait, C11=M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6
            # Let's just add the M contributions to W later.
            # For now, store the M definition.
        ]
        
        # Actually, it's easier to just hardcode the 7 vectors.
        # For each of the 7 multiplications, we add 1 column to U, V, W.
        # The W column will be the coefficient of that M in the final C entries.
        
        # M1
        # A: A00, A11 (1)
        # B: B00, B11 (1)
        # C: C00 (1), C11 (1)  <-- Wait, C22 has M1. C11 has M1.
        # C11 = M1 + M4 - M5 + M7
        # C22 = M1 - M2 + M3 + M6
        # So M1 contributes +1 to C00 and +1 to C11.
        
        # Let's define a helper to add a multiplication.
        def add_mult(U, V, W, col, A_coeffs, B_coeffs, C_coeffs):
            for (ar, ac), val in A_coeffs.items():
                global_r = A_row_offset + ar
                global_c = A_col_offset + ac
                idx = global_r * m + global_c
                U[idx, col] += val * sign
                
            for (br, bc), val in B_coeffs.items():
                global_r = B_row_offset + br
                global_c = B_col_offset + bc
                idx = global_r * p + global_c
                V[idx, col] += val * sign
                
            for (cr, cc), val in C_coeffs.items():
                global_r = C_row_offset + cr
                global_c = C_col_offset + cc
                idx = global_c * n + global_r
                W[idx, col] += val * sign
        
        # Define the 7 Strassen multiplications
        # M1
        add_mult(U, V, W, col_offset, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1})
        col_offset += 1
        # M2
        add_mult(U, V, W, col_offset, {(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1})
        col_offset += 1
        # M3
        add_mult(U, V, W, col_offset, {(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1})
        col_offset += 1
        # M4
        add_mult(U, V, W, col_offset, {(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1})
        col_offset += 1
        # M5
        add_mult(U, V, W, col_offset, {(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (0,0):-1})
        col_offset += 1
        # M6
        add_mult(U, V, W, col_offset, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1})
        col_offset += 1
        # M7
        add_mult(U, V, W, col_offset, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1})
        col_offset += 1
        
        return col_offset

    # Now apply for each output block column
    # Output block col 0: C[0:2, 0:2]
    # A0: A[0:2, 0:2], B_top0: B[0:2, 0:2]
    col = add_strassen_product(0, 0, 0, 0, 0, 0, 0)
    # A1: A[0:2, 2:4], B_bot0: B[2:4, 0:2]
    col = add_strassen_product(col, 0, 2, 2, 0, 0, 0)
    
    # Output block col 1: C[0:2, 2:4]
    # A0: A[0:2, 0:2], B_top1: B[0:2, 2:4]
    col = add_strassen_product(col, 0, 0, 0, 2, 0, 2)
    # A1: A[0:2, 2:4], B_bot1: B[2:4, 2:4]
    col = add_strassen_product(col, 0, 2, 2, 2, 0, 2)
    
    # Output block col 2: C[0:2, 4:6]
    # A0: A[0:2, 0:2], B_top2: B[0:2, 4:6]
    col = add_strassen_product(col, 0, 0, 0, 4, 0, 4)
    # A1: A[0:2, 2:4], B_bot2: B[2:4, 4:6]
    col = add_strassen_product(col, 0, 2, 2, 4, 0, 4)
    
    if col != rank:
        print(f"Warning: Strassen construction ended at col {col}, expected {rank}")
        
    return U, V, W

def _schoolbook_decompose(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    n, m, p = 2, 4, 6
    
    # Try Strassen first
    try:
        U, V, W = _strassen_decompose(n, m, p)
        if U is not None:
            if _verify(U, V, W, n, m, p, iters=20):
                # Convert to list of lists for interface compatibility?
                # The prompt says "factor matrices of shapes... with integer/half-integer entries"
                # The seed returns lists of lists.
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        pass
        
    # Fallback to schoolbook
    U, V, W = _schoolbook_decompose(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
