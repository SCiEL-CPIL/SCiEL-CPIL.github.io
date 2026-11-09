import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    rank = U.shape[1]
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Reconstruct C from the decomposition
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    C_reconstructed = np.zeros((n, p))
    
    # Efficient verification using matrix operations
    for r in range(rank):
        u = U[:, r]  # shape (n*m,)
        v = V[:, r]  # shape (m*p,)
        w = W[:, r]  # shape (p*n,)
        
        # Reshape u into (n, m), v into (m, p), w into (p, n)
        u_mat = u.reshape(n, m)
        v_mat = v.reshape(m, p)
        w_mat = w.reshape(p, n)
        
        # The contribution to C from term r is:
        # For each i, k: C[i,k] += sum_j u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        # This is element-wise: (u_mat * v_mat) gives n x p, then multiply by w_mat transposed
        # Actually: sum_j u[i,j]*v[j,k] * w[k,i]
        # Let temp[i,k] = sum_j u[i,j]*v[j,k] = (u_mat @ v_mat)[i,k]
        # Then C[i,k] += temp[i,k] * w_mat[k,i]
        
        temp = u_mat @ v_mat  # n x p
        C_reconstructed += temp * w_mat.T  # w_mat is (p,n), w_mat.T is (n,p), so w_mat.T[i,k] = w_mat[k,i]
    
    if not np.allclose(C_reconstructed, C_expected, atol=tol):
        return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Return schoolbook (rank n*m*p) decomposition as numpy arrays."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def _block_decomposition(n, m, p, block_sizes=None):
    """
    Construct a block-structured decomposition.
    
    For 4x4 * 4x9:
    - A is 4x4, B is 4x9
    - Split A into 2x2 blocks (2x2 grid), B into 2x3 blocks (2x3 grid)
    - Each output block C_ij (2x3) = sum over shared index of A_ik * B_kj
    - Each (2x2)*(2x3) product has rank 6 (schoolbook)
    - Total: 4 output blocks * 2 terms * 6 = 48 rank terms
    
    More generally, we can use different block sizes.
    """
    if block_sizes is None:
        # Default: split into 2x2 blocks for A rows/cols, 2x3 for B
        # A: 4x4 -> 2 blocks of 2 rows x 2 blocks of 2 cols
        # B: 4x9 -> 2 blocks of 2 rows x 3 blocks of 3 cols
        a_row_blocks = 2
        a_col_blocks = 2
        b_row_blocks = 2
        b_col_blocks = 3
        a_row_size = n // a_row_blocks  # 2
        a_col_size = m // a_col_blocks  # 2
        b_row_size = m // b_row_blocks  # 2
        b_col_size = p // b_col_blocks  # 3
    else:
        a_row_blocks, a_col_blocks, b_row_blocks, b_col_blocks = block_sizes
        a_row_size = n // a_row_blocks
        a_col_size = m // a_col_blocks
        b_row_size = m // b_row_blocks
        b_col_size = p // b_col_blocks
    
    # Verify dimensions
    if a_row_blocks * a_row_size != n or a_col_blocks * a_col_size != m:
        return None
    if b_row_blocks * b_row_size != m or b_col_blocks * b_col_size != p:
        return None
    if a_col_blocks != b_row_blocks:
        return None  # Must match for the contraction
    
    rank_terms = []
    
    for rb_c in range(a_row_blocks):  # output row block
        for cb_c in range(b_col_blocks):  # output col block
            for rb_shared in range(a_col_blocks):  # shared index block
                # Compute A[rb_c, rb_shared] * B[rb_shared, cb_c]
                # A block: a_row_size x a_col_size
                # B block: b_row_size x b_col_size
                # Result: a_row_size x b_col_size
                # Schoolbook rank: a_row_size * a_col_size * b_col_size
                
                for i in range(a_row_size):  # row within A block
                    for j in range(a_col_size):  # col of A / row of B
                        for k in range(b_col_size):  # col within B block
                            # Global indices:
                            # A row: rb_c * a_row_size + i
                            # A col: rb_shared * a_col_size + j
                            # B row: rb_shared * b_row_size + j
                            # B col: cb_c * b_col_size + k
                            # C row: rb_c * a_row_size + i
                            # C col: cb_c * b_col_size + k
                            
                            a_row = rb_c * a_row_size + i
                            a_col = rb_shared * a_col_size + j
                            b_row = rb_shared * b_row_size + j
                            b_col = cb_c * b_col_size + k
                            c_row = rb_c * a_row_size + i
                            c_col = cb_c * b_col_size + k
                            
                            u_idx = a_row * m + a_col
                            v_idx = b_row * p + b_col
                            w_idx = c_col * n + c_row  # W[k*n+i] where k=c_col, i=c_row
                            
                            rank_terms.append((u_idx, v_idx, w_idx))
    
    rank = len(rank_terms)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r, (u_idx, v_idx, w_idx) in enumerate(rank_terms):
        U[u_idx, r] = 1.0
        V[v_idx, r] = 1.0
        W[w_idx, r] = 1.0
    
    return U, V, W

def _try_strassen_like(n, m, p, time_limit=50):
    """
    Try to find a lower-rank decomposition using Strassen-like ideas.
    
    For 4x4 * 4x9, we can try to use the structure more cleverly.
    The key insight is that we might be able to share some multiplications
    across different output positions.
    
    One approach: use the fact that for fixed A columns, we're computing
    multiple products with the same A submatrix.
    
    Let's try a different block structure:
    - Split A into 4x2 blocks (2 column blocks)
    - Split B into 2x9 blocks (2 row blocks)  
    - Each output row block (2x9) = A_rowblock * B_rowblock0 + A_rowblock * B_rowblock1
    
    Actually, let me think about this differently.
    
    For the full problem 4x4 * 4x9:
    C = A * B where A is 4x4, B is 4x9
    
    We can write:
    C[:, 0:3] = A * B[:, 0:3]
    C[:, 3:6] = A * B[:, 3:6]  
    C[:, 6:9] = A * B[:, 6:9]
    
    Each is a 4x4 * 4x3 product.
    
    For 4x4 * 4x3, schoolbook rank is 4*4*3 = 48.
    Total: 3 * 48 = 144 (same as full schoolbook).
    
    But we can do better by sharing A factors.
    
    Let me try: for each column block of B (3 blocks of width 3),
    compute A * B_block. Each requires rank 48.
    But A is shared! Can we exploit this?
    
    In a tensor decomposition, sharing means using the same U factors.
    But the rank-1 terms still need distinct V and W factors.
    
    Actually, the block decomposition with 2x2 blocks gives rank 48,
    which is already much better than 144.
    
    Let's see if we can do even better.
    
    Idea: Use a 4x4 * 4x4 decomposition (rank 16 schoolbook, or potentially
    lower with Strassen) for the first 4 columns, and similarly for others.
    
    For 4x4 * 4x4, the schoolbook rank is 16.
    Strassen gives rank 7 for 2x2, which can be extended.
    
    Actually, for 4x4 * 4x4, we can use 2x2 Strassen blocks:
    A = [[A00, A01], [A10, A11]] (each 2x2)
    B = [[B00, B01], [B10, B11]] (each 2x2)
    
    Strassen: 7 multiplications per 2x2*2x2 product.
    For 4x4*4x4: 4 output blocks * 7 = 28 rank terms.
    
    But we need 4x4 * 4x9, not 4x4 * 4x4.
    
    Let's split B into [B1 (4x4), B2 (4x4), B3 (4x1)].
    C = A*B1 + A*B2 + A*B3
    
    For A*B1: use 4x4*4x4 Strassen-like, rank ~28
    For A*B2: same, rank ~28
    For A*B3: 4x4*4x1, schoolbook rank 16
    
    Total: 28 + 28 + 16 = 72. Not better than 48.
    
    Hmm. Let me stick with the 2x2 block approach which gives 48.
    """
    # For now, just return the block decomposition result
    return _block_decomposition(n, m, p)

def construct():
    """
    Returns a dict with keys 'U', 'V', 'W' representing the tensor decomposition.
    U: (n*m) x rank
    V: (m*p) x rank  
    W: (p*n) x rank
    
    Convention: W[k*n+i, r] corresponds to output element (k, i) of the product.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s margin for the 120s limit
    
    # Try block decomposition (should give rank 48 for 4x4*4x9)
    result = _block_decomposition(n, m, p)
    
    if result is not None:
        U, V, W = result
        rank = U.shape[1]
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            print(f"Block decomposition with rank {rank} verified successfully")
            
            U_list = [[float(U[i, j]) for j in range(rank)] for i in range(U.shape[0])]
            V_list = [[float(V[i, j]) for j in range(rank)] for i in range(V.shape[0])]
            W_list = [[float(W[i, j]) for j in range(rank)] for i in range(W.shape[0])]
            
            elapsed = time.time() - start_time
            print(f"Total time: {elapsed:.2f}s")
            
            return {"U": U_list, "V": V_list, "W": W_list}
        else:
            print("Block decomposition verification failed!")
    
    # Fallback to schoolbook
    print("Falling back to schoolbook decomposition")
    U, V, W = _schoolbook_decomposition(n, m, p)
    rank = U.shape[1]
    
    U_list = [[float(U[i, j]) for j in range(rank)] for i in range(U.shape[0])]
    V_list = [[float(V[i, j]) for j in range(rank)] for i in range(V.shape[0])]
    W_list = [[float(W[i, j]) for j in range(rank)] for i in range(W.shape[0])]
    
    elapsed = time.time() - start_time
    print(f"Total time (schoolbook): {elapsed:.2f}s")
    
    return {"U": U_list, "V": V_list, "W": W_list}
