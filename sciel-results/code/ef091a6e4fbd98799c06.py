"""Program for matrix multiplication tensor rank minimization."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]
# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0  # safety margin before 120s kill
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m + j, j*p + k, k*n + i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Start with schoolbook rank 56
    best_rank = n * m * p  # 56
    best_factors = _schoolbook(n, m, p, best_rank)
    
    # Try structured Strassen construction
    # n=2: split into 1 row-block of size 1 (no 2x2 Strassen on n dimension)
    # m=4: split into 2 column-blocks of size 2 -> Strassen on m
    # p=7: split into 3 row-blocks (sizes 2,2,3) -> Strassen on p for 2x2 blocks
    
    # Strategy: Use Strassen where both dimensions >= 2
    # m=4 -> blocks of 2: 2 blocks
    # p=7 -> blocks of 2,2,3: 3 blocks
    # n=2 -> blocks of 2: 1 block (but Strassen needs 2x2, so n=2 works as one block)
    
    # For n=2, we can't do Strassen in n (need 2 blocks of size >=1, but Strassen is 2x2x2)
    # Actually Strassen works on 2x2x2 blocks. n=2 means we have 1 block of size 2 in n.
    # We need at least 2 blocks in each dimension for Strassen.
    # n=2: can split as 1+1 (two blocks of size 1) - but Strassen needs size >= 2 per block?
    # No, Strassen's 7 multiplications work for any 2x2x2 block regardless of sub-size.
    # Actually the standard Strassen is for 2x2 matrices. Let me reconsider.
    
    # Better approach: think of it as multiplying (2x4) by (4x7)
    # = (2x2 | 2x2) by (2x2; 2x2; 2x2; 1x2; 1x1) -- irregular
    
    # Let me use a cleaner decomposition:
    # Split A (2x4) into columns: A_left (2x2), A_right (2x2)
    # Split B (4x7) into rows: B_top (2x7), B_bot (2x7)
    # Result = A_left @ B_top + A_right @ B_bot
    # Each term is (2x2) @ (2x7) = 2x7
    # For (2x2)@(2x7): split B into (2x2)+(2x2)+(2x3)
    #   (2x2)@(2x2): Strassen rank 7
    #   (2x2)@(2x2): Strassen rank 7  
    #   (2x2)@(2x3): schoolbook 2*2*3=12
    # Total per half: 7+7+12 = 26
    # Two halves: 52
    
    # Even better: for (2x2)@(2x3), we can do better than schoolbook
    # (2x2)@(2x3): 4*3=12 schoolbook. Can we do Strassen-like?
    # Actually 2x2 times 2x3: treat as 2x2 @ [2x2 | 2x1]
    # = (2x2 Strassen 7) + (2x2)@(2x1) = 7 + 4 = 11
    # Or: 2x3 as 3 columns of 2x1. (2x2)@(2x1) needs 4 mults each, 3 cols = 12.
    # Hmm, let me just use the block approach systematically.
    
    # Let me try a more systematic Strassen-style decomposition:
    # A is 2x4, B is 4x7, C is 2x7
    
    # Decomposition 1: Split m=4 into 2+2
    # C = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:]
    # Each term: (2x2) @ (2x7)
    
    # For (2x2) @ (2x7):
    # Split p=7 into 2+2+3
    # (2x2)@(2x2): Strassen rank 7
    # (2x2)@(2x2): Strassen rank 7  
    # (2x2)@(2x3): need decomposition
    
    # For (2x2)@(2x3): 
    # Let A2 = [[a,b],[c,d]], B3 = [[e,f,g],[h,i,j]]
    # C = [[ae+bf, af+bg, ag+bj], [ce+df, cf+dg, cg+dj]]
    # 12 products in schoolbook.
    # Can we do better? The tensor rank of 2x2x3 multiplication...
    # Known: rank of 2x2x2 is 7 (Strassen). For 2x2x3, it's at most 7+4=11
    # (Strassen for first 2x2 block + 4 for the 2x1 remainder)
    # Or we could try: 2x2 @ 2x3 = 2x2 @ [2x2 | 2x1]
    # Strassen gives 7 for 2x2@2x2, and 2x2@2x1 needs 4 mults (each output is a linear combo)
    # Actually 2x2 @ 2x1: result is 2x1, each entry is a dot product of 2 elements = 2 mults each, 4 total.
    # So 7 + 4 = 11 for (2x2)@(2x3).
    
    # So (2x2)@(2x7) = 7 + 7 + 11 = 25
    # Total: 25 * 2 = 50
    
    # Let me implement this: rank 50 decomposition
    
    rank_50 = _strassen_decomposition(n, m, p)
    if rank_50 is not None:
        r = rank_50[0]
        U, V, W = rank_50[1], rank_50[2], rank_50[3]
        # Verify
        decomp = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(decomp, T, atol=1e-10):
            best_rank = r
            best_factors = (U, V, W)
    
    # Try to find even lower rank with SVD-based rank reduction
    if time.time() < deadline - 20:
        # Try various target ranks below best_rank
        for target_r in range(best_rank - 1, max(best_rank - 10, 27), -1):
            if time.time() > deadline - 10:
                break
            success, factors = _try_rank_reduction(n, m, p, T, best_factors, target_r, deadline)
            if success:
                best_rank = target_r
                best_factors = factors
                break
    
    U, V, W = best_factors
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _schoolbook(n, m, p, rank):
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W


def _strassen_decomposition(n, m, p):
    """
    Build a Strassen-based decomposition for (n x m) @ (m x p).
    n=2, m=4, p=7.
    
    Split m into 2+2:
      C = A_left(2x2) @ B_top(2x7) + A_right(2x2) @ B_bot(2x7)
    
    For each (2x2)@(2x7):
      Split p into 2+2+3:
        (2x2)@(2x2): Strassen (rank 7)
        (2x2)@(2x2): Strassen (rank 7)
        (2x2)@(2x3): Strassen(2x2) + schoolbook(2x1) = 7+4 = 11
    
    Total: (7+7+11)*2 = 50
    """
    if n != 2 or m != 4 or p != 7:
        return None
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        """Add a rank-1 term."""
        U_list.append((u_idx, u_val))
        V_list.append((v_idx, v_val))
        W_list.append((w_idx, w_val))
    
    def strassen_2x2x2(A_rows, B_rows, C_cols, row_offset=0, col_offset=0):
        """
        Add Strassen terms for (2x2)@(2x2) -> (2x2).
        A_rows: [a00, a01, a10, a11] indices in U
        B_rows: [b00, b01, b10, b11] indices in V  
        C_cols: [c00, c01, c10, c11] indices in W
        
        Strassen:
        m1 = (a00+a11)*(b00+b11)
        m2 = (a10+a11)*b00
        m3 = a00*(b01-b11)
        m4 = a11*(b10-b00)
        m5 = (a00+a01)*b11
        m6 = (a10-a00)*(b00+b01)
        m7 = (a01-a11)*(b10+b11)
        
        c00 = m1 + m4 - m5 + m7
        c01 = m3 + m5
        c10 = m2 + m4
        c11 = m1 - m2 + m3 + m6
        """
        # m1 = (a00+a11)*(b00+b11)
        add_term(A_rows[0]+A_rows[3], B_rows[0]+B_rows[3], C_cols[0]+C_cols[3], 1, 1, 1)
        # m1 contributes to c00 (+), c11 (+)
        # We handle by adding to each W position
        
        # Let me redo this more carefully.
        # Each Strassen multiplication creates a rank-1 term.
        # m1 = (a00+a11) * (b00+b11), contributes to c00 (+1), c11 (+1)
        # But a single rank-1 term U[:,r]*V[:,r]*W[:,r] contributes to ALL (i,j,k) simultaneously.
        # So we need to be more careful.
        
        # Actually, let me build the full factor matrices directly.
        pass
    
    # Let me build this differently - construct U, V, W matrices directly.
    
    total_rank = 50
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    
    # Helper: add schoolbook terms for (a x b) @ (b x c)
    def add_schoolbook(A_row_range, B_row_range, C_col_range, A_col_offset, B_col_offset, C_row_offset):
        """
        Add schoolbook decomposition for a sub-matrix multiplication.
        A_row_range: range of rows in A (original n x m matrix)
        B_row_range: range of rows in B (original m x p matrix)  
        C_col_range: range of columns in C (original n x p result)
        """
        nonlocal col
        for i in A_row_range:
            for j in B_row_range:
                for k in C_col_range:
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
    
    # Helper: add Strassen terms for (2x2)@(2x2)
    def add_strassen_2x2x2(i_offset, j_offset, k_offset):
        """
        Add Strassen 7-multiplication decomposition for A[i_offset:i_offset+2, j_offset:j_offset+2]
        @ B[j_offset:j_offset+2, k_offset:k_offset+2]
        -> C[i_offset:i_offset+2, k_offset:k_offset+2]
        
        A indices: (i_offset, j_offset), (i_offset, j_offset+1), (i_offset+1, j_offset), (i_offset+1, j_offset+1)
        B indices: (j_offset, k_offset), (j_offset, k_offset+1), (j_offset+1, k_offset), (j_offset+1, k_offset+1)
        C indices: (i_offset, k_offset), (i_offset, k_offset+1), (i_offset+1, k_offset), (i_offset+1, k_offset+1)
        """
        nonlocal col
        
        a00 = (i_offset, j_offset)
        a01 = (i_offset, j_offset+1)
        a10 = (i_offset+1, j_offset)
        a11 = (i_offset+1, j_offset+1)
        
        b00 = (j_offset, k_offset)
        b01 = (j_offset, k_offset+1)
        b10 = (j_offset+1, k_offset)
        b11 = (j_offset+1, k_offset+1)
        
        c00 = (i_offset, k_offset)
        c01 = (i_offset, k_offset+1)
        c10 = (i_offset+1, k_offset)
        c11 = (i_offset+1, k_offset+1)
        
        def U_idx(i, j): return i*m + j
        def V_idx(j, k): return j*p + k
        def W_idx(k, i): return k*n + i
        
        # m1 = (a00 + a11) * (b00 + b11)
        # U: a00 and a11 have coefficient 1
        U[U_idx(a00[0], a00[1]), col] = 1
        U[U_idx(a11[0], a11[1]), col] = 1
        V[V_idx(b00[0], b00[1]), col] = 1
        V[V_idx(b11[0], b11[1]), col] = 1
        # W: c00 (+1), c11 (+1)
        W[W_idx(c00[0], c00[1]), col] = 1
        W[W_idx(c11[0], c11[1]), col] = 1
        col += 1
        
        # m2 = (a10 + a11) * b00
        U[U_idx(a10[0], a10[1]), col] = 1
        U[U_idx(a11[0], a11[1]), col] = 1
        V[V_idx(b00[0], b00[1]), col] = 1
        # W: c10 (+1)
        W[W_idx(c10[0], c10[1]), col] = 1
        col += 1
        
        # m3 = a00 * (b01 - b11)
        U[U_idx(a00[0], a00[1]), col] = 1
        V[V_idx(b01[0], b01[1]), col] = 1
        V[V_idx(b11[0], b11[1]), col] = -1
        # W: c01 (+1)
        W[W_idx(c01[0], c01[1]), col] = 1
        col += 1
        
        # m4 = a11 * (b10 - b00)
        U[U_idx(a11[0], a11[1]), col] = 1
        V[V_idx(b10[0], b10[1]), col] = 1
        V[V_idx(b00[0], b00[1]), col] = -1
        # W: c00 (+1), c10 (+1)
        W[W_idx(c00[0], c00[1]), col] = 1
        W[W_idx(c10[0], c10[1]), col] = 1
        col += 1
        
        # m5 = (a00 + a01) * b11
        U[U_idx(a00[0], a00[1]), col] = 1
        U[U_idx(a01[0], a01[1]), col] = 1
        V[V_idx(b11[0], b11[1]), col] = 1
        # W: c00 (-1), c01 (+1)
        W[W_idx(c00[0], c00[1]), col] = -1
        W[W_idx(c01[0], c01[1]), col] = 1
        col += 1
        
        # m6 = (a10 - a00) * (b00 + b01)
        U[U_idx(a10[0], a10[1]), col] = 1
        U[U_idx(a00[0], a00[1]), col] = -1
        V[V_idx(b00[0], b00[1]), col] = 1
        V[V_idx(b01[0], b01[1]), col] = 1
        # W: c11 (+1)
        W[W_idx(c11[0], c11[1]), col] = 1
        col += 1
        
        # m7 = (a01 - a11) * (b10 + b11)
        U[U_idx(a01[0], a01[1]), col] = 1
        U[U_idx(a11[0], a11[1]), col] = -1
        V[V_idx(b10[0], b10[1]), col] = 1
        V[V_idx(b11[0], b11[1]), col] = 1
        # W: c00 (+1), c11 (-1)
        W[W_idx(c00[0], c00[1]), col] = 1
        W[W_idx(c11[0], c11[1]), col] = -1
        col += 1
    
    # Helper: add Strassen + schoolbook for (2x2)@(2x3)
    def add_strassen_2x2x3(i_offset, j_offset, k_offset):
        """
        (2x2)@(2x3) = (2x2)@(2x2) + (2x2)@(2x1)
        Strassen for first part (7 mults), schoolbook for second (4 mults)
        Total: 11
        """
        nonlocal col
        # First: Strassen for columns k_offset, k_offset+1
        add_strassen_2x2x2(i_offset, j_offset, k_offset)
        # Then: schoolbook for column k_offset+2
        add_schoolbook(
            range(i_offset, i_offset+2),
            range(j_offset, j_offset+2),
            range(k_offset+2, k_offset+3),
            0, 0, 0
        )
    
    # Now build the full decomposition:
    # C = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:]
    
    # Term 1: A[:,0:2] @ B[0:2, 0:2] -> C[:,0:2] (Strassen 2x2x2)
    add_strassen_2x2x2(0, 0, 0)
    
    # Term 2: A[:,0:2] @ B[0:2, 2:4] -> C[:,2:4] (Strassen 2x2x2)
    add_strassen_2x2x2(0, 0, 2)
    
    # Term 3: A[:,0:2] @ B[0:2, 4:7] -> C[:,4:7] (Strassen 2x2x3)
    add_strassen_2x2x3(0, 0, 4)
    
    # Term 4: A[:,2:4] @ B[2:4, 0:2] -> C[:,0:2] (Strassen 2x2x2)
    add_strassen_2x2x2(0, 2, 0)
    
    # Term 5: A[:,2:4] @ B[2:4, 2:4] -> C[:,2:4] (Strassen 2x2x2)
    add_strassen_2x2x2(0, 2, 2)
    
    # Term 6: A[:,2:4] @ B[2:4, 4:7] -> C[:,4:7] (Strassen 2x2x3)
    add_strassen_2x2x3(0, 2, 4)
    
    # col should be 50
    assert col == total_rank, f"Expected rank 50, got {col}"
    
    return (total_rank, U, V, W)


def _try_rank_reduction(n, m, p, T, current_factors, target_rank, deadline):
    """Try to reduce rank by finding a lower-rank decomposition."""
    U_cur, V_cur, W_cur = current_factors
    cur_rank = U_cur.shape[1]
    
    if target_rank >= cur_rank:
        return False, None
    
    # Use SVD on the tensor to find a potentially lower rank decomposition
    # Reshape tensor to (n*m, m*p*p*n) and do SVD
    T_mat = T.reshape(n*m, m*p * p*n)
    
    try:
        from scipy.linalg import svd
        U_svd, s, Vt_svd = svd(T_mat, full_matrices=False)
        
        # Check if we can truncate
        if s[target_rank] < 1e-10:
            # The tensor has rank <= target_rank in this matricization
            # But we need a rank-decomposition, not SVD
            pass
    except:
        pass
    
    # Alternative: try random initialization with gradient descent for the specific target rank
    from scipy.optimize import minimize
    
    size_U = n*m * target_rank
    size_V = m*p * target_rank
    size_W = p*n * target_rank
    total_size = size_U + size_V + size_W
    
    def objective(x):
        U = x[:size_U].reshape(n*m, target_rank)
        V = x[size_U:size_U+size_V].reshape(m*p, target_rank)
        W = x[size_U+size_V:].reshape(p*n, target_rank)
        decomposed = np.einsum('ar,br,cr->abc', U, V, W)
        residual = decomposed - T
        return np.sum(residual**2)
    
    # Initialize from current solution (pad with zeros if needed)
    np.random.seed(12345)
    
    max_restarts = 3
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        if restart == 0:
            # Initialize from current (truncate)
            x0 = np.zeros(total_size)
            min_r = min(target_rank, cur_rank)
            x0[:n*m*min_r] = U_cur[:, :min_r].flatten()
            x0[n*m*min_r:n*m*min_r + m*p*min_r] = V_cur[:, :min_r].flatten()
            x0[n*m*min_r + m*p*min_r:n*m*min_r + m*p*min_r + p*n*min_r] = W_cur[:, :min_r].flatten()
        else:
            x0 = np.random.randn(total_size) * 0.1
        
        result = minimize(objective, x0, method='L-BFGS-B',
                         options={'maxiter': 500, 'maxfun': 2000, 'disp': False})
        
        if result.fun < 1e-8:
            U = result.x[:size_U].reshape(n*m, target_rank)
            V = result.x[size_U:size_U+size_V].reshape(m*p, target_rank)
            W = result.x[size_U+size_V:].reshape(p*n, target_rank)
            decomposed = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(decomposed, T, atol=1e-8):
                return True, (U, V, W)
    
    return False, None

# EVOLVE-BLOCK-END
