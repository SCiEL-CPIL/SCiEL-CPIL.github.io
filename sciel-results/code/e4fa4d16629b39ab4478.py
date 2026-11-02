"""Program for 6x6x6 matrix multiplication rank minimization."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U,V,W form a valid decomposition of the matmul tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    # The decomposition gives: C[i,k] = sum_a U[i*m+j? no...]
    # Actually the convention is:
    # C[i,k] = sum_{j,a} A[i,j] * B[j,k]
    # The tensor T[i,j,k] = A[i,j]*B[j,k] should equal sum_a U[i*m+j, a] * V[j*p+k, a] * W[k*n+i, a]
    # Wait, let me re-read the convention: W[k*n+i, r]
    # So the reconstruction is: C[i,k] = sum_{j,a} A[i,j] * (sum_a U[i*m+j,a] * V[j*p+k,a] * W[k*n+i,a])
    # Actually the standard form is:
    # sum_a U[a, i,j] * V[a, j,k] * W[a, k,i] ... no.
    # Let me use the explicit convention from the task:
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # C[i,k] = sum_{j, a} A[i,j] * B[j,k] = sum_a (sum_j U[i*m+j, a] * A[i,j]) * (sum_j V[j*p+k, a] * B[j,k]) * W[k*n+i, a]?
    # No, that's not right either. The correct form for tensor decomposition of matmul:
    # T_{i,j,k} = sum_a U_{i,j,a} V_{j,k,a} W_{k,i,a}
    # where T_{i,j,k} = 1 if this is the matmul tensor... actually T_{i,j,k} represents the coefficient
    # In the matmul tensor, T_{i,j,k} = delta_{...} hmm.
    # The matmul tensor is: for computing C = A*B, C[i,k] = sum_j A[i,j] B[j,k]
    # So the "tensor" is M[i,j,k] = 1 for all i,j,k (it's the identity-like tensor that maps A,B to C)
    # Actually no. The tensor for matrix multiplication is defined as:
    # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
    # So T is a 3D array where T[i,j,k] = 1 for all valid i,j,k? No.
    # 
    # Let me think again. The matmul tensor has entries such that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This can be written as C[i,k] = sum_{j,a} A[i,j] * B[j,k] * delta_{something}
    # The tensor T[i,j,k] is defined so that C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    # But T[i,j,k] = 1 for all i,j,k doesn't work because we need the right indices.
    #
    # Actually, the standard formulation: the matmul tensor of size (n,m,p) has a rank-r decomposition:
    # T[i,j,k] = sum_{a=0}^{r-1} U[i,j,a] * V[j,k,a] * W[k,i,a]
    # where T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]B[j,k], each (i,j,k) triple contributes)
    # Wait no, T[i,j,k] should be 1 only when the indices are consistent. For matmul, 
    # T[i,j,k] = 1 for ALL i in [0,n), j in [0,m), k in [0,p). Because every combination (i,j,k) 
    # appears exactly once in the computation C[i,k] = sum_j A[i,j] B[j,k].
    #
    # So we need: sum_a U[i*m+j, a] * V[j*p+k, a] * W[k*n+i, a] = 1 for all i,j,k.
    #
    # Let's verify with schoolbook: U[i*m+j, col] = 1 where col = i*m*p + j*p + k
    # V[j*p+k, col] = 1 at same col
    # W[k*n+i, col] = 1 at same col
    # So for a given (i,j,k), the column col=i*m*p+j*p+k has U=1, V=1, W=1, giving product=1.
    # All other columns have at least one of U,V,W be 0 for this (i,j,k).
    # So sum = 1. Correct!
    
    # Check: for all i,j,k, sum_a U[i*m+j,a]*V[j*p+k,a]*W[k*n+i,a] == 1
    # This is expensive for large rank, so let's do it efficiently
    # Result[i,j,k] = sum_a U[i*m+j,a] * (V[j*p+k,a] * W[k*n+i,a])
    # We can compute this as a tensor contraction
    
    # For efficiency, skip full verification if rank is too large or time is tight
    if rank > 1000:
        return False  # Too large to verify quickly
    
    # Compute the reconstructed tensor
    # res[i,j,k] = sum_a U[i*m+j,a] * V[j*p+k,a] * W[k*n+i,a]
    # This is a 3-tensor contraction. Let's do it in chunks for memory.
    
    # Build the result tensor
    res = np.zeros((n, m, p), dtype=np.float64)
    
    # Process in chunks of a (rank dimension)
    chunk_size = min(50, rank)
    for a_start in range(0, rank, chunk_size):
        a_end = min(a_start + chunk_size, rank)
        U_chunk = U_arr[:, a_start:a_end]  # (n*m, chunk)
        V_chunk = V_arr[:, a_start:a_end]  # (m*p, chunk)
        W_chunk = W_arr[:, a_start:a_end]  # (p*n, chunk)
        
        # For each a in chunk, add outer product contribution
        for a in range(a_end - a_start):
            u = U_chunk[:, a]  # (n*m,)
            v = V_chunk[:, a]  # (m*p,)
            w = W_chunk[:, a]  # (p*n,)
            
            # Reshape u to (n, m), v to (m, p), w to (p, n)
            u_2d = u.reshape(n, m)
            v_2d = v.reshape(m, p)
            w_2d = w.reshape(p, n)
            
            # res[i,j,k] += u_2d[i,j] * v_2d[j,k] * w_2d[k,i]
            # This is an outer product of three 2D arrays with appropriate indices
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res[i, j, k] += u_2d[i, j] * v_2d[j, k] * w_2d[k, i]
    
    expected = np.ones((n, m, p), dtype=np.float64)
    max_err = np.max(np.abs(res - expected))
    return max_err < tol


def _strassen_decompose(n, m, p, depth=0, max_depth=2):
    """
    Build a Strassen-based decomposition.
    Returns (U, V, W) as lists of lists.
    
    For n=m=p=6, split into 3x3 blocks.
    Strassen uses 7 multiplications of 3x3 blocks.
    Each 3x3 multiplication can be done with schoolbook (rank 27) or further Strassen.
    """
    if n != m or m != p:
        # Fallback to schoolbook for non-cubic
        return _schoolbook(n, m, p)
    
    N = n
    
    if N <= 1 or depth >= max_depth:
        return _schoolbook(N, N, N)
    
    # Split into even-sized halves
    half = N // 2
    if N % 2 != 0:
        # For odd sizes, pad to next even
        # For N=6, this won't happen
        pass
    
    h = half
    r = N - 2 * h  # remainder (should be 0 for even N)
    
    # We'll handle the general case by padding
    if r > 0:
        # Pad to even
        Np = N + r
        U_pad, V_pad, W_pad = _strassen_decompose(Np, Np, Np, depth + 1, max_depth)
        # Truncate back
        U = [row[:U_pad[0][0] * (Np*Np // (N*N) if N*N>0 else 1)] for row in U_pad[:N*N]]
        # This is getting complicated. For N=6, h=3, r=0, so we're fine.
        return _schoolbook(N, N, N)
    
    # N is even, h = N/2
    # Block decomposition:
    # A = [[A11, A12], [A21, A22]]  each h x h
    # B = [[B11, B12], [B21, B22]]  each h x h
    # C = A*B = [[C11, C12], [C21, C22]]
    #
    # Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Recursively decompose each M_i (which is an h x h matmul)
    sub_decomps = []
    for i in range(7):
        sub_decomps.append(_strassen_decompose(h, h, h, depth + 1, max_depth))
    
    # Now build the global U, V, W
    # Each sub-decomp has shape (h*h, sub_rank)
    # Total rank = 7 * sub_rank
    sub_rank = sub_decomps[0][0][0]  # number of columns in first sub U
    total_rank = 7 * sub_rank
    
    U = [[0] * total_rank for _ in range(N * N)]
    V = [[0] * total_rank for _ in range(N * N)]
    W = [[0] * total_rank for _ in range(N * N)]
    
    # Helper to add a sub-decomposition contribution
    # For M_i = (A_combination)(B_combination), we need:
    # - U contribution: for each (i,j) in A_combination, add U_sub[i,j,a] to global U
    # - V contribution: for each (j,k) in B_combination, add V_sub[j,k,a] to global V  
    # - W contribution: for each (k,i) in output block, add W_sub[k,i,a] to global W
    
    # Define the combinations
    # M1: (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    # M2: (A21+A22)B11 -> C11 -= M2, C21 += M2, C22 -= M2
    # M3: A11(B12-B22) -> C11 += M3, C12 += M3
    # M4: A22(B21-B11) -> C11 += M4, C21 += M4
    # M5: (A11+A12)B22 -> C11 -= M5, C12 += M5
    # M6: (A21-A11)(B11+B12) -> C22 += M6
    # M7: (A11+A12? no, A12-A22)(B21+B22) -> C11 += M7
    
    # Block index mapping:
    # A11: rows 0..h-1, cols 0..h-1 -> index (i,j) with i,j in 0..h-1
    # A12: rows 0..h-1, cols h..N-1 -> index (i, h+j) with i,j in 0..h-1
    # A21: rows h..N-1, cols 0..h-1 -> index (h+i, j) with i,j in 0..h-1
    # A22: rows h..N-1, cols h..N-1 -> index (h+i, h+j) with i,j in 0..h-1
    # Similarly for B and C blocks.
    
    def add_contribution(sub_U, sub_V, sub_W, 
                         a_rows, a_cols, b_rows, b_cols, 
                         c_row_offset, c_col_offset,
                         sign, col_offset):
        """
        Add a sub-decomposition to the global U, V, W.
        a_rows, a_cols: lists of (row, col, sign) for A block contributions
        b_rows, b_cols: lists of (row, col, sign) for B block contributions
        c_row_offset, c_col_offset: offset for C block (which output block this M contributes to)
        sign: overall sign multiplier
        col_offset: starting column in global U,V,W
        """
        sr = len(sub_U[0])
        for a in range(sr):
            # U contributions
            for (ar, ac, s) in a_rows:
                # Global row index for A[ar, ac]
                global_row = ar * N + ac
                U[global_row][col_offset + a] += sign * s * sub_U[ar * h + ac][a]
            
            # V contributions
            for (br, bc, s) in b_rows:
                global_row = br * N + bc
                V[global_row][col_offset + a] += sign * s * sub_V[br * h + bc][a]
            
            # W contributions
            for (cr, cc, s) in c_rows:
                global_row = cr * N + cc
                W[global_row][col_offset + a] += sign * s * sub_W[cr * h + cc][a]
    
    # Let me restructure. For each M_i, I need to specify:
    # - Which A blocks contribute (with signs)
    # - Which B blocks contribute (with signs)
    # - Which C blocks receive the result (with signs)
    
    # For M1 = (A11+A22)(B11+B22):
    # A side: A11 (+1), A22 (+1)
    # B side: B11 (+1), B22 (+1)
    # C side: C11 (+1), C22 (+1)
    
    # For M2 = (A21+A22)B11:
    # A side: A21 (+1), A22 (+1)
    # B side: B11 (+1)
    # C side: C11 (-1), C21 (+1), C22 (-1)
    
    # For M3 = A11(B12-B22):
    # A side: A11 (+1)
    # B side: B12 (+1), B22 (-1)
    # C side: C11 (+1), C12 (+1)
    
    # For M4 = A22(B21-B11):
    # A side: A22 (+1)
    # B side: B21 (+1), B11 (-1)
    # C side: C11 (+1), C21 (+1)
    
    # For M5 = (A11+A12)B22:
    # A side: A11 (+1), A12 (+1)
    # B side: B22 (+1)
    # C side: C11 (-1), C12 (+1)
    
    # For M6 = (A21-A11)(B11+B12):
    # A side: A21 (+1), A11 (-1)
    # B side: B11 (+1), B12 (+1)
    # C side: C22 (+1)
    
    # For M7 = (A12-A22)(B21+B22):
    # A side: A12 (+1), A22 (-1)
    # B side: B21 (+1), B22 (+1)
    # C side: C11 (+1)
    
    # Define block positions
    # Block (bi, bj) where bi=0 means top half rows, bi=1 means bottom half
    # A block (bi,bj): rows [bi*h, (bi+1)*h), cols [bj*h, (bj+1)*h)
    
    def block_rows(bi, bj, size):
        """Return list of (global_row, local_row, local_col) for a block."""
        rows = []
        for lr in range(size):
            for lc in range(size):
                gr = bi * size + lr
                gc = bj * size + lc
                rows.append((gr, lr, lc))
        return rows
    
    # For each M_i, process
    M_defs = [
        # (a_blocks, b_blocks, c_blocks)
        # a_blocks: list of (block_idx, sign) where block_idx = (bi, bj)
        # b_blocks: same
        # c_blocks: same
        (
            [((0,0), 1), ((1,1), 1)],   # A11, A22
            [((0,0), 1), ((1,1), 1)],   # B11, B22
            [((0,0), 1), ((1,1), 1)]    # C11, C22
        ),
        (
            [((1,0), 1), ((1,1), 1)],   # A21, A22
            [((0,0), 1)],               # B11
            [((0,0), -1), ((1,0), 1), ((1,1), -1)]  # -C11, +C21, -C22
        ),
        (
            [((0,0), 1)],               # A11
            [((0,1), 1), ((1,1), -1)],  # B12, -B22
            [((0,0), 1), ((0,1), 1)]    # C11, C12
        ),
        (
            [((1,1), 1)],               # A22
            [((1,0), 1), ((0,0), -1)],  # B21, -B11
            [((0,0), 1), ((1,0), 1)]    # C11, C21
        ),
        (
            [((0,0), 1), ((0,1), 1)],   # A11, A12
            [((1,1), 1)],               # B22
            [((0,0), -1), ((0,1), 1)]   # -C11, +C12
        ),
        (
            [((1,0), 1), ((0,0), -1)],  # A21, -A11
            [((0,0), 1), ((0,1), 1)],   # B11, B12
            [((1,1), 1)]                # C22
        ),
        (
            [((0,1), 1), ((1,1), -1)],  # A12, -A22
            [((1,0), 1), ((1,1), 1)],   # B21, B22
            [((0,0), 1)]                # C11
        ),
    ]
    
    for mi in range(7):
        a_blocks, b_blocks, c_blocks = M_defs[mi]
        sub_U, sub_V, sub_W = sub_decomps[mi]
        col_start = mi * sub_rank
        
        # Build the effective U, V, W for this M_i by combining the block contributions
        # For the sub-decomp, sub_U[local_a_row, a] where local_a_row = la_i * h + la_j
        # We need to map each A block contribution to the global U
        
        # For each column a in the sub-decomp:
        for a in range(sub_rank):
            # U contribution: for each A block (bi,bj) with sign s
            for (bi, bj), s in a_blocks:
                for lr in range(h):
                    for lc in range(h):
                        global_row = bi * h + lr  # This is wrong, should be bi*h + lr for N=2h
                        # Wait, global row in A is: bi * h + lr (row within N)
                        # global col in A is: bj * h + lc
                        # So global index in U is: (bi*h + lr) * N + (bj*h + lc)
                        g_row = bi * h + lr
                        g_col = bj * h + lc
                        local_row = lr
                        local_col = lc
                        U[g_row * N + g_col][col_start + a] += s * sub_U[local_row * h + local_col][a]
            
            # V contribution: for each B block (bi,bj) with sign s
            for (bi, bj), s in b_blocks:
                for lr in range(h):
                    for lc in range(h):
                        g_row = bi * h + lr
                        g_col = bj * h + lc
                        local_row = lr
                        local_col = lc
                        V[g_row * N + g_col][col_start + a] += s * sub_V[local_row * h + local_col][a]
            
            # W contribution: for each C block (bi,bj) with sign s
            for (bi, bj), s in c_blocks:
                for lr in range(h):
                    for lc in range(h):
                        # W index is [k*n + i] = [col_of_C * N + row_of_C]
                        # C block (bi,bj): rows bi*h to (bi+1)*h, cols bj*h to (bj+1)*h
                        # So W index = col * N + row
                        g_col = bj * h + lc
                        g_row = bi * h + lr
                        local_row = lc  # This is the 'k' index (column of C)
                        local_col = lr  # This is the 'i' index (row of C)
                        # sub_W has shape (h*h, sub_rank), indexed by [k*h + i]
                        W[g_col * N + g_row][col_start + a] += s * sub_W[local_row * h + local_col][a]
    
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe within 120s limit
    
    n, m, p = DIMENSIONS
    best_result = None
    best_rank = float('inf')
    
    # Try schoolbook first as baseline
    try:
        U, V, W = _schoolbook(n, m, p)
        best_result = {"U": U, "V": V, "W": W}
        best_rank = n * m * p
    except Exception:
        pass
    
    # Try Strassen with depth 1 (one level of recursion, 3x3 schoolbook)
    try:
        if time.time() < deadline:
            U, V, W = _strassen_decompose(n, m, p, depth=0, max_depth=1)
            if len(U[0]) < best_rank:
                # Verify
                if _verify(U, V, W, n, m, p):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = len(U[0])
    except Exception:
        pass
    
    # Try Strassen with depth 2 (two levels, 3x3 -> 1x1 with Strassen not applicable, so 3x3 schoolbook)
    # Actually for 6 -> 3 -> 1, depth 2 would try Strassen on 3x3 which is odd, so it falls back to schoolbook.
    # Let's just try it.
    try:
        if time.time() < deadline:
            U, V, W = _strassen_decompose(n, m, p, depth=0, max_depth=2)
            if len(U[0]) < best_rank:
                if _verify(U, V, W, n, m, p):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = len(U[0])
    except Exception:
        pass
    
    return best_result

# EVOLVE-BLOCK-END
