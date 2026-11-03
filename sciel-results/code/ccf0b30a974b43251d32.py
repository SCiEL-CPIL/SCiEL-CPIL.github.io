import time
import numpy as np

DEADLINE_S = 110.0
N, M, P = 4, 5, 5


def _validate(U, V, W, tol=1e-9):
    """Check sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1 for all i,j,k."""
    n, m, p = N, M, P
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                if abs(s - 1.0) > tol:
                    return False
    return True


def _strassen_2x2x2():
    """Return (U,V,W) for rank-7 Strassen decomposition of 2x2x2 matrix multiplication.
    U: (4,7), V: (4,7), W: (4,7) with W[k*2+i, r].
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[2, 0] = 1; W[1, 0] = -1; W[3, 0] = -1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[3, 4] = -1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[2, 6] = 1
    
    return U, V, W


def _strassen_4x4x4():
    """Return (U,V,W) for rank-49 Strassen decomposition of 4x4x4 matrix multiplication.
    U: (16,49), V: (16,49), W: (16,49) with W[k*4+i, r].
    """
    s2U, s2V, s2W = _strassen_2x2x2()
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    strassen_ops = [
        ({'A00': 1, 'A11': 1}, {'B00': 1, 'B11': 1}, {'C00': 1, 'C11': 1, 'C01': -1, 'C10': -1}),
        ({'A10': 1, 'A11': 1}, {'B00': 1}, {'C10': 1}),
        ({'A00': 1}, {'B10': 1, 'B11': -1}, {'C01': 1, 'C11': -1}),
        ({'A11': 1}, {'B01': 1, 'B00': -1}, {'C00': -1, 'C01': 1}),
        ({'A00': 1, 'A01': 1}, {'B11': 1}, {'C00': 1, 'C11': -1}),
        ({'A10': 1, 'A00': -1}, {'B00': 1, 'B10': 1}, {'C11': 1}),
        ({'A01': 1, 'A11': -1}, {'B01': 1, 'B11': 1}, {'C00': -1, 'C10': 1}),
    ]
    
    block_U = {
        'A00': (0, 4),
        'A01': (2, 6),
        'A10': (4, 8),
        'A11': (6, 10),
    }
    block_V = {
        'B00': (0, 4),
        'B01': (2, 6),
        'B10': (4, 8),
        'B11': (6, 10),
    }
    block_W = {
        'C00': (0, 4),
        'C01': (2, 6),
        'C10': (4, 8),
        'C11': (6, 10),
    }
    
    for op_idx, (u_blocks, v_blocks, w_blocks) in enumerate(strassen_ops):
        base_r = op_idx * 7
        for r2 in range(7):
            r = base_r + r2
            for blk, c in u_blocks.items():
                s, e = block_U[blk]
                for i2 in range(2):
                    for j2 in range(2):
                        U[s + i2*2 + j2, r] += c * s2U[i2*2 + j2, r2]
            
            for blk, c in v_blocks.items():
                s, e = block_V[blk]
                for j2 in range(2):
                    for k2 in range(2):
                        V[s + j2*2 + k2, r] += c * s2V[j2*2 + k2, r2]
            
            for blk, c in w_blocks.items():
                s, e = block_W[blk]
                for k2 in range(2):
                    for i2 in range(2):
                        W[s + k2*2 + i2, r] += c * s2W[k2*2 + i2, r2]
    
    return U, V, W


def _combine_decompositions(decomps):
    """Combine a list of (U,V,W) decompositions into one."""
    if not decomps:
        return None
    
    n, m, p = N, M, P
    total_rank = sum(U.shape[1] for U, V, W in decomps)
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    c = 0
    for U_i, V_i, W_i in decomps:
        r = U_i.shape[1]
        U[:, c:c+r] = U_i
        V[:, c:c+r] = V_i
        W[:, c:c+r] = W_i
        c += r
    
    return U, V, W


def _fast_peel(U, V, W, n, m, p, tol=1e-9):
    """Fast peeling using tensor-space linear algebra.
    Build matrix T where T[:,r] = vec(U[:,r] ⊗ V[:,r] ⊗ W[:,r]) reshaped.
    Find null space and remove dependent terms.
    """
    R = U.shape[1]
    if R <= 1:
        return U, V, W
    
    # Build the tensor contribution matrix: each column r is the vectorized
    # outer product u_r ⊗ v_r ⊗ w_r, which has size n*m*p
    # T has shape (n*m*p, R)
    # T[i*m+j, k*n+i', r] ... actually let's think about this differently.
    # 
    # The tensor T_{i,j,k} = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # We need T_{i,j,k} = delta(i,j,k) for the MM tensor... no wait.
    # 
    # Actually the MM tensor is T_{i,j,k} = 1 if the (i,j,k) entry is "active"
    # For MM: C[i,k] = sum_j A[i,j]*B[j,k], so the tensor is:
    # T_{i,j,k} = 1 for all i,j,k (it's the full MM tensor)
    # 
    # Wait no. The tensor for MM(n,m,p) has T_{i,j,k} = 1 for all i,j,k.
    # The decomposition is: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    #
    # So each term r contributes a tensor: u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    # This is a 3D tensor of size n x m x p.
    # 
    # To check linear dependence, I can flatten each term's tensor and stack them.
    # But the mapping from (i,j,k) to a single index needs to be consistent.
    # 
    # Let's use index = i*m*p + j*p + k for the tensor space (size n*m*p = 100)
    # Term r contributes: for each (i,j,k), value = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    tensor_dim = n * m * p
    T = np.zeros((tensor_dim, R))
    
    for r in range(R):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    T[idx, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Now find the null space of T. If there's a null vector c with T*c = 0,
    # then sum_r c_r * (term r) = 0 tensor, so we can remove terms.
    # 
    # Strategy: find a subset of columns that are linearly dependent.
    # Use QR with column pivoting to find the rank and identify dependent columns.
    
    # QR with pivoting
    Q, Rmat, piv = np.linalg.qr(T, mode='complete', pivoting=True)
    
    # Determine rank from Rmat
    tol_rank = tol * max(T.shape)
    rank_T = 0
    for i in range(min(T.shape)):
        if abs(Rmat[i, i]) > tol_rank:
            rank_T += 1
        else:
            break
    
    if rank_T >= R:
        # Full rank, no peeling possible
        return U, V, W
    
    # Columns beyond rank_T in the pivoted order are dependent
    # piv gives the column permutation
    # The first rank_T columns (in pivoted order) are independent
    # The rest are dependent
    
    # We want to remove the dependent columns.
    # In the original ordering, the dependent columns are:
    # piv[rank_T:] are the dependent column indices in original space
    
    keep_mask = np.ones(R, dtype=bool)
    for idx in piv[rank_T:]:
        keep_mask[idx] = False
    
    n_keep = keep_mask.sum()
    if n_keep == R:
        return U, V, W
    
    U_new = U[:, keep_mask]
    V_new = V[:, keep_mask]
    W_new = W[:, keep_mask]
    
    # Validate
    if _validate(U_new, V_new, W_new, tol=1e-5):
        return U_new, V_new, W_new
    
    # If validation fails (shouldn't happen if math is right), return original
    return U, V, W


def _iterative_peel(U, V, W, n, m, p, time_limit=30.0):
    """Iteratively peel until no more reduction."""
    t0 = time.time()
    U_cur, V_cur, W_cur = U, V, W
    
    for _ in range(100):
        if time.time() - t0 > time_limit:
            break
        
        U_new, V_new, W_new = _fast_peel(U_cur, V_cur, W_cur, n, m, p)
        if U_new.shape[1] >= U_cur.shape[1]:
            break
        
        U_cur, V_cur, W_cur = U_new, V_new, W_new
    
    return U_cur, V_cur, W_cur


def construct():
    t0 = time.time()
    n, m, p = N, M, P  # 4, 5, 5
    
    # Deadline guard
    if time.time() - t0 > DEADLINE_S:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strategy: Split the inner dimension (5) into 1+4.
    decomps = []
    
    def make_schoolbook(n_b, m_b, p_b, i_off, j_off, k_off):
        """Create schoolbook for a block."""
        rank = n_b * m_b * p_b
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U[(i_off+i)*m + (j_off+j), c] = 1.0
                    V[(j_off+j)*p + (k_off+k), c] = 1.0
                    W[(k_off+k)*n + (i_off+i), c] = 1.0
                    c += 1
        return U, V, W
    
    # A1*B11: A[0:4,0:1] * B[0:1,0:1] -> C[0:4,0:1]
    decomps.append(make_schoolbook(4, 1, 1, 0, 0, 0))
    
    # A1*B12: A[0:4,0:1] * B[0:1,1:5] -> C[0:4,1:5]
    decomps.append(make_schoolbook(4, 1, 4, 0, 0, 1))
    
    # A2*B21: A[0:4,1:5] * B[1:5,0:1] -> C[0:4,0:1]
    decomps.append(make_schoolbook(4, 4, 1, 0, 1, 0))
    
    # A2*B22: A[0:4,1:5] * B[1:5,1:5] -> C[0:4,1:5]
    s4U, s4V, s4W = _strassen_4x4x4()
    U_embed = np.zeros((n*m, 49))
    V_embed = np.zeros((m*p, 49))
    W_embed = np.zeros((p*n, 49))
    
    for i in range(4):
        for j in range(4):
            U_embed[i*5 + (1+j), :] = s4U[i*4+j, :]
    for j in range(4):
        for k in range(4):
            V_embed[(1+j)*5 + (1+k), :] = s4V[j*4+k, :]
    for k in range(4):
        for i in range(4):
            W_embed[(1+k)*4 + i, :] = s4W[k*4+i, :]
    
    decomps.append((U_embed, V_embed, W_embed))
    
    # Combine all decompositions
    U, V, W = _combine_decompositions(decomps)
    
    # Deadline guard
    if time.time() - t0 > DEADLINE_S - 10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Validate
    if not _validate(U, V, W):
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank via fast peeling
    if time.time() - t0 < DEADLINE_S - 15:
        remaining = DEADLINE_S - 15 - (time.time() - t0)
        U_red, V_red, W_red = _iterative_peel(U, V, W, n, m, p, time_limit=remaining)
        if U_red.shape[1] < U.shape[1]:
            U, V, W = U_red, V_red, W_red
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
