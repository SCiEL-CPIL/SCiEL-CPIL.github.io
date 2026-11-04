import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u = i * m + k
                v = k * p + j
                w = j * n + i
                T[u, v, w] = 1.0
    return T

def _extract_rank1_from_residual(T, tol=1e-9):
    """
    Try to extract a rank-1 term from the residual tensor T.
    Returns (u, v, w) vectors if found, else None.
    Uses the approach: find a nonzero entry, then try to factor.
    """
    # Find a nonzero entry
    nz = np.nonzero(T)
    if len(nz[0]) == 0:
        return None
    
    # Get the first nonzero entry
    u0, v0, w0 = nz[0][0], nz[1][0], nz[2][0]
    
    # Try to extract a rank-1 term using the structure
    # A rank-1 term has T[u,v,w] = U[u]*V[v]*W[w]
    # We can try to find U, V, W by looking at the support
    
    # Get all nonzero entries
    u_idx, v_idx, w_idx = nz
    T_nz = T[u_idx, v_idx, w_idx]
    
    # For a rank-1 tensor, T[u,v,w] / T[u0,v,w] * T[u,v0,w] / T[u0,v0,w] should be consistent
    # Let's try a different approach: use SVD-like extraction
    
    # Pick a reference point
    # Find u, v, w such that T[u,v,w] != 0
    # Then U[u'] = T[u', v0, w0] / T[u0, v0, w0]
    # V[v'] = T[u0, v', w0] / T[u0, v0, w0]
    # W[w'] = T[u0, v0, w'] / T[u0, v0, w0]
    
    ref_val = T[u0, v0, w0]
    if abs(ref_val) < tol:
        return None
    
    # Extract U
    U = np.zeros(T.shape[0])
    for u in range(T.shape[0]):
        if abs(T[u, v0, w0]) > tol:
            U[u] = T[u, v0, w0] / ref_val
    
    # Extract V
    V = np.zeros(T.shape[1])
    for v in range(T.shape[1]):
        if abs(T[u0, v, w0]) > tol:
            V[v] = T[u0, v, w0] / ref_val
    
    # Extract W
    W = np.zeros(T.shape[2])
    for w in range(T.shape[2]):
        if abs(T[u0, v0, w]) > tol:
            W[w] = T[u0, v0, w] / ref_val
    
    # Check if this rank-1 term is actually contained in T
    # T should be >= U*V*W in some sense, or exactly equal on support
    rank1_term = np.einsum('u,v,w->uvw', U, V, W)
    
    # Check: T - rank1_term should have smaller support or be zero
    residual = T - rank1_term
    
    # Verify that rank1_term doesn't introduce negative values where T is zero
    # and that T - rank1_term is "simpler"
    
    # Count nonzeros
    nz_orig = len(nz[0])
    nz_resid = np.count_nonzero(np.abs(residual) > tol)
    
    # Only accept if residual has fewer nonzeros (or is zero)
    if nz_resid <= nz_orig:
        return (U, V, W)
    
    # Try other reference points
    # Sample a few more
    for trial in range(min(50, len(nz[0]))):
        idx = np.random.randint(0, len(nz[0]))
        u0_t, v0_t, w0_t = nz[0][idx], nz[1][idx], nz[2][idx]
        
        ref_val_t = T[u0_t, v0_t, w0_t]
        if abs(ref_val_t) < tol:
            continue
        
        U_t = np.zeros(T.shape[0])
        for u in range(T.shape[0]):
            if abs(T[u, v0_t, w0_t]) > tol:
                U_t[u] = T[u, v0_t, w0_t] / ref_val_t
        
        V_t = np.zeros(T.shape[1])
        for v in range(T.shape[1]):
            if abs(T[u0_t, v, w0_t]) > tol:
                V_t[v] = T[u0_t, v, w0_t] / ref_val_t
        
        W_t = np.zeros(T.shape[2])
        for w in range(T.shape[2]):
            if abs(T[u0_t, v0_t, w]) > tol:
                W_t[w] = T[u0_t, v0_t, w] / ref_val_t
        
        rank1_term_t = np.einsum('u,v,w->uvw', U_t, V_t, W_t)
        residual_t = T - rank1_term_t
        nz_resid_t = np.count_nonzero(np.abs(residual_t) > tol)
        
        if nz_resid_t < nz_orig:
            return (U_t, V_t, W_t)
    
    return None

def _greedy_reduce(T_target, max_iter=200, tol=1e-9):
    """
    Greedily extract rank-1 terms from the target tensor.
    Returns list of (U, V, W) tuples.
    """
    T = T_target.copy()
    terms = []
    
    for _ in range(max_iter):
        extracted = _extract_rank1_from_residual(T, tol)
        if extracted is None:
            break
        U, V, W = extracted
        terms.append((U, V, W))
        T -= np.einsum('u,v,w->uvw', U, V, W)
        
        # Check if T is now zero
        if np.max(np.abs(T)) < tol:
            break
    
    # Verify the decomposition
    T_reconstructed = np.zeros_like(T_target)
    for U, V, W in terms:
        T_reconstructed += np.einsum('u,v,w->uvw', U, V, W)
    
    if np.max(np.abs(T_reconstructed - T_target)) < 1e-6:
        return terms
    else:
        return None

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[u, v, w] = sum_r U[u,r]*V[v,r]*W[w,r] should equal 1
    iff there exist i,k,j with u=i*m+k, v=k*p+j, w=j*n+i (and consistent i,k,j).
    """
    rank = U.shape[1]
    # Build target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u = i * m + k
                v = k * p + j
                w = j * n + i
                T_target[u, v, w] = 1.0
    
    # Compute decomposition tensor
    T_decomp = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_decomp += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
    
    error = np.max(np.abs(T_decomp - T_target))
    return error < tol, error

def _to_halfint_check(U, V, W, tol=1e-9):
    """
    Check if all entries in U, V, W are close to integer or half-integer values.
    Returns True if they are.
    """
    for mat in [U, V, W]:
        for val in mat.flatten():
            if abs(val) < tol:
                continue
            # Check if close to integer or half-integer
            nearest_int = round(val)
            nearest_half = round(2*val)/2.0
            if abs(val - nearest_int) > tol and abs(val - nearest_half) > tol:
                return False
    return True

def _strassen_block(U, V, W, col, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, m, p, n):
    """
    Adds 7 rank-1 terms corresponding to a Strassen decomposition of a 2x2x2 block.
    """
    u_a = a_rows[0] * m + a_cols[0]
    u_b = a_rows[0] * m + a_cols[1]
    u_c = a_rows[1] * m + a_cols[0]
    u_d = a_rows[1] * m + a_cols[1]
    
    v_e = b_rows[0] * p + b_cols[0]
    v_f = b_rows[0] * p + b_cols[1]
    v_g = b_rows[1] * p + b_cols[0]
    v_h = b_rows[1] * p + b_cols[1]
    
    w_00 = c_cols[0] * n + c_rows[0]
    w_01 = c_cols[1] * n + c_rows[0]
    w_10 = c_cols[0] * n + c_rows[1]
    w_11 = c_cols[1] * n + c_rows[1]
    
    # M1: (a+d)(e+h)
    U[u_a, col] += 1.0
    U[u_d, col] += 1.0
    V[v_e, col] += 1.0
    V[v_h, col] += 1.0
    W[w_00, col] += 1.0
    W[w_11, col] += 1.0
    col += 1
    
    # M2: (c+d)e
    U[u_c, col] += 1.0
    U[u_d, col] += 1.0
    V[v_e, col] += 1.0
    W[w_10, col] += 1.0
    W[w_11, col] += -1.0
    col += 1
    
    # M3: a(f-h)
    U[u_a, col] += 1.0
    V[v_f, col] += 1.0
    V[v_h, col] += -1.0
    W[w_01, col] += 1.0
    W[w_11, col] += 1.0
    col += 1
    
    # M4: d(g-e)
    U[u_d, col] += 1.0
    V[v_g, col] += 1.0
    V[v_e, col] += -1.0
    W[w_00, col] += 1.0
    W[w_10, col] += 1.0
    col += 1
    
    # M5: (a+b)h
    U[u_a, col] += 1.0
    U[u_b, col] += 1.0
    V[v_h, col] += 1.0
    W[w_00, col] += -1.0
    W[w_01, col] += 1.0
    col += 1
    
    # M6: (c-a)(e+f)
    U[u_c, col] += 1.0
    U[u_a, col] += -1.0
    V[v_e, col] += 1.0
    V[v_f, col] += 1.0
    W[w_11, col] += 1.0
    col += 1
    
    # M7: (b-d)(g+h)
    U[u_b, col] += 1.0
    U[u_d, col] += -1.0
    V[v_g, col] += 1.0
    V[v_h, col] += 1.0
    W[w_00, col] += 1.0
    col += 1
    
    return col

def _construct_rank_88():
    """
    Construct the rank-88 decomposition using Strassen blocks.
    C = A * B where A is 3x4, B is 4x8, C is 3x8.
    
    Split A into A_top (2x4) and A_bottom (1x4).
    Split B into B_left (4x4) and B_right (4x4).
    
    C_top = A_top * B  (2x8)
    C_bottom = A_bottom * B  (1x8)
    
    For C_top:
      Split A_top into A_top_left (2x2) and A_top_right (2x2)
      Split B into B_left (4x4) and B_right (4x4)
      
      C_top = A_top_left * B_left + A_top_left * B_right + A_top_right * B_left + A_top_right * B_right
      
      Each 2x2 * 2x4 block can be done with Strassen on 2x2x2 sub-blocks.
      Actually, let's think of it differently.
      
      A_top is 2x4, B is 4x8.
      C_top[i,j] = sum_{k=0}^{3} A_top[i,k] * B[k,j]
      
      We can split the k-sum into two parts: k=0,1 and k=2,3.
      C_top = A_top[:, 0:2] * B[0:2, :] + A_top[:, 2:4] * B[2:4, :]
      
      Each term is (2x2) * (2x8) -> 2x8.
      For (2x2)*(2x8): split B[0:2,:] into B[0:2,0:2] and B[0:2,2:4], etc.
      Actually, (2x2)*(2x8) = (2x2)*(2x2) + (2x2)*(2x2) where the second factor's columns are split.
      
      Let me just use the approach from the original code:
      - For A_top[:, 0:2] * B[0:2, :]: use Strassen on 2x2x2 blocks for each pair of columns of B.
        B[0:2, :] has 8 columns, so 4 pairs of 2 columns. Each 2x2x2 block uses 7 terms. Total: 4*7=28.
      - For A_top[:, 2:4] * B[2:4, :]: similarly 4*7=28.
      - For A_bottom * B: this is (1x4)*(4x8) = 32 terms (schoolbook).
      
      Total: 28 + 28 + 32 = 88.
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    rank = 88
    U = np.zeros((n * m, rank), dtype=float)  # (12, 88)
    V = np.zeros((m * p, rank), dtype=float)  # (32, 88)
    W = np.zeros((p * n, rank), dtype=float)  # (24, 88)
    
    col = 0
    
    # Part 1a: A[0:2, 0:2] * B[0:2, :]
    # B[0:2, :] has columns 0..7, split into pairs: (0,1), (2,3), (4,5), (6,7)
    for k_start in range(0, 8, 2):
        col = _strassen_block(
            U, V, W, col,
            a_rows=[0, 1], a_cols=[0, 1],
            b_rows=[0, 1], b_cols=[k_start, k_start+1],
            c_rows=[0, 1], c_cols=[k_start, k_start+1],
            m=m, p=p, n=n
        )
    
    # Part 1b: A[0:2, 2:4] * B[2:4, :]
    for k_start in range(0, 8, 2):
        col = _strassen_block(
            U, V, W, col,
            a_rows=[0, 1], a_cols=[2, 3],
            b_rows=[2, 3], b_cols=[k_start, k_start+1],
            c_rows=[0, 1], c_cols=[k_start, k_start+1],
            m=m, p=p, n=n
        )
    
    # Part 2: A[2:3, :] * B[:, :]
    # A[2, k] * B[k, j] for k=0..3, j=0..7
    for k in range(4):
        for j in range(8):
            U[2 * m + k, col] = 1.0  # A[2, k]
            V[k * p + j, col] = 1.0  # B[k, j]
            W[j * n + 2, col] = 1.0  # C[2, j]
            col += 1
    
    if col != rank:
        raise Exception(f"Rank mismatch: expected {rank}, got {col}")
    
    # Verify
    valid, error = _verify_decomposition(U, V, W, n, m, p)
    if not valid:
        raise Exception(f"Verification failed with error {error}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _construct_rank_84():
    """
    Try to construct a rank-84 decomposition by merging some Strassen terms.
    
    Idea: In the two Strassen blocks for A_top[:, 0:2]*B[0:2,:] and A_top[:, 2:4]*B[2:4,:],
    some of the U factors are the same (e.g., a, b, c, d appear in both). 
    We might be able to share some of the 7 terms.
    
    Actually, the U factors for A_top[:, 0:2] are:
      a = A[0,0], b = A[0,1], c = A[1,0], d = A[1,1]
    And for A_top[:, 2:4]:
      a' = A[0,2], b' = A[0,3], c' = A[1,2], d' = A[1,3]
    
    These are different, so we can't directly merge.
    
    However, we can try a different decomposition. Let's think of the full problem as:
    C = A * B where A is 3x4, B is 4x8.
    
    Alternative: Split B into 4 row vectors of length 8: B[0,:], B[1,:], B[2,:], B[3,:].
    Then C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j] + A[i,2]*B[2,j] + A[i,3]*B[3,j].
    
    This is a sum of 4 outer products: A[:,k] * B[k,:].
    Each outer product has rank 1 in the tensor sense? No, the tensor for each term is:
    T_k[u, v, w] = 1 if u=i*m+k, v=k*p+j, w=j*n+i.
    This is a rank-1 tensor for each k? Let's check:
    T_k[i*m+k, k*p+j, j*n+i] = 1.
    For fixed k, this is a tensor of shape (n*m, m*p, p*n) with non-zeros at specific positions.
    Is it rank 1? A rank-1 tensor has the form u ⊗ v ⊗ w, i.e., T[u,v,w] = U[u]*V[v]*W[w].
    For our T_k, T_k[u,v,w] = 1 if u=i*m+k, v=k*p+j, w=j*n+i for some i,j.
    This is NOT of the form U[u]*V[v]*W[w] in general, because the constraint couples all three indices.
    
    So the schoolbook decomposition has rank n*m*p = 3*4*8 = 96.
    
    Strassen reduces this. The current rank-88 is already an improvement over 96.
    
    Can we do better? Let's think about the tensor rank of the 3x4x8 matrix multiplication.
    
    The tensor rank of T_{n,m,p} is at most n*m*p (schoolbook) and at least max(n*m, m*p, p*n) = max(12, 32, 24) = 32.
    Also, it's known that the tensor rank of T_{n,m,p} is at least n*m + m*p + p*n - n - m - p + 1 (from some bounds), but I'm not sure of the exact lower bound.
    
    For 3x4x8, the lower bound is at least 32. The current best is 88. Can we get closer to 32?
    
    Actually, I recall that the tensor rank of T_{n,m,p} is exactly n*m*p for small dimensions, but Strassen shows it's less for 2x2x2 (7 < 8). For larger dimensions, the rank is not known exactly.
    
    Given the time constraints, let's stick with the rank-88 solution and just make sure it's correct and fast.
    """
    # For now, just return the rank-88 solution
    return _construct_rank_88()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0
    
    # First, get the rank-88 solution as baseline
    result_rank88 = _construct_rank_88()
    
    # Try to find a lower-rank decomposition using greedy reduction
    # This is a heuristic search that may or may not find something better
    T_target = _build_target_tensor(n, m, p)
    
    # Set a time limit for the search
    search_deadline = start_time + 100.0  # Leave 20s margin
    
    try:
        # Try greedy reduction with limited iterations
        max_iter = 100
        terms = _greedy_reduce(T_target, max_iter=max_iter)
        
        if terms is not None and len(terms) < 88:
            # Check if entries are half-integers
            rank = len(terms)
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            for r, (U_r, V_r, W_r) in enumerate(terms):
                U[:, r] = U_r
                V[:, r] = V_r
                W[:, r] = W_r
            
            # Verify the decomposition
            valid, error = _verify_decomposition(U, V, W, n, m, p)
            if valid and _to_halfint_check(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # If greedy didn't find something better, try a more targeted approach
        # Try to merge some of the Strassen terms by looking for linear dependencies
        if time.time() < search_deadline:
            # Try to find a lower-rank decomposition by solving a system
            # This is more complex, so we'll just return the rank-88 solution
            pass
    
    except Exception as e:
        # If anything goes wrong, fall back to rank-88
        pass
    
    # Return the rank-88 solution
    return result_rank88
