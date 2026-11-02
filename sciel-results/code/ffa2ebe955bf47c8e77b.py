"""Matrix multiplication 2,2,11 - rank-minimal decomposition via structured search."""
import time
import numpy as np

DIMENSIONS = [2, 2, 11]

def _build_target_tensor():
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    n, m, p = 2, 2, 11
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T=None):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    if T is None:
        T = _build_target_tensor()
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T, T_decomp, atol=1e-9)

def _verify_fast(U, V, W, T):
    """Fast verification using precomputed tensor structure."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T, T_decomp, atol=1e-9)

def _build_strassen_decomposition():
    """
    Build a decomposition using Strassen for 2x2 blocks.
    For 2x2 * 2x11: use Strassen on pairs of columns.
    
    11 columns = 5 pairs + 1 single column
    Each Strassen block (2x2 * 2x2) has rank 7
    Each single column (2x2 * 2x1) has rank 4 (schoolbook for that column)
    
    Total: 5*7 + 4 = 39
    """
    n, m, p = 2, 2, 11
    num_pairs = p // 2  # 5 pairs
    remainder = p % 2   # 1 remaining column
    
    rank = num_pairs * 7 + remainder * 4
    U = np.zeros((n * m, rank), dtype=np.float64)  # (4, rank)
    V = np.zeros((m * p, rank), dtype=np.float64)  # (22, rank)
    W = np.zeros((p * n, rank), dtype=np.float64)  # (22, rank)
    
    col = 0
    for b in range(num_pairs):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        
        w_c00 = k0 * n + 0
        w_c01 = k1 * n + 0
        w_c10 = k0 * n + 1
        w_c11 = k1 * n + 1
        
        u_a = 0
        u_b = 1
        u_c = 2
        u_d = 3
        
        # m1 = (a+d)(e+h)
        U[u_a, col] = 1.0; U[u_d, col] = 1.0
        V[v_e, col] = 1.0; V[v_h, col] = 1.0
        W[w_c00, col] = 1.0; W[w_c11, col] = 1.0
        col += 1
        
        # m2 = (c+d)e
        U[u_c, col] = 1.0; U[u_d, col] = 1.0
        V[v_e, col] = 1.0
        W[w_c10, col] = 1.0; W[w_c11, col] = -1.0
        col += 1
        
        # m3 = a(f-h)
        U[u_a, col] = 1.0
        V[v_f, col] = 1.0; V[v_h, col] = -1.0
        W[w_c01, col] = 1.0; W[w_c11, col] = 1.0
        col += 1
        
        # m4 = d(g-e)
        U[u_d, col] = 1.0
        V[v_g, col] = 1.0; V[v_e, col] = -1.0
        W[w_c00, col] = 1.0; W[w_c10, col] = 1.0
        col += 1
        
        # m5 = (a+b)h
        U[u_a, col] = 1.0; U[u_b, col] = 1.0
        V[v_h, col] = 1.0
        W[w_c00, col] = -1.0; W[w_c01, col] = 1.0
        col += 1
        
        # m6 = (c-a)(e+f)
        U[u_c, col] = 1.0; U[u_a, col] = -1.0
        V[v_e, col] = 1.0; V[v_f, col] = 1.0
        W[w_c11, col] = 1.0
        col += 1
        
        # m7 = (b-d)(g+h)
        U[u_b, col] = 1.0; U[u_d, col] = -1.0
        V[v_g, col] = 1.0; V[v_h, col] = 1.0
        W[w_c00, col] = 1.0
        col += 1
    
    if remainder == 1:
        k2 = num_pairs * 2  # = 10
        
        v_e2 = k2
        v_g2 = p + k2
        w_c02 = k2 * n + 0
        w_c12 = k2 * n + 1
        
        U[0, col] = 1.0
        V[v_e2, col] = 1.0
        W[w_c02, col] = 1.0
        col += 1
        
        U[1, col] = 1.0
        V[v_g2, col] = 1.0
        W[w_c02, col] = 1.0
        col += 1
        
        U[2, col] = 1.0
        V[v_e2, col] = 1.0
        W[w_c12, col] = 1.0
        col += 1
        
        U[3, col] = 1.0
        V[v_g2, col] = 1.0
        W[w_c12, col] = 1.0
        col += 1
    
    return U, V, W

def _build_schoolbook():
    """Build the schoolbook rank-44 decomposition."""
    n, m, p = 2, 2, 11
    rank = n * m * p  # 44
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def _slice_matrices(U, V, W):
    """
    Build slice matrices for fast rank reduction.
    For each index a in U's domain, compute the matrix M_a = diag(U[a,:]) * (V^T * W)
    where V^T * W is (m*p, p*n) and we select the row of W for each a.
    
    Actually, for tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r],
    the slice T[a,:,:] = U[a,:] outer (V * W^T) ... no.
    
    Better: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    Fix a: T[a,b,c] = sum_r (U[a,r] * V[b,r]) * W[c,r]
    This is a sum of rank-1 matrices in (b,c) space.
    
    For rank reduction, we can use the approach:
    - The tensor has rank <= R iff for every fixed a, the matrix T[a,:,:] has rank <= R
    - More usefully, we can check if removing a column still gives a valid decomposition
      by checking if the resulting tensor matches.
    
    Fast approach: precompute the target slices and use residual checks.
    """
    n, m, p = 2, 2, 11
    dim_a = n * m  # 4
    dim_b = m * p  # 22
    dim_c = p * n  # 22
    rank = U.shape[1]
    
    # Precompute target tensor slices
    T_target = _build_target_tensor()
    
    # For fast checking: compute current tensor and compare
    # Use einsum for speed
    return T_target

def _fast_check(U, V, W, T_target):
    """Fast check if decomposition matches target."""
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_target, T_decomp, atol=1e-9)

def _rank_reduction_via_dependence(U, V, W, T_target, max_time=60):
    """
    Try to find linear dependencies among the rank-1 terms.
    If term r can be expressed as a linear combination of others, remove it.
    
    Key insight: T = sum_r U_r * V_r * W_r (outer products)
    If we can write U_r = sum_s alpha_s * U_s (and similarly for V, W),
    then term r is redundant.
    
    Practical approach: try removing each column and check.
    Use a smarter ordering: try columns that are "sparse" first.
    """
    start_time = time.time()
    current_rank = U.shape[1]
    if current_rank <= 1:
        return U, V, W
    
    # Sort columns by sparsity (number of non-zero entries) - sparser first
    sparsity_U = np.count_nonzero(U, axis=0)
    sparsity_V = np.count_nonzero(V, axis=0)
    sparsity_W = np.count_nonzero(W, axis=0)
    total_sparsity = sparsity_U + sparsity_V + sparsity_W
    order = np.argsort(total_sparsity)
    
    removed_any = True
    while removed_any and (time.time() - start_time) < max_time:
        removed_any = False
        current_rank = U.shape[1]
        for idx in order:
            if time.time() - start_time > max_time:
                break
            if idx >= current_rank:
                continue
            # Try removing column idx
            mask = np.ones(current_rank, dtype=bool)
            mask[idx] = False
            U_test = U[:, mask]
            V_test = V[:, mask]
            W_test = W[:, mask]
            if _fast_check(U_test, V_test, W_test, T_target):
                U, V, W = U_test, V_test, W_test
                removed_any = True
                # Recompute sparsity for remaining
                current_rank = U.shape[1]
                sparsity_U = np.count_nonzero(U, axis=0)
                sparsity_V = np.count_nonzero(V, axis=0)
                sparsity_W = np.count_nonzero(W, axis=0)
                total_sparsity = sparsity_U + sparsity_V + sparsity_W
                order = np.argsort(total_sparsity)
                break
    
    return U, V, W

def _try_rank_reduction(U, V, W, T, max_time=60):
    """
    Attempt to reduce rank by finding linearly dependent terms.
    Uses fast sparsity-ordered greedy removal.
    """
    start_time = time.time()
    current_rank = U.shape[1]
    
    if current_rank <= 1:
        return U, V, W
    
    # Use fast rank reduction
    U, V, W = _rank_reduction_via_dependence(U, V, W, T, max_time=max_time)
    
    return U, V, W

def _try_3col_block():
    """
    Try to construct a rank-10 decomposition for 2x2 * 2x3.
    If achievable, total = 4*7 + 10 = 38.
    
    For 2x2 * 2x3, the tensor has dimensions (4, 6, 6).
    Schoolbook rank = 12. Strassen gives 7+4=11 for (2x2*2x2) + (2x2*2x1).
    
    Let's try a direct approach: use the structure of the multiplication.
    A is 2x2, B is 2x3, C = A*B is 2x3.
    
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    Using Strassen on the first two columns (2x2*2x2, rank 7) + 
    schoolbook on the third column (2x2*2x1, rank 4) = rank 11.
    
    Can we do better? Try sharing some terms.
    """
    n, m, p = 2, 2, 3
    # Build 2x2 * 2x3 tensor
    T3 = np.zeros((4, 6, 6), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            for k in range(3):
                a = i * 2 + j
                b = j * 3 + k
                c = k * 2 + i
                T3[a, b, c] = 1.0
    
    # Strassen for cols 0,1 (rank 7) + schoolbook for col 2 (rank 4) = 11
    rank = 11
    U = np.zeros((4, rank), dtype=np.float64)
    V = np.zeros((6, rank), dtype=np.float64)
    W = np.zeros((6, rank), dtype=np.float64)
    
    # Strassen for 2x2 * 2x2 (cols 0,1)
    k0, k1 = 0, 1
    v_e, v_f = k0, k1
    v_g, v_h = 3 + k0, 3 + k1
    w_c00, w_c01 = k0*2+0, k1*2+0
    w_c10, w_c11 = k0*2+1, k1*2+1
    u_a, u_b, u_c, u_d = 0, 1, 2, 3
    
    col = 0
    U[u_a, col] = 1; U[u_d, col] = 1
    V[v_e, col] = 1; V[v_h, col] = 1
    W[w_c00, col] = 1; W[w_c11, col] = 1
    col += 1
    U[u_c, col] = 1; U[u_d, col] = 1
    V[v_e, col] = 1
    W[w_c10, col] = 1; W[w_c11, col] = -1
    col += 1
    U[u_a, col] = 1
    V[v_f, col] = 1; V[v_h, col] = -1
    W[w_c01, col] = 1; W[w_c11, col] = 1
    col += 1
    U[u_d, col] = 1
    V[v_g, col] = 1; V[v_e, col] = -1
    W[w_c00, col] = 1; W[w_c10, col] = 1
    col += 1
    U[u_a, col] = 1; U[u_b, col] = 1
    V[v_h, col] = 1
    W[w_c00, col] = -1; W[w_c01, col] = 1
    col += 1
    U[u_c, col] = 1; U[u_a, col] = -1
    V[v_e, col] = 1; V[v_f, col] = 1
    W[w_c11, col] = 1
    col += 1
    U[u_b, col] = 1; U[u_d, col] = -1
    V[v_g, col] = 1; V[v_h, col] = 1
    W[w_c00, col] = 1
    col += 1
    
    # Schoolbook for col 2
    k2 = 2
    v_e2, v_g2 = k2, 3 + k2
    w_c02, w_c12 = k2*2+0, k2*2+1
    
    U[0, col] = 1; V[v_e2, col] = 1; W[w_c02, col] = 1; col += 1
    U[1, col] = 1; V[v_g2, col] = 1; W[w_c02, col] = 1; col += 1
    U[2, col] = 1; V[v_e2, col] = 1; W[w_c12, col] = 1; col += 1
    U[3, col] = 1; V[v_g2, col] = 1; W[w_c12, col] = 1; col += 1
    
    if _verify(U, V, W, T3):
        return U, V, W, True
    return None, None, None, False

def _build_strassen_4pairs_3single():
    """
    Alternative: 4 Strassen pairs (2x2*2x2, rank 7 each) + 3 single columns (rank 4 each).
    Total: 4*7 + 3*4 = 28 + 12 = 40. Worse than 39.
    """
    return None

def _build_mixed_decomposition():
    """
    Try: 3 Strassen pairs (2x2*2x2, rank 7) + 1 block of 2x2*2x3 (rank 11) = 21 + 11 = 32?
    No, that's 3*7 + 11 = 32. But 2x2*2x3 has rank 11 with our construction.
    Wait: 3 pairs use 6 columns, remaining 5 columns as 2x2*2x5.
    2x2*2x5: Strassen pair (7) + Strassen pair (7) + single (4) = 18. Total = 21+18=39. Same.
    
    Or: 2x2*2x5 could potentially have rank < 18?
    2x2*2x5 schoolbook = 20. Strassen: 2 pairs + 1 single = 7+7+4=18.
    
    Let me try: 1 Strassen pair (cols 0,1) + 2x2*2x9 for remaining.
    2x2*2x9: 4 pairs + 1 single = 4*7+4 = 32. Total = 7+32=39. Same.
    
    The key question: can 2x2*2x11 be done in rank < 39?
    Lower bound: omega(2)*11/2 ... not straightforward.
    
    Let's just try the standard construction and fast rank reduction.
    """
    return None

def construct():
    start_time = time.time()
    DEADLINE = 145  # seconds, leave small margin before 150s limit
    
    T = _build_target_tensor()
    
    # Try the Strassen-based construction (rank 39)
    U, V, W = _build_strassen_decomposition()
    
    if _verify(U, V, W, T):
        # Try rank reduction with time budget
        remaining_time = DEADLINE - (time.time() - start_time) - 5
        if remaining_time > 3:
            U_red, V_red, W_red = _try_rank_reduction(U, V, W, T, max_time=min(remaining_time, 120))
            if U_red.shape[1] < U.shape[1]:
                return {'U': U_red.tolist(), 'V': V_red.tolist(), 'W': W_red.tolist()}
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Fall back to schoolbook rank 44
    U, V, W = _build_schoolbook()
    if _verify(U, V, W, T):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Should not reach here
    raise RuntimeError("No valid decomposition found")
