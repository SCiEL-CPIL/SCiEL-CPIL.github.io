"""Matrix multiplication 4,4,5 tensor decomposition with integer-aware rank reduction."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2_terms():
    """
    Returns 7 terms for 2x2x2 Strassen.
    Each term is (u_sparse, v_sparse, w_sparse) where sparse is list of (index, value).
    U index: i*2+j
    V index: j*2+k
    W index: k*2+i
    """
    return [
        # M1 = (a+d)(e+h)
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        # M2 = (c+d)e
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        # M3 = a(f-h)
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        # M4 = d(g-e)
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        # M5 = (a+b)h
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        # M6 = (c-a)(e+f)
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        # M7 = (b-d)(g+h)
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]

def _construct_rank_65(n=4, m=4, p=5):
    """
    Constructs a rank-65 decomposition.
    49 terms from 4x4x4 Kronecker Strassen.
    16 terms from 4x4x1 Naive.
    """
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def get_u_idx(i, j): return i * m + j
    def get_v_idx(j, k): return j * p + k
    def get_w_idx(k, i): return k * n + i

    strassen_terms = _strassen_2x2x2_terms()
    
    # 1. 4x4x4 part (k=0..3)
    for (u1_sparse, v1_sparse, w1_sparse) in strassen_terms:
        for (u2_sparse, v2_sparse, w2_sparse) in strassen_terms:
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            
            for (idx1, val1) in u1_sparse:
                i1, j1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in u2_sparse:
                    i2, j2 = idx2 // 2, idx2 % 2
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    u[get_u_idx(i, j)] = val1 * val2
            
            for (idx1, val1) in v1_sparse:
                j1, k1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in v2_sparse:
                    j2, k2 = idx2 // 2, idx2 % 2
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    v[get_v_idx(j, k)] = val1 * val2
            
            for (idx1, val1) in w1_sparse:
                k1, i1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in w2_sparse:
                    k2, i2 = idx2 // 2, idx2 % 2
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    w[get_w_idx(k, i)] = val1 * val2
            
            add_term(u, v, w)
            
    # 2. 4x4x1 part (k=4)
    for i in range(n):
        for j in range(m):
            u = np.zeros(n*m); u[get_u_idx(i, j)] = 1
            v = np.zeros(m*p); v[get_v_idx(j, 4)] = 1
            w = np.zeros(p*n); w[get_w_idx(4, i)] = 1
            add_term(u, v, w)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat, V_mat, W_mat

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i*m+j, :]
                v_val = V_np[j*p+k, :]
                w_val = W_np[k*n+i, :]
                total = np.dot(u_val * v_val, w_val)
                if abs(total - 1.0) > tol:
                    return False
    return True

def _verify_integer_half_integer(U, V, W, tol=1e-6):
    """Check that all entries are integer or half-integer (multiple of 0.5)."""
    for M in [U, V, W]:
        for val in M:
            if abs(val * 2 - round(val * 2)) > tol:
                return False
    return True

def _build_schoolbook():
    """Fallback schoolbook decomposition with rank n*m*p = 80."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, column] = 1
                V[j*p+k, column] = 1
                W[k*n+i, column] = 1
                column += 1
    return U, V, W

def _compute_term_vectors(U, V, W, n, m, p):
    """Compute the flattened tensor vector for each rank-1 term."""
    rank = U.shape[1]
    num_entries = n * m * p
    vecs = np.zeros((num_entries, rank))
    
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        
        for i in range(n):
            for j in range(m):
                u_val = u[i*m+j]
                if u_val == 0: continue
                for k in range(p):
                    v_val = v[j*p+k]
                    if v_val == 0: continue
                    w_val = w[k*n+i]
                    if w_val == 0: continue
                    val = u_val * v_val * w_val
                    if val != 0:
                        flat_idx = (i * m + j) * p + k
                        vecs[flat_idx, r] = val
    return vecs

def _try_integer_aware_reduction(U, V, W, n, m, p, max_time=60):
    """
    Attempts to reduce rank by 1 by checking if removing a term allows the remaining
    terms to be scaled to half-integers while still summing to the target tensor.
    """
    start_time = time.time()
    rank = U.shape[1]
    target_flat = np.ones((n, m, p)).flatten()
    
    # Pre-compute term vectors for speed
    term_vecs = _compute_term_vectors(U, V, W, n, m, p)
    
    for r_remove in range(rank):
        if time.time() > start_time + max_time:
            break
        
        remaining = [x for x in range(rank) if x != r_remove]
        M_remaining = term_vecs[:, remaining]
        
        # Solve M c = target
        try:
            c, residuals, rank_m, sv = np.linalg.lstsq(M_remaining, target_flat, rcond=None)
        except Exception:
            continue
            
        if not residuals or len(residuals) == 0 or residuals[0] > 1e-8:
            continue
            
        # Check if scaling by c results in half-integer entries
        valid = True
        new_U = np.zeros((n*m, len(remaining)))
        new_V = np.zeros((m*p, len(remaining)))
        new_W = np.zeros((p*n, len(remaining)))
        
        for idx, r_idx in enumerate(remaining):
            # Scale factor is c[idx]
            # Check U
            u_scaled = U[:, r_idx] * c[idx]
            # Check if u_scaled is close to half-integers
            if not _is_half_integer_vector(u_scaled, tol=1e-5):
                valid = False
                break
            new_U[:, idx] = np.round(u_scaled * 2) / 2.0
            
            # Check V
            v_scaled = V[:, r_idx] * c[idx]
            if not _is_half_integer_vector(v_scaled, tol=1e-5):
                valid = False
                break
            new_V[:, idx] = np.round(v_scaled * 2) / 2.0
            
            # Check W
            w_scaled = W[:, r_idx] * c[idx]
            if not _is_half_integer_vector(w_scaled, tol=1e-5):
                valid = False
                break
            new_W[:, idx] = np.round(w_scaled * 2) / 2.0
        
        if valid:
            # Verify the snapped decomposition
            if _verify_decomposition(new_U, new_V, new_W, n, m, p, tol=1e-6):
                return new_U, new_V, new_W
                
    return None

def _structured_random_projection_elimination(U, V, W, n, m, p, max_time=90, n_random=50):
    """
    Uses structured random projection with iterative null-space elimination.
    
    Strategy:
    1. Add random half-integer perturbations to create a higher-rank system
    2. Find the null space of the term vector matrix using relative threshold
    3. Identify linearly dependent terms via null space vectors
    4. Eliminate terms (one or two at a time) while maintaining exactness
    5. Use a smooth log-sum-exp surrogate to guide which terms to eliminate
    """
    start_time = time.time()
    deadline = start_time + max_time
    rank = U.shape[1]
    target_flat = np.ones((n, m, p)).flatten()
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Pre-compute term vectors for the current decomposition
    term_vecs = _compute_term_vectors(U, V, W, n, m, p)
    
    # Try multiple random perturbations
    random.seed(42)
    
    for attempt in range(n_random):
        if time.time() > deadline - 5:
            break
        
        # Create random half-integer perturbations
        perturbed_U = U.copy()
        perturbed_V = V.copy()
        perturbed_W = W.copy()
        
        for r in range(rank):
            u_pert = np.random.randint(-1, 2, size=n*m) * 0.5
            v_pert = np.random.randint(-1, 2, size=m*p) * 0.5
            w_pert = np.random.randint(-1, 2, size=p*n) * 0.5
            
            scale = 0.1
            perturbed_U[:, r] += u_pert * scale
            perturbed_V[:, r] += v_pert * scale
            perturbed_W[:, r] += w_pert * scale
        
        # Recompute term vectors
        perturbed_term_vecs = _compute_term_vectors(perturbed_U, perturbed_V, perturbed_W, n, m, p)
        
        # Find null space of the term vector matrix
        try:
            _, s, vh = np.linalg.svd(perturbed_term_vecs, full_matrices=False)
        except Exception:
            continue
        
        # Use relative threshold: singular values below 1e-6 * max(s) are considered null
        if s.size == 0:
            continue
        s_max = s[0] if s[0] > 0 else 1.0
        null_threshold = max(1e-8, 1e-6 * s_max)
        null_indices = np.where(s < null_threshold)[0]
        
        if len(null_indices) == 0:
            continue
        
        # Use log-sum-exp surrogate to score null vectors
        # Higher norm in the null vector = more potential for elimination
        null_vecs = vh[null_indices, :]
        
        # Score each null vector by its L2 norm (smooth proxy for elimination potential)
        norms = np.linalg.norm(null_vecs, axis=1)
        
        # Sort by norm descending (most promising first)
        sorted_order = np.argsort(-norms)
        
        for null_idx_pos in sorted_order:
            if time.time() > deadline - 5:
                break
            
            null_vec = null_vecs[null_idx_pos]
            
            # Try eliminating 1 or 2 terms
            # Sort indices by |coefficient| in null vector
            abs_coeffs = np.abs(null_vec)
            sorted_term_idx = np.argsort(-abs_coeffs)
            
            # Try eliminating top 1
            elim_set = [sorted_term_idx[0]]
            remaining = [x for x in range(rank) if x not in elim_set]
            
            if _try_elimination(term_vecs, U, V, W, target_flat, n, m, p, remaining):
                new_U, new_V, new_W = _apply_elimination(term_vecs, U, V, W, target_flat, n, m, p, remaining)
                if new_U is not None:
                    best_U, best_V, best_W = new_U, new_V, new_W
                    best_rank = len(remaining)
                    U, V, W = new_U, new_V, new_W
                    rank = len(remaining)
                    term_vecs = _compute_term_vectors(U, V, W, n, m, p)
                    break
            
            # Try eliminating top 2
            if rank >= 3:
                elim_set = [sorted_term_idx[0], sorted_term_idx[1]]
                remaining = [x for x in range(rank) if x not in elim_set]
                
                if _try_elimination(term_vecs, U, V, W, target_flat, n, m, p, remaining):
                    new_U, new_V, new_W = _apply_elimination(term_vecs, U, V, W, target_flat, n, m, p, remaining)
                    if new_U is not None:
                        best_U, best_V, best_W = new_U, new_V, new_W
                        best_rank = len(remaining)
                        U, V, W = new_U, new_V, new_W
                        rank = len(remaining)
                        term_vecs = _compute_term_vectors(U, V, W, n, m, p)
                        break
        
        # If we made progress, restart the loop with the new decomposition
        if rank < best_rank:
            continue
    
    if best_rank < rank:
        return best_U, best_V, best_W
    
    return None

def _try_elimination(term_vecs, U, V, W, target_flat, n, m, p, remaining):
    """Check if eliminating specified terms yields a valid half-integer decomposition."""
    M_remaining = term_vecs[:, remaining]
    
    try:
        c, residuals, rank_m, sv = np.linalg.lstsq(M_remaining, target_flat, rcond=None)
    except Exception:
        return False
    
    if not residuals or len(residuals) == 0 or residuals[0] > 1e-8:
        return False
    
    # Check if the solution gives half-integer entries
    for idx, r_idx in enumerate(remaining):
        u_scaled = U[:, r_idx] * c[idx]
        if not _is_half_integer_vector(u_scaled, tol=1e-5):
            return False
        
        v_scaled = V[:, r_idx] * c[idx]
        if not _is_half_integer_vector(v_scaled, tol=1e-5):
            return False
        
        w_scaled = W[:, r_idx] * c[idx]
        if not _is_half_integer_vector(w_scaled, tol=1e-5):
            return False
    
    return True

def _apply_elimination(term_vecs, U, V, W, target_flat, n, m, p, remaining):
    """Apply elimination and return new matrices, or None if invalid."""
    M_remaining = term_vecs[:, remaining]
    
    try:
        c, residuals, rank_m, sv = np.linalg.lstsq(M_remaining, target_flat, rcond=None)
    except Exception:
        return None, None, None
    
    if not residuals or len(residuals) == 0 or residuals[0] > 1e-8:
        return None, None, None
    
    new_U = np.zeros((n*m, len(remaining)))
    new_V = np.zeros((m*p, len(remaining)))
    new_W = np.zeros((p*n, len(remaining)))
    
    for idx, r_idx in enumerate(remaining):
        u_scaled = U[:, r_idx] * c[idx]
        new_U[:, idx] = np.round(u_scaled * 2) / 2.0
        
        v_scaled = V[:, r_idx] * c[idx]
        new_V[:, idx] = np.round(v_scaled * 2) / 2.0
        
        w_scaled = W[:, r_idx] * c[idx]
        new_W[:, idx] = np.round(w_scaled * 2) / 2.0
    
    if _verify_decomposition(new_U, new_V, new_W, n, m, p, tol=1e-6):
        return new_U, new_V, new_W
    
    return None, None, None

def _is_half_integer_vector(vec, tol=1e-5):
    """Check if a vector contains only half-integers."""
    doubled = vec * 2
    rounded = np.round(doubled)
    if np.any(np.abs(doubled - rounded) > tol):
        return False
    return True

def _final_polish(U, V, W, n, m, p, max_time=10):
    """
    High-precision polish pass: attempt to snap near-half-integer entries and verify
    with tighter tolerance. Also tries removing redundant terms via exact rational check.
    """
    start_time = time.time()
    deadline = start_time + max_time
    rank = U.shape[1]
    target_flat = np.ones((n, m, p)).flatten()
    
    # First, snap all entries to nearest half-integer
    U_snapped = np.round(U * 2) / 2.0
    V_snapped = np.round(V * 2) / 2.0
    W_snapped = np.round(W * 2) / 2.0
    
    if not _verify_decomposition(U_snapped, V_snapped, W_snapped, n, m, p, tol=1e-9):
        # Snapping broke it; revert
        U_snapped, V_snapped, W_snapped = U.copy(), V.copy(), W.copy()
    
    # Try to remove redundant terms one at a time with exact verification
    term_vecs = _compute_term_vectors(U_snapped, V_snapped, W_snapped, n, m, p)
    current_rank = U_snapped.shape[1]
    
    for r_remove in range(current_rank):
        if time.time() > deadline:
            break
        
        remaining = [x for x in range(current_rank) if x != r_remove]
        M_remaining = term_vecs[:, remaining]
        
        try:
            c, residuals, rank_m, sv = np.linalg.lstsq(M_remaining, target_flat, rcond=None)
        except Exception:
            continue
        
        if not residuals or len(residuals) == 0 or residuals[0] > 1e-10:
            continue
        
        # Check half-integer property with tight tolerance
        valid = True
        new_U = np.zeros((n*m, len(remaining)))
        new_V = np.zeros((m*p, len(remaining)))
        new_W = np.zeros((p*n, len(remaining)))
        
        for idx, r_idx in enumerate(remaining):
            u_scaled = U_snapped[:, r_idx] * c[idx]
            if not _is_half_integer_vector(u_scaled, tol=1e-7):
                valid = False
                break
            new_U[:, idx] = np.round(u_scaled * 2) / 2.0
            
            v_scaled = V_snapped[:, r_idx] * c[idx]
            if not _is_half_integer_vector(v_scaled, tol=1e-7):
                valid = False
                break
            new_V[:, idx] = np.round(v_scaled * 2) / 2.0
            
            w_scaled = W_snapped[:, r_idx] * c[idx]
            if not _is_half_integer_vector(w_scaled, tol=1e-7):
                valid = False
                break
            new_W[:, idx] = np.round(w_scaled * 2) / 2.0
        
        if valid and _verify_decomposition(new_U, new_V, new_W, n, m, p, tol=1e-9):
            U_snapped, V_snapped, W_snapped = new_U, new_V, new_W
            current_rank = len(remaining)
            term_vecs = _compute_term_vectors(U_snapped, V_snapped, W_snapped, n, m, p)
    
    return U_snapped, V_snapped, W_snapped

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Approach 1: Strassen-based rank-65 construction
    U, V, W = _construct_rank_65(n, m, p)
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if construction fails (should not happen)
        U, V, W = _build_schoolbook()
        return {"U": [[float(x) for x in row] for row in U],
                "V": [[float(x) for x in row] for row in V],
                "W": [[float(x) for x in row] for row in W]}
    
    current_rank = U.shape[1]
    
    # Approach 2: Integer-aware rank reduction
    if time.time() < deadline - 10:
        try:
            reduced = _try_integer_aware_reduction(U, V, W, n, m, p, max_time=min(40, deadline - time.time() - 5))
            if reduced is not None:
                U, V, W = reduced
                current_rank = U.shape[1]
        except Exception:
            pass
    
    # Approach 3: Structured random projection with iterative null-space elimination
    if time.time() < deadline - 15 and current_rank > 60:
        try:
            reduced = _structured_random_projection_elimination(U, V, W, n, m, p, max_time=min(60, deadline - time.time() - 10))
            if reduced is not None:
                U, V, W = reduced
                current_rank = U.shape[1]
        except Exception:
            pass
    
    # Approach 4: Final high-precision polish pass
    if time.time() < deadline - 5:
        try:
            U, V, W = _final_polish(U, V, W, n, m, p, max_time=min(10, deadline - time.time() - 2))
            current_rank = U.shape[1]
        except Exception:
            pass
    
    # Convert to lists of floats
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
