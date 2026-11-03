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

def _is_half_integer_vector(vec, tol=1e-5):
    """Check if a vector contains only half-integers."""
    doubled = vec * 2
    rounded = np.round(doubled)
    if np.any(np.abs(doubled - rounded) > tol):
        return False
    return True

def _try_remove_single_term(U, V, W, n, m, p, r_remove, tol=1e-8):
    """
    Try to remove term r_remove. If the remaining terms can be scaled to half-integers
    while still summing to the target tensor, return the new decomposition.
    """
    rank = U.shape[1]
    target_flat = np.ones((n, m, p)).flatten()
    
    term_vecs = _compute_term_vectors(U, V, W, n, m, p)
    remaining = [x for x in range(rank) if x != r_remove]
    M_remaining = term_vecs[:, remaining]
    
    try:
        c, residuals, rank_m, sv = np.linalg.lstsq(M_remaining, target_flat, rcond=None)
    except Exception:
        return None
        
    if not residuals or len(residuals) == 0 or residuals[0] > tol:
        return None
        
    # Check if scaling by c results in half-integer entries
    valid = True
    new_U = np.zeros((n*m, len(remaining)))
    new_V = np.zeros((m*p, len(remaining)))
    new_W = np.zeros((p*n, len(remaining)))
    
    for idx, r_idx in enumerate(remaining):
        u_scaled = U[:, r_idx] * c[idx]
        if not _is_half_integer_vector(u_scaled, tol=1e-5):
            valid = False
            break
        new_U[:, idx] = np.round(u_scaled * 2) / 2.0
        
        v_scaled = V[:, r_idx] * c[idx]
        if not _is_half_integer_vector(v_scaled, tol=1e-5):
            valid = False
            break
        new_V[:, idx] = np.round(v_scaled * 2) / 2.0
        
        w_scaled = W[:, r_idx] * c[idx]
        if not _is_half_integer_vector(w_scaled, tol=1e-5):
            valid = False
            break
        new_W[:, idx] = np.round(w_scaled * 2) / 2.0
    
    if valid:
        if _verify_decomposition(new_U, new_V, new_W, n, m, p, tol=1e-6):
            return new_U, new_V, new_W
    return None

def _try_remove_two_terms(U, V, W, n, m, p, r_remove1, r_remove2, tol=1e-8):
    """
    Try to remove terms r_remove1 and r_remove2 simultaneously.
    """
    rank = U.shape[1]
    target_flat = np.ones((n, m, p)).flatten()
    
    term_vecs = _compute_term_vectors(U, V, W, n, m, p)
    remaining = [x for x in range(rank) if x != r_remove1 and x != r_remove2]
    M_remaining = term_vecs[:, remaining]
    
    try:
        c, residuals, rank_m, sv = np.linalg.lstsq(M_remaining, target_flat, rcond=None)
    except Exception:
        return None
        
    if not residuals or len(residuals) == 0 or residuals[0] > tol:
        return None
        
    valid = True
    new_U = np.zeros((n*m, len(remaining)))
    new_V = np.zeros((m*p, len(remaining)))
    new_W = np.zeros((p*n, len(remaining)))
    
    for idx, r_idx in enumerate(remaining):
        u_scaled = U[:, r_idx] * c[idx]
        if not _is_half_integer_vector(u_scaled, tol=1e-5):
            valid = False
            break
        new_U[:, idx] = np.round(u_scaled * 2) / 2.0
        
        v_scaled = V[:, r_idx] * c[idx]
        if not _is_half_integer_vector(v_scaled, tol=1e-5):
            valid = False
            break
        new_V[:, idx] = np.round(v_scaled * 2) / 2.0
        
        w_scaled = W[:, r_idx] * c[idx]
        if not _is_half_integer_vector(w_scaled, tol=1e-5):
            valid = False
            break
        new_W[:, idx] = np.round(w_scaled * 2) / 2.0
    
    if valid:
        if _verify_decomposition(new_U, new_V, new_W, n, m, p, tol=1e-6):
            return new_U, new_V, new_W
    return None

def _structured_random_projection_elimination(U, V, W, n, m, p, max_time=90, n_random=50):
    """
    Uses structured random projection with iterative null-space elimination.
    """
    start_time = time.time()
    deadline = start_time + max_time
    rank = U.shape[1]
    target_flat = np.ones((n, m, p)).flatten()
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    term_vecs = _compute_term_vectors(U, V, W, n, m, p)
    
    random.seed(42)
    
    for attempt in range(n_random):
        if time.time() > deadline - 5:
            break
        
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
        
        perturbed_term_vecs = _compute_term_vectors(perturbed_U, perturbed_V, perturbed_W, n, m, p)
        
        try:
            _, s, vh = np.linalg.svd(perturbed_term_vecs, full_matrices=False)
        except Exception:
            continue
        
        null_threshold = 1e-8
        null_indices = np.where(s < null_threshold)[0]
        
        if len(null_indices) == 0:
            continue
        
        for null_idx in null_indices:
            if time.time() > deadline - 5:
                break
            
            null_vec = vh[null_idx, :]
            max_coeff_idx = np.argmax(np.abs(null_vec))
            
            remaining = [x for x in range(rank) if x != max_coeff_idx]
            M_remaining = term_vecs[:, remaining]
            
            try:
                c, residuals, rank_m, sv = np.linalg.lstsq(M_remaining, target_flat, rcond=None)
            except Exception:
                continue
            
            if not residuals or len(residuals) == 0 or residuals[0] > 1e-8:
                continue
            
            valid = True
            new_U = np.zeros((n*m, len(remaining)))
            new_V = np.zeros((m*p, len(remaining)))
            new_W = np.zeros((p*n, len(remaining)))
            
            for idx, r_idx in enumerate(remaining):
                u_scaled = U[:, r_idx] * c[idx]
                if not _is_half_integer_vector(u_scaled, tol=1e-5):
                    valid = False
                    break
                new_U[:, idx] = np.round(u_scaled * 2) / 2.0
                
                v_scaled = V[:, r_idx] * c[idx]
                if not _is_half_integer_vector(v_scaled, tol=1e-5):
                    valid = False
                    break
                new_V[:, idx] = np.round(v_scaled * 2) / 2.0
                
                w_scaled = W[:, r_idx] * c[idx]
                if not _is_half_integer_vector(w_scaled, tol=1e-5):
                    valid = False
                    break
                new_W[:, idx] = np.round(w_scaled * 2) / 2.0
            
            if valid and _verify_decomposition(new_U, new_V, new_W, n, m, p, tol=1e-6):
                if len(remaining) < best_rank:
                    best_U, best_V, best_W = new_U, new_V, new_W
                    best_rank = len(remaining)
                    U, V, W = new_U, new_V, new_W
                    rank = len(remaining)
                    term_vecs = _compute_term_vectors(U, V, W, n, m, p)
                    break
        
        if rank < best_rank:
            continue
    
    if best_rank < rank:
        return best_U, best_V, best_W
    return None

def _aggressive_reduction(U, V, W, n, m, p, max_time=120):
    """
    Aggressive reduction: try removing single terms, then pairs, then use SVD-based
    null space to find more complex dependencies.
    """
    start_time = time.time()
    deadline = start_time + max_time
    rank = U.shape[1]
    
    # First pass: try removing each single term
    for r in range(rank):
        if time.time() > deadline - 10:
            break
        result = _try_remove_single_term(U, V, W, n, m, p, r)
        if result is not None:
            U, V, W = result
            rank = U.shape[1]
    
    # Second pass: try removing pairs of terms
    for r1 in range(rank):
        if time.time() > deadline - 10:
            break
        for r2 in range(r1 + 1, rank):
            if time.time() > deadline - 10:
                break
            result = _try_remove_two_terms(U, V, W, n, m, p, r1, r2)
            if result is not None:
                U, V, W = result
                rank = U.shape[1]
                break  # Restart pair search with new rank
    
    # Third pass: SVD-based null space elimination
    if time.time() < deadline - 20 and rank > 60:
        reduced = _structured_random_projection_elimination(U, V, W, n, m, p, max_time=min(60, deadline - time.time() - 5))
        if reduced is not None:
            U, V, W = reduced
            rank = U.shape[1]
    
    return U, V, W

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Approach 1: Strassen-based rank-65 construction
    U, V, W = _construct_rank_65(n, m, p)
    if not _verify_decomposition(U, V, W, n, m, p):
        U, V, W = _build_schoolbook()
        return {"U": [[float(x) for x in row] for row in U],
                "V": [[float(x) for x in row] for row in V],
                "W": [[float(x) for x in row] for row in W]}
    
    current_rank = U.shape[1]
    
    # Approach 2: Aggressive reduction
    if time.time() < deadline - 10:
        try:
            U, V, W = _aggressive_reduction(U, V, W, n, m, p, max_time=min(100, deadline - time.time() - 10))
            current_rank = U.shape[1]
        except Exception:
            pass
    
    # Convert to lists of floats
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
