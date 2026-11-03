"""Matrix multiplication 4,4,5 tensor decomposition with Strassen baseline and ALS refinement."""
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
    # Kronecker product of two 2x2x2 Strassen decompositions
    for (u1_sparse, v1_sparse, w1_sparse) in strassen_terms:
        for (u2_sparse, v2_sparse, w2_sparse) in strassen_terms:
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            
            # U: i = i1*2 + i2, j = j1*2 + j2
            for (idx1, val1) in u1_sparse:
                i1, j1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in u2_sparse:
                    i2, j2 = idx2 // 2, idx2 % 2
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    u[get_u_idx(i, j)] = val1 * val2
            
            # V: j = j1*2 + j2, k = k1*2 + k2
            for (idx1, val1) in v1_sparse:
                j1, k1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in v2_sparse:
                    j2, k2 = idx2 // 2, idx2 % 2
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    v[get_v_idx(j, k)] = val1 * val2
            
            # W: k = k1*2 + k2, i = i1*2 + i2
            for (idx1, val1) in w1_sparse:
                k1, i1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in w2_sparse:
                    k2, i2 = idx2 // 2, idx2 % 2
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    w[get_w_idx(k, i)] = val1 * val2
            
            add_term(u, v, w)
            
    # 2. 4x4x1 part (k=4)
    # Naive: A[i,j] * B[j,4] -> C[4,i]
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
    
    rank = U_np.shape[1]
    # Pre-compute for each (i,j,k) the triple product
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

def _compute_tensor_residual(U, V, W, n, m, p):
    """Compute the residual tensor T - T_approx where T_approx = sum_r U_r ⊙ V_r ⊙ W_r."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    rank = U_np.shape[1]
    
    T_approx = np.zeros((n, m, p))
    for r in range(rank):
        u = U_np[:, r].reshape(n, m)
        v = V_np[:, r].reshape(m, p)
        w = W_np[:, r].reshape(p, n)
        # T_approx[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        for i in range(n):
            for j in range(m):
                uj = u[i, j]
                if uj == 0:
                    continue
                for k in range(p):
                    T_approx[i, j, k] += uj * v[j, k] * w[k, i]
    
    return np.ones((n, m, p)) - T_approx

def _verify_integer_half_integer(U, V, W, tol=1e-9):
    """Check that all entries are integer or half-integer (multiple of 0.5)."""
    for name, M in [("U", U), ("V", V), ("W", W)]:
        for row in M:
            for val in row:
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

def _als_refine(U_init, V_init, W_init, n, m, p, target_rank, max_iters=200, tol=1e-8, deadline=None):
    """
    Attempts to refine the decomposition to a lower rank using gradient-based optimization.
    Uses a vectorized approach to compute gradients efficiently.
    """
    if deadline is None:
        deadline = time.time() + 30
    
    U = np.array(U_init, dtype=np.float64)
    V = np.array(V_init, dtype=np.float64)
    W = np.array(W_init, dtype=np.float64)
    
    current_rank = U.shape[1]
    if current_rank <= target_rank:
        U = U[:, :target_rank].copy()
        V = V[:, :target_rank].copy()
        W = W[:, :target_rank].copy()
    else:
        # Random projection to break symmetry
        np.random.seed(42)
        P_U = np.random.randn(current_rank, target_rank)
        P_V = np.random.randn(current_rank, target_rank)
        P_W = np.random.randn(current_rank, target_rank)
        U = (U @ P_U).copy()
        V = (V @ P_V).copy()
        W = (W @ P_W).copy()
    
    # Pre-compute index arrays for vectorized computation
    # For each (i,j,k), we need indices into U, V, W
    i_arr, j_arr, k_arr = np.meshgrid(range(n), range(m), range(p), indexing='ij')
    i_flat = i_arr.flatten()
    j_flat = j_arr.flatten()
    k_flat = k_arr.flatten()
    
    u_idx = i_flat * m + j_flat  # shape (n*m*p,)
    v_idx = j_flat * p + k_flat  # shape (n*m*p,)
    w_idx = k_flat * n + i_flat  # shape (n*m*p,)
    
    # Pre-compute V and W rows for each (i,j,k)
    V_rows = V[v_idx, :]  # shape (n*m*p, target_rank)
    W_rows = W[w_idx, :]  # shape (n*m*p, target_rank)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.01
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute current approximation
        U_rows = U[u_idx, :]  # shape (n*m*p, target_rank)
        T_approx = np.sum(U_rows * V_rows * W_rows, axis=1)  # shape (n*m*p,)
        residual = 1.0 - T_approx  # shape (n*m*p,)
        err = np.sum(residual**2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
        
        # Compute gradients
        # dL/dU[u_idx[r], r] = -2 * residual[r] * V_rows[r, r'] * W_rows[r, r'] for each r'
        # More precisely: dL/dU[i*m+j, r] = -2 * sum_{(i',j',k'): i'*m+j' = i*m+j} residual[i',j',k'] * V[j',k',r] * W[k',i',r]
        
        # For efficiency, compute gradient for each matrix
        # grad_U[i*m+j, r] = -2 * sum over (j',k') where i*m+j fixed: residual[i,j',k'] * V[j',k',r] * W[k',i,r]
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Vectorized gradient computation
        # For U: group by u_idx
        for r in range(target_rank):
            # grad_U[:, r] = -2 * (residual * V_rows[:, r] * W_rows[:, r]) grouped by u_idx
            contrib = -2.0 * residual * V_rows[:, r] * W_rows[:, r]
            np.add.at(grad_U, u_idx, contrib)
        
        for r in range(target_rank):
            contrib = -2.0 * residual * U_rows[:, r] * W_rows[:, r]
            np.add.at(grad_V, v_idx, contrib)
        
        for r in range(target_rank):
            contrib = -2.0 * residual * U_rows[:, r] * V_rows[:, r]
            np.add.at(grad_W, w_idx, contrib)
        
        # Gradient step
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Decay learning rate
        lr *= 0.99
    
    # Check if best result is valid
    if best_err < 1e-6:
        if _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-6):
            return best_U, best_V, best_W
    
    return None

def _build_term_matrix(U, V, W, n, m, p, exclude_indices=None):
    """Build the coefficient matrix M where M[flat_idx, col] = U[r]V[r]W[r] for term r."""
    rank = U.shape[1]
    num_entries = n * m * p
    
    if exclude_indices is None:
        cols = list(range(rank))
    else:
        cols = [r for r in range(rank) if r not in exclude_indices]
    
    M = np.zeros((num_entries, len(cols)))
    
    i_arr, j_arr, k_arr = np.meshgrid(range(n), range(m), range(p), indexing='ij')
    i_flat = i_arr.flatten()
    j_flat = j_arr.flatten()
    k_flat = k_arr.flatten()
    
    u_idx = i_flat * m + j_flat
    v_idx = j_flat * p + k_flat
    w_idx = k_flat * n + i_flat
    
    for col_idx, r in enumerate(cols):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        M[:, col_idx] = u[u_idx] * v[v_idx] * w[w_idx]
    
    return M, cols

def _try_rank_reduction_by_removal(U, V, W, n, m, p, tol=1e-9, deadline=None):
    """
    Attempts to reduce rank by removing one or more terms and solving for coefficients.
    """
    if deadline is None:
        deadline = time.time() + 10
        
    rank = U.shape[1]
    num_entries = n * m * p
    target_flat = np.ones(num_entries)
    
    # Try removing single terms
    for r_remove in range(rank):
        if time.time() > deadline: break
        
        M, cols = _build_term_matrix(U, V, W, n, m, p, exclude_indices={r_remove})
        
        try:
            c, residuals, rank_m, sv = np.linalg.lstsq(M, target_flat, rcond=None)
            
            if residuals and residuals[0] < tol:
                U_new = np.zeros((n*m, rank-1))
                V_new = np.zeros((m*p, rank-1))
                W_new = np.zeros((p*n, rank-1))
                
                for idx, r in enumerate(cols):
                    U_new[:, idx] = U[:, r] * c[idx]
                    V_new[:, idx] = V[:, r]
                    W_new[:, idx] = W[:, r]
                
                if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-8):
                    return U_new, V_new, W_new
        except Exception:
            pass
    
    # Try removing pairs of terms
    for r1 in range(rank):
        for r2 in range(r1+1, rank):
            if time.time() > deadline: break
            if r2 - r1 > 1:  # Only try adjacent pairs for efficiency
                continue
            
            M, cols = _build_term_matrix(U, V, W, n, m, p, exclude_indices={r1, r2})
            
            try:
                c, residuals, rank_m, sv = np.linalg.lstsq(M, target_flat, rcond=None)
                
                if residuals and residuals[0] < tol:
                    U_new = np.zeros((n*m, rank-2))
                    V_new = np.zeros((m*p, rank-2))
                    W_new = np.zeros((p*n, rank-2))
                    
                    for idx, r in enumerate(cols):
                        U_new[:, idx] = U[:, r] * c[idx]
                        V_new[:, idx] = V[:, r]
                        W_new[:, idx] = W[:, r]
                    
                    if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-8):
                        return U_new, V_new, W_new
            except Exception:
                pass
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Approach 1: Strassen-based rank-65 construction
    try:
        U, V, W = _construct_rank_65(n, m, p)
        rank = U.shape[1]
        
        if _verify_decomposition(U, V, W, n, m, p):
            candidates = [(rank, U, V, W)]
        else:
            candidates = []
    except Exception:
        candidates = []
    
    # Pick the best candidate
    if candidates:
        candidates.sort(key=lambda x: x[0])
        best_rank, U, V, W = candidates[0]
        best_result = (U, V, W)
    
    # Approach 2: Try rank reduction by term removal
    if best_result is not None and time.time() < deadline - 20:
        try:
            U_r, V_r, W_r = best_result
            reduced = _try_rank_reduction_by_removal(U_r, V_r, W_r, n, m, p, deadline=deadline - 20)
            if reduced is not None:
                best_result = reduced
                best_rank = reduced[0].shape[1]
        except Exception:
            pass
    
    # Approach 3: Try ALS refinement to lower ranks
    if best_result is not None and time.time() < deadline - 10:
        U_r, V_r, W_r = best_result
        current_rank = U_r.shape[1]
        
        # Try reducing to current_rank - 1, current_rank - 2, etc.
        for target in range(current_rank - 1, max(current_rank - 5, 40), -1):
            if time.time() > deadline - 10:
                break
            try:
                refined = _als_refine(U_r, V_r, W_r, n, m, p, target, 
                                     max_iters=100, deadline=deadline - 10)
                if refined is not None:
                    U_new, V_new, W_new = refined
                    new_rank = U_new.shape[1]
                    if new_rank < best_rank:
                        best_result = (U_new, V_new, W_new)
                        best_rank = new_rank
                        U_r, V_r, W_r = U_new, V_new, W_new
            except Exception:
                pass
    
    if best_result is not None:
        U, V, W = best_result
        # Round to half-integers if close
        for M in [U, V, W]:
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    M[i, j] = round(M[i, j] * 2) / 2.0
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U, V, W = _build_schoolbook()
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
