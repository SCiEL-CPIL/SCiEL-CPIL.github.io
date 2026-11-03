"""Matrix multiplication 4,4,5 tensor decomposition with combinatorial subspace elimination."""
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

def _get_tensor_vectors(U, V, W, n, m, p):
    """
    Computes the vector representation of each rank-1 term.
    Returns a matrix M of shape (n*m*p, rank) where M[:, r] is the flattened tensor for term r.
    """
    rank = U.shape[1]
    num_entries = n * m * p
    M = np.zeros((num_entries, rank))
    
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
                        M[flat_idx, r] = val
    return M

def _try_rank_reduction(U, V, W, n, m, p, start_rank, min_rank, deadline):
    """
    Attempts to reduce rank using structured subspace analysis and systematic column elimination.
    """
    target = np.ones(n * m * p)
    
    # Pre-compute tensor vectors for the current U, V, W
    M_full = _get_tensor_vectors(U, V, W, n, m, p)
    current_rank = U.shape[1]
    
    try:
        # Strategy 1: QR with column pivoting to identify numerical rank
        Q, Rmat, P = np.linalg.qr(M_full, mode='complete', pivoting=True)
        diag = np.abs(np.diag(Rmat))
        tol = max(M_full.shape) * np.finfo(float).eps * diag[0] if diag[0] > 0 else 1e-10
        num_rank = np.sum(diag > tol)
        
        best_subset = list(P[:num_rank])
        best_res = float('inf')
        
        # Compute residual with initial subset
        M_sub = M_full[:, best_subset]
        c, residuals, rank_m, sv = np.linalg.lstsq(M_sub, target, rcond=None)
        best_res = residuals[0] if len(residuals) > 0 else np.linalg.norm(M_sub @ c - target)
        
        # Strategy 2: Systematic removal - try removing one column at a time
        # Keep the best configuration at each step
        improved = True
        while improved and time.time() < deadline:
            improved = False
            M_sub = M_full[:, best_subset]
            c, residuals, rank_m, sv = np.linalg.lstsq(M_sub, target, rcond=None)
            current_res = residuals[0] if len(residuals) > 0 else np.linalg.norm(M_sub @ c - target)
            
            if current_res < 1e-8:
                break
            
            # Try removing each column and see if we can still represent target
            for i in range(len(best_subset)):
                if time.time() > deadline:
                    break
                candidate = best_subset[:i] + best_subset[i+1:]
                if len(candidate) < min_rank:
                    continue
                M_test = M_full[:, candidate]
                c_test, res_test, _, _ = np.linalg.lstsq(M_test, target, rcond=None)
                test_res = res_test[0] if len(res_test) > 0 else np.linalg.norm(M_test @ c_test - target)
                
                if test_res < 1e-8:
                    best_subset = candidate
                    best_res = test_res
                    improved = True
                    break
                elif test_res < best_res:
                    # Keep track of best candidate even if not exact
                    pass
        
        # Check if we found a valid subset
        if len(best_subset) < current_rank:
            M_sub = M_full[:, best_subset]
            c, residuals, rank_m, sv = np.linalg.lstsq(M_sub, target, rcond=None)
            if len(residuals) > 0 and residuals[0] < 1e-8:
                R_new = len(best_subset)
                U_new = np.zeros((n*m, R_new))
                V_new = np.zeros((m*p, R_new))
                W_new = np.zeros((p*n, R_new))
                
                for idx, orig_idx in enumerate(best_subset):
                    U_new[:, idx] = U[:, orig_idx] * c[idx]
                    V_new[:, idx] = V[:, orig_idx] * c[idx]
                    W_new[:, idx] = W[:, orig_idx] * c[idx]
                
                if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-8):
                    if _verify_integer_half_integer(U_new, V_new, W_new, tol=1e-8):
                        return U_new, V_new, W_new
                    else:
                        U_rounded = np.round(U_new * 2) / 2.0
                        V_rounded = np.round(V_new * 2) / 2.0
                        W_rounded = np.round(W_new * 2) / 2.0
                        if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
                            return U_rounded, V_rounded, W_rounded
    except Exception:
        pass
    
    # Strategy 3: Structured subspace analysis
    # The 4x4x4 part has rank 49 (from Kronecker Strassen), 4x4x1 part has rank 16
    # Try to find linear dependencies within each block
    
    try:
        # Separate the terms into 4x4x4 block (first 49) and 4x4x1 block (last 16)
        if current_rank >= 65:
            block1_end = 49
            block2_start = 49
            block2_end = current_rank
            
            # Check if we can reduce within block1 (4x4x4 part)
            M_block1 = M_full[:, :block1_end]
            Q1, R1, P1 = np.linalg.qr(M_block1, mode='complete', pivoting=True)
            diag1 = np.abs(np.diag(R1))
            tol1 = max(M_block1.shape) * np.finfo(float).eps * diag1[0] if diag1[0] > 0 else 1e-10
            num_rank1 = np.sum(diag1 > tol1)
            
            # Check if we can reduce within block2 (4x4x1 part)
            M_block2 = M_full[:, block2_start:block2_end]
            Q2, R2, P2 = np.linalg.qr(M_block2, mode='complete', pivoting=True)
            diag2 = np.abs(np.diag(R2))
            tol2 = max(M_block2.shape) * np.finfo(float).eps * diag2[0] if diag2[0] > 0 else 1e-10
            num_rank2 = np.sum(diag2 > tol2)
            
            # Try to construct a reduced decomposition
            reduced_rank1 = max(num_rank1, 45)  # Keep at least 45 from block1
            reduced_rank2 = max(num_rank2, 12)  # Keep at least 12 from block2
            
            if reduced_rank1 + reduced_rank2 < current_rank:
                # Select columns from each block
                cols1 = list(P1[:reduced_rank1])
                cols2 = [block2_start + i for i in P2[:reduced_rank2]]
                combined_cols = cols1 + cols2
                combined_cols = sorted(combined_cols)
                
                M_sub = M_full[:, combined_cols]
                c, residuals, rank_m, sv = np.linalg.lstsq(M_sub, target, rcond=None)
                
                if len(residuals) > 0 and residuals[0] < 1e-8:
                    R_new = len(combined_cols)
                    U_new = np.zeros((n*m, R_new))
                    V_new = np.zeros((m*p, R_new))
                    W_new = np.zeros((p*n, R_new))
                    
                    for idx, orig_idx in enumerate(combined_cols):
                        U_new[:, idx] = U[:, orig_idx] * c[idx]
                        V_new[:, idx] = V[:, orig_idx] * c[idx]
                        W_new[:, idx] = W[:, orig_idx] * c[idx]
                    
                    if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-8):
                        if _verify_integer_half_integer(U_new, V_new, W_new, tol=1e-8):
                            return U_new, V_new, W_new
                        else:
                            U_rounded = np.round(U_new * 2) / 2.0
                            V_rounded = np.round(V_new * 2) / 2.0
                            W_rounded = np.round(W_new * 2) / 2.0
                            if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
                                return U_rounded, V_rounded, W_rounded
    except Exception:
        pass
    
    # Strategy 4: Random subset search with more attempts
    for R in range(min(current_rank - 1, 64), min_rank - 1, -1):
        if time.time() > deadline:
            break
        
        # Try more random subsets with weighted sampling based on column importance
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            # Weight columns by their norm (more important columns get higher weight)
            col_norms = np.linalg.norm(M_full, axis=0)
            col_norms = col_norms + 1e-10  # Avoid zero weights
            weights = col_norms / col_norms.sum()
            
            indices = np.random.choice(current_rank, R, replace=False, p=weights)
            M_sub = M_full[:, indices]
            c, residuals, rank_m, sv = np.linalg.lstsq(M_sub, target, rcond=None)
            
            if len(residuals) > 0 and residuals[0] < 1e-8:
                U_new = np.zeros((n*m, R))
                V_new = np.zeros((m*p, R))
                W_new = np.zeros((p*n, R))
                
                for idx, orig_idx in enumerate(indices):
                    U_new[:, idx] = U[:, orig_idx] * c[idx]
                    V_new[:, idx] = V[:, orig_idx] * c[idx]
                    W_new[:, idx] = W[:, orig_idx] * c[idx]
                
                if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-8):
                    if _verify_integer_half_integer(U_new, V_new, W_new, tol=1e-8):
                        return U_new, V_new, W_new
                    else:
                        U_rounded = np.round(U_new * 2) / 2.0
                        V_rounded = np.round(V_new * 2) / 2.0
                        W_rounded = np.round(W_new * 2) / 2.0
                        if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
                            return U_rounded, V_rounded, W_rounded
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Approach 1: Strassen-based rank-65 construction
    try:
        U, V, W = _construct_rank_65(n, m, p)
        rank = U.shape[1]
        
        if _verify_decomposition(U, V, W, n, m, p):
            best_result = (U, V, W)
            best_rank = rank
    except Exception:
        pass
    
    # Approach 2: Rank reduction
    if best_result is not None:
        U, V, W = best_result
        # Try to reduce from 65 down to 49
        reduced = _try_rank_reduction(U, V, W, n, m, p, best_rank, 49, deadline)
        if reduced is not None:
            best_result = reduced
            best_rank = reduced[0].shape[1]
    
    if best_result is not None:
        U, V, W = best_result
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
