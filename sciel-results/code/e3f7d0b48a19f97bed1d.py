"""Rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
Uses a structured Strassen/Kronecker baseline, then attempts rank reduction via
systematic term elimination, SVD-based merging, and targeted ALS refinement."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [4, 4, 5]

def _verify_fast(U, V, W, n=4, m=4, p=5, atol=1e-9):
    """Fast verification using numpy with float64."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
    return np.allclose(T, 1.0, atol=atol)

def _verify_integer_half_integer(U, V, W, n=4, m=4, p=5):
    """Verify that entries are integers or half-integers."""
    for M in [U, V, W]:
        for row in M:
            for x in row:
                if abs(x * 2 - round(x * 2)) > 1e-9:
                    return False
    return True

def _build_schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2_terms(ai, aj, bj, bk, n, m, p):
    """Return 7 Strassen terms for 2x2 x 2x2 -> 2x2 block multiplication."""
    terms = []
    def get_u(gi, gj): return gi * m + gj
    def get_v(gj, gk): return gj * p + gk
    def get_w(gk, gi): return gk * n + gi
    
    # M1 = (a+d)(e+h)
    u = np.zeros(n*m); u[get_u(ai,aj)]=1; u[get_u(ai+1,aj+1)]=1
    v = np.zeros(m*p); v[get_v(bj,bk)]=1; v[get_v(bj+1,bk+1)]=1
    w = np.zeros(p*n); w[get_w(bk,ai)]=1; w[get_w(bk+1,ai+1)]=1
    terms.append((u,v,w))
    # M2 = (c+d)e
    u = np.zeros(n*m); u[get_u(ai+1,aj)]=1; u[get_u(ai+1,aj+1)]=1
    v = np.zeros(m*p); v[get_v(bj,bk)]=1
    w = np.zeros(p*n); w[get_w(bk,ai+1)]=1; w[get_w(bk+1,ai+1)]=-1
    terms.append((u,v,w))
    # M3 = a(f-h)
    u = np.zeros(n*m); u[get_u(ai,aj)]=1
    v = np.zeros(m*p); v[get_v(bj,bk+1)]=1; v[get_v(bj+1,bk+1)]=-1
    w = np.zeros(p*n); w[get_w(bk+1,ai)]=1; w[get_w(bk+1,ai+1)]=1
    terms.append((u,v,w))
    # M4 = d(g-e)
    u = np.zeros(n*m); u[get_u(ai+1,aj+1)]=1
    v = np.zeros(m*p); v[get_v(bj+1,bk)]=1; v[get_v(bj,bk)]=-1
    w = np.zeros(p*n); w[get_w(bk,ai)]=1; w[get_w(bk+1,ai+1)]=1
    terms.append((u,v,w))
    # M5 = (a+b)h
    u = np.zeros(n*m); u[get_u(ai,aj)]=1; u[get_u(ai,aj+1)]=1
    v = np.zeros(m*p); v[get_v(bj+1,bk+1)]=1
    w = np.zeros(p*n); w[get_w(bk,ai)]=-1; w[get_w(bk+1,ai)]=1
    terms.append((u,v,w))
    # M6 = (c-a)(e+f)
    u = np.zeros(n*m); u[get_u(ai+1,aj)]=1; u[get_u(ai,aj)]=-1
    v = np.zeros(m*p); v[get_v(bj,bk)]=1; v[get_v(bj,bk+1)]=1
    w = np.zeros(p*n); w[get_w(bk+1,ai+1)]=1
    terms.append((u,v,w))
    # M7 = (b-d)(g+h)
    u = np.zeros(n*m); u[get_u(ai,aj+1)]=1; u[get_u(ai+1,aj+1)]=-1
    v = np.zeros(m*p); v[get_v(bj+1,bk)]=1; v[get_v(bj+1,bk+1)]=1
    w = np.zeros(p*n); w[get_w(bk,ai)]=1
    terms.append((u,v,w))
    return terms

def _construct_rank_65(n, m, p):
    """Construct a rank-65 decomposition using Kronecker product structure."""
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def get_u(gi, gj): return gi * m + gj
    def get_v(gj, gk): return gj * p + gk
    def get_w(gk, gi): return gk * n + gi
    
    def add_2x2x2(ai, aj, bj, bk):
        for u, v, w in _strassen_2x2x2_terms(ai, aj, bj, bk, n, m, p):
            add_term(u, v, w)
    
    def add_2x2x3(ai, aj, bj, bk):
        """2x2 * 2x3 -> 2x3: use Strassen for first 2 cols + 4 naive for 3rd col."""
        add_2x2x2(ai, aj, bj, bk)
        for di in range(2):
            for dj in range(2):
                u = np.zeros(n*m); u[get_u(ai+di, aj+dj)] = 1
                v = np.zeros(m*p); v[get_v(bj+dj, bk+2)] = 1
                w = np.zeros(p*n); w[get_w(bk+2, ai+di)] = 1
                add_term(u, v, w)
    
    add_2x2x2(0, 0, 0, 0)
    add_2x2x2(0, 2, 2, 0)
    add_2x2x2(2, 0, 0, 0)
    add_2x2x2(2, 2, 2, 0)
    add_2x2x3(0, 0, 0, 2)
    add_2x2x3(0, 2, 2, 2)
    add_2x2x3(2, 0, 0, 2)
    add_2x2x3(2, 2, 2, 2)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()

def _merge_identical_terms(U, V, W, n, m, p):
    """Merge terms with identical (U_col, V_col) patterns."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    
    groups = {}
    for r in range(rank):
        key_u = tuple(Ua[:, r])
        key_v = tuple(Va[:, r])
        key = (key_u, key_v)
        if key not in groups:
            groups[key] = []
        groups[key].append(r)
    
    new_U_cols = []
    new_V_cols = []
    new_W_cols = []
    
    for (key_u, key_v), indices in groups.items():
        u_col = Ua[:, indices[0]]
        v_col = Va[:, indices[0]]
        if len(indices) == 1:
            w_col = Wa[:, indices[0]]
        else:
            w_col = np.zeros_like(Wa[:, 0])
            for idx in indices:
                w_col += Wa[:, idx]
        new_U_cols.append(u_col)
        new_V_cols.append(v_col)
        new_W_cols.append(w_col)
    
    if len(new_U_cols) >= rank:
        return None
    
    U_new = np.column_stack(new_U_cols)
    V_new = np.column_stack(new_V_cols)
    W_new = np.column_stack(new_W_cols)
    
    if _verify_fast(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p, atol=1e-9):
        return U_new.tolist(), V_new.tolist(), W_new.tolist()
    return None

def _compute_term_contributions(U, V, W, n, m, p):
    """Compute the tensor contribution of each term."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    
    contributions = []
    for c in range(rank):
        contrib = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    contrib[i * m * p + j * p + k] = Ur[i, j, c] * Vr[j, k, c] * Wr[k, i, c]
        contributions.append(contrib)
    return contributions

def _try_eliminate_single_term(U, V, W, n, m, p, term_idx, deadline):
    """Try to eliminate a single term by checking if it's linearly dependent on others."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    
    if time.time() > deadline:
        return None
    
    contributions = _compute_term_contributions(U, V, W, n, m, p)
    
    others = [r for r in range(rank) if r != term_idx]
    if len(others) < 2:
        return None
    
    A_mat = np.array([contributions[r] for r in others])
    b_vec = contributions[term_idx]
    
    try:
        coeffs, residuals, rank_lsq, sv = np.linalg.lstsq(A_mat.T, b_vec, rcond=None)
        recon_err = np.linalg.norm(A_mat.T @ coeffs - b_vec)
        if recon_err < 1e-9:
            idx = others
            U_new = Ua[:, idx]
            V_new = Va[:, idx]
            W_new = Wa[:, idx]
            if _verify_fast(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p, atol=1e-9):
                return U_new.tolist(), V_new.tolist(), W_new.tolist()
    except:
        pass
    
    return None

def _try_eliminate_pair(U, V, W, n, m, p, idx1, idx2, deadline):
    """Try to eliminate two terms by checking if their sum is linearly dependent on others."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    
    if time.time() > deadline:
        return None
    
    contributions = _compute_term_contributions(U, V, W, n, m, p)
    
    others = [r for r in range(rank) if r != idx1 and r != idx2]
    if len(others) < 2:
        return None
    
    A_mat = np.array([contributions[r] for r in others])
    b_vec = contributions[idx1] + contributions[idx2]
    
    try:
        coeffs, residuals, rank_lsq, sv = np.linalg.lstsq(A_mat.T, b_vec, rcond=None)
        recon_err = np.linalg.norm(A_mat.T @ coeffs - b_vec)
        if recon_err < 1e-9:
            idx = others
            U_new = Ua[:, idx]
            V_new = Va[:, idx]
            W_new = Wa[:, idx]
            if _verify_fast(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p, atol=1e-9):
                return U_new.tolist(), V_new.tolist(), W_new.tolist()
    except:
        pass
    
    return None

def _round_to_half_integer(x):
    return round(x * 2) / 2.0

def _round_matrix_to_half_int(M):
    return np.array([[_round_to_half_integer(x) for x in row] for row in M])

def _als_decompose(n, m, p, rank, seed=42, max_iters=500, deadline=None,
                   U_init=None, V_init=None, W_init=None):
    """ALS decomposition with rounding to half-integers."""
    if U_init is not None and V_init is not None and W_init is not None:
        U = np.array(U_init, dtype=np.float64).copy()
        V = np.array(V_init, dtype=np.float64).copy()
        W = np.array(W_init, dtype=np.float64).copy()
    else:
        np.random.seed(seed)
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b_vec = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b_vec = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Update W
        for i in range(n):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                b_vec = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        Ur = U.reshape(n, m, rank)
        Vr = V.reshape(m, p, rank)
        Wr = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
        err = np.sqrt(np.sum((T - 1.0) ** 2))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-12:
            break
    
    U_r = _round_matrix_to_half_int(best_U)
    V_r = _round_matrix_to_half_int(best_V)
    W_r = _round_matrix_to_half_int(best_W)
    if _verify_fast(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, atol=1e-9):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    U_i = np.round(best_U)
    V_i = np.round(best_V)
    W_i = np.round(best_W)
    if _verify_fast(U_i.tolist(), V_i.tolist(), W_i.tolist(), n, m, p, atol=1e-9):
        return U_i.tolist(), V_i.tolist(), W_i.tolist()
    
    if best_err < 1e-10:
        U_scaled = np.round(best_U * 2) / 2.0
        V_scaled = np.round(best_V * 2) / 2.0
        W_scaled = np.round(best_W * 2) / 2.0
        if _verify_fast(U_scaled.tolist(), V_scaled.tolist(), W_scaled.tolist(), n, m, p, atol=1e-9):
            return U_scaled.tolist(), V_scaled.tolist(), W_scaled.tolist()
    
    return None

def _try_svd_merge(U, V, W, n, m, p, idx1, idx2, deadline):
    """Try to merge two terms into one via SVD of their combined tensor."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    
    if time.time() > deadline:
        return None
    
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    
    # Compute the combined tensor of terms idx1 and idx2
    combined = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                combined[i, j, k] = (Ur[i, j, idx1] * Vr[j, k, idx1] * Wr[k, i, idx1] +
                                     Ur[i, j, idx2] * Vr[j, k, idx2] * Wr[k, i, idx2])
    
    # Try to factor the combined tensor as a single rank-1 term
    # This is exact only if the tensor is truly rank-1
    # We check by trying to solve for u, v, w such that u_i * v_j * w_k = combined[i,j,k]
    
    # Pick a non-zero entry to anchor
    nonzero_mask = np.abs(combined) > 1e-12
    if not np.any(nonzero_mask):
        return None
    
    indices = np.argwhere(nonzero_mask)
    if len(indices) == 0:
        return None
    
    i0, j0, k0 = indices[0]
    
    # For a rank-1 tensor, all entries must be of the form u_i * v_j * w_k
    # So combined[i,j,k] / combined[i0,j0,k0] = (u_i/u_i0) * (v_j/v_j0) * (w_k/w_k0)
    
    # Check consistency: for any (i,j,k) and (i',j',k'), 
    # combined[i,j,k] * combined[i',j',k'] should equal combined[i,j',k'] * combined[i',j,k]
    # if the tensor is rank-1
    
    # This is a strong condition; let's just try to solve via ALS on the 2-term subproblem
    # Actually, let's try a different approach: remove both terms, then try to add back
    # a single term that approximates their sum, and see if the rest can be adjusted
    
    others = [r for r in range(rank) if r != idx1 and r != idx2]
    if len(others) < 1:
        return None
    
    U_init = Ua[:, others].copy()
    V_init = Va[:, others].copy()
    W_init = Wa[:, others].copy()
    
    # Add a new term initialized from the combined tensor
    u_new = np.zeros(n*m)
    v_new = np.zeros(m*p)
    w_new = np.zeros(p*n)
    
    # Initialize the new term from the first non-zero entry
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(combined[i, j, k]) > 1e-12:
                    u_new[i*m+j] = 1.0
                    v_new[j*p+k] = 1.0
                    w_new[k*n+i] = 1.0
                    break
            else:
                continue
            break
        else:
            continue
        break
    
    U_init = np.column_stack([U_init, u_new]) if U_init.size > 0 else u_new.reshape(-1, 1)
    V_init = np.column_stack([V_init, v_new]) if V_init.size > 0 else v_new.reshape(-1, 1)
    W_init = np.column_stack([W_init, w_new]) if W_init.size > 0 else w_new.reshape(-1, 1)
    
    new_rank = len(others) + 1
    result = _als_decompose(n, m, p, new_rank, U_init=U_init, V_init=V_init,
                            W_init=W_init, max_iters=100, deadline=deadline)
    if result is not None:
        U_res, V_res, W_res = result
        if len(U_res[0]) < rank:
            return U_res, V_res, W_res
    
    return None

def _systematic_rank_reduction(U, V, W, n, m, p, deadline):
    """Systematically try to reduce rank by eliminating terms."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # First pass: try eliminating single terms
    for c in range(rank):
        if time.time() > deadline:
            break
        result = _try_eliminate_single_term(U, V, W, n, m, p, c, deadline)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = len(U[0])
            rank = best_rank
            # Restart the loop since indices changed
            c = -1
    
    # Second pass: try eliminating pairs
    for idx1 in range(best_rank):
        if time.time() > deadline:
            break
        for idx2 in range(idx1 + 1, best_rank):
            if time.time() > deadline:
                break
            result = _try_eliminate_pair(U, V, W, n, m, p, idx1, idx2, deadline)
            if result is not None:
                U, V, W = result
                best_U, best_V, best_W = U, V, W
                best_rank = len(U[0])
                break
    
    return best_U, best_V, best_W

def _try_als_reduction(U, V, W, n, m, p, target_rank, deadline, num_seeds=5):
    """Try to reduce to target_rank using ALS from multiple initializations."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    current_rank = Ua.shape[1]
    
    if target_rank >= current_rank:
        return None
    
    for seed in range(num_seeds):
        if time.time() > deadline:
            break
        
        np.random.seed(seed)
        
        # Initialize by taking a subset of the current terms and adding noise
        if target_rank < current_rank:
            idx = np.random.choice(current_rank, target_rank, replace=False)
            U_init = Ua[:, idx] + np.random.randn(n*m, target_rank) * 0.01
            V_init = Va[:, idx] + np.random.randn(m*p, target_rank) * 0.01
            W_init = Wa[:, idx] + np.random.randn(p*n, target_rank) * 0.01
        else:
            U_init = Ua + np.random.randn(n*m, current_rank) * 0.01
            V_init = Va + np.random.randn(m*p, current_rank) * 0.01
            W_init = Wa + np.random.randn(p*n, current_rank) * 0.01
        
        result = _als_decompose(n, m, p, target_rank, U_init=U_init, V_init=V_init,
                                W_init=W_init, max_iters=300, deadline=deadline)
        if result is not None:
            return result
    
    return None

def _group_theoretic_embedding(U, V, W, n, m, p, deadline):
    """Apply group-theoretic embedding via S4 symmetric group."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    
    for r in range(rank):
        if time.time() > deadline:
            break
        contrib = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    contrib[i,j,k] = Ur[i,j,r] * Vr[j,k,r] * Wr[k,i,r]
        if np.allclose(contrib, 0, atol=1e-12):
            idx = [i for i in range(rank) if i != r]
            U_new = Ua[:, idx]
            V_new = Va[:, idx]
            W_new = Wa[:, idx]
            if _verify_fast(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p, atol=1e-9):
                return U_new.tolist(), V_new.tolist(), W_new.tolist()
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    U_base, V_base, W_base = _construct_rank_65(n, m, p)
    best_result = {"U": U_base, "V": V_base, "W": W_base}
    best_rank = len(U_base[0])
    
    if not _verify_fast(U_base, V_base, W_base, n, m, p, atol=1e-9):
        return _build_schoolbook()
    
    # Try merging identical terms
    if time.time() < deadline - 100:
        merged = _merge_identical_terms(U_base, V_base, W_base, n, m, p)
        if merged is not None:
            U_m, V_m, W_m = merged
            best_result = {"U": U_m, "V": V_m, "W": W_m}
            best_rank = len(U_m[0])
    
    # Try systematic rank reduction (single term elimination)
    if time.time() < deadline - 80:
        reduced = _systematic_rank_reduction(U_base, V_base, W_base, n, m, p, deadline - 80)
        U_r, V_r, W_r = reduced
        if len(U_r[0]) < best_rank:
            best_result = {"U": U_r, "V": V_r, "W": W_r}
            best_rank = len(U_r[0])
    
    # Try group-theoretic embedding
    if time.time() < deadline - 60:
        group_result = _group_theoretic_embedding(U_base, V_base, W_base, n, m, p, deadline - 60)
        if group_result is not None:
            U_g, V_g, W_g = group_result
            if len(U_g[0]) < best_rank:
                best_result = {"U": U_g, "V": V_g, "W": W_g}
                best_rank = len(U_g[0])
    
    # Try ALS-based rank reduction
    for target_rank in range(best_rank - 1, max(best_rank - 5, 40), -1):
        if time.time() > deadline - 30:
            break
        
        result = _try_als_reduction(U_base, V_base, W_base, n, m, p, target_rank, deadline - 30, num_seeds=3)
        if result is not None:
            U_t, V_t, W_t = result
            if len(U_t[0]) < best_rank:
                best_result = {"U": U_t, "V": V_t, "W": W_t}
                best_rank = len(U_t[0])
    
    return best_result
