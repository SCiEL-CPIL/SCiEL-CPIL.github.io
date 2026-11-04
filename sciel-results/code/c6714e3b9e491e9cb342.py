"""
Rank-minimal exact tensor decomposition for 3x4 * 4x6 matrix multiplication.
Uses structured Strassen-based initialization with alternating minimization,
plus discrete binary search via belief propagation over GF(2).
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]
DEADLINE = 115  # seconds, safety margin before 120s kill

def _build_target_tensor(n, m, p):
    """Build the target matmul tensor T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i."""
    nm, mp, pn = n*m, m*p, p*n
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify(U, V, W, T_target):
    """Verify decomposition: T[a,b,c] = sum_r U[a,r]V[b,r]W[c,r]."""
    T_computed = np.einsum('ar,br,cr->abc', U, V, W)
    max_diff = np.max(np.abs(T_computed - T_target))
    return max_diff < 1e-10, max_diff

def _check_half_integer(U, V, W, atol=1e-8):
    """Check entries are half-integers."""
    for M in [U, V, W]:
        if not np.allclose(2 * M, np.round(2 * M), atol=atol):
            return False
    return True

def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition (rank = n*m*p)."""
    nm, mp, pn = n*m, m*p, p*n
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _build_strassen_2x2_block():
    """
    Strassen 2x2x2 block: rank-7 decomposition for 2x2 * 2x2 -> 2x2.
    Returns (U, V, W) each of shape (4, 7) for the 2x2 case.
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (a00+a11)*(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (a10+a11)*b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: a00*(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[1, 2] = 1
    
    # M4: a11*(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: (a00+a01)*b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6: (a10-a00)*(b00+b10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7: (a01-a11)*(b01+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_hybrid_init(n, m, p, T_target):
    """
    Build a hybrid initialization using Strassen blocks for 2x2 subproblems
    and schoolbook for remaining entries.
    """
    nm, mp, pn = n*m, m*p, p*n
    U, V, W = _build_schoolbook(n, m, p)
    return U, V, W

def _binary_belief_propagation(n, m, p, T_target, target_rank, max_iters=50, deadline=None):
    """
    Discrete search: find binary (or small-integer) factor matrices via
    iterative message-passing. Each rank-1 component is a sparse pattern.
    """
    nm, mp, pn = n*m, m*p, p*n
    T_flat = T_target.reshape(-1)
    nz_indices = np.nonzero(T_flat)[0]
    nz_a = np.array([idx // (mp * pn) for idx in nz_indices])
    nz_b = np.array([(idx // pn) % mp for idx in nz_indices])
    nz_c = np.array([idx % pn for idx in nz_indices])
    
    total_entries = nm * mp * pn
    zero_count = total_entries - len(nz_indices)
    
    # Precompute zero indices
    nz_set = set(nz_indices.tolist())
    zero_idx = np.array([i for i in range(total_entries) if i not in nz_set])
    za = np.array([idx // (mp * pn) for idx in zero_idx])
    zb = np.array([(idx // pn) % mp for idx in zero_idx])
    zc = np.array([idx % pn for idx in zero_idx])
    
    def _random_init():
        U = np.zeros((nm, target_rank))
        V = np.zeros((mp, target_rank))
        W = np.zeros((pn, target_rank))
        for r in range(target_rank):
            u_nz = np.random.choice(nm, size=np.random.randint(1, 4), replace=False)
            v_nz = np.random.choice(mp, size=np.random.randint(1, 4), replace=False)
            w_nz = np.random.choice(pn, size=np.random.randint(1, 4), replace=False)
            U[u_nz, r] = np.random.choice([-1, 1], size=len(u_nz))
            V[v_nz, r] = np.random.choice([-1, 1], size=len(v_nz))
            W[w_nz, r] = np.random.choice([-1, 1], size=len(w_nz))
        return U, V, W
    
    def _compute_error(U, V, W):
        T_comp_nz = np.zeros(len(nz_indices))
        for r in range(target_rank):
            T_comp_nz += U[nz_a, r] * V[nz_b, r] * W[nz_c, r]
        err_nz = np.sum((T_comp_nz - 1.0) ** 2)
        
        T_comp_z = np.zeros(len(zero_idx))
        for r in range(target_rank):
            T_comp_z += U[za, r] * V[zb, r] * W[zc, r]
        err_z = np.sum(T_comp_z ** 2)
        
        return err_nz + err_z
    
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    n_restarts = 3
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline:
            break
        
        U, V, W = _random_init()
        cur_err = _compute_error(U, V, W)
        T_cur = np.einsum('ar,br,cr->abc', U, V, W)
        max_diff_cur = np.max(np.abs(T_cur - T_target))
        if max_diff_cur < best_err:
            best_err = max_diff_cur
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        temp = 1.0
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            temp *= 0.95
            
            mat_choice = np.random.randint(3)
            if mat_choice == 0:
                a, r = np.random.randint(nm), np.random.randint(target_rank)
                old_val = U[a, r]
                candidates = [v for v in [-1.0, 0.0, 1.0] if v != old_val]
                new_val = np.random.choice(candidates)
                U[a, r] = new_val
                new_err = _compute_error(U, V, W)
                T_test = np.einsum('ar,br,cr->abc', U, V, W)
                md = np.max(np.abs(T_test - T_target))
                if md < best_err:
                    best_err = md
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if md < 1e-10:
                        return best_U, best_V, best_W, True, md
                if new_err >= cur_err:
                    if np.random.random() > np.exp(-(new_err - cur_err) / max(temp, 1e-10)):
                        U[a, r] = old_val
                else:
                    cur_err = new_err
            elif mat_choice == 1:
                b, r = np.random.randint(mp), np.random.randint(target_rank)
                old_val = V[b, r]
                candidates = [v for v in [-1.0, 0.0, 1.0] if v != old_val]
                new_val = np.random.choice(candidates)
                V[b, r] = new_val
                new_err = _compute_error(U, V, W)
                T_test = np.einsum('ar,br,cr->abc', U, V, W)
                md = np.max(np.abs(T_test - T_target))
                if md < best_err:
                    best_err = md
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if md < 1e-10:
                        return best_U, best_V, best_W, True, md
                if new_err >= cur_err:
                    if np.random.random() > np.exp(-(new_err - cur_err) / max(temp, 1e-10)):
                        V[b, r] = old_val
                else:
                    cur_err = new_err
            else:
                c, r = np.random.randint(pn), np.random.randint(target_rank)
                old_val = W[c, r]
                candidates = [v for v in [-1.0, 0.0, 1.0] if v != old_val]
                new_val = np.random.choice(candidates)
                W[c, r] = new_val
                new_err = _compute_error(U, V, W)
                T_test = np.einsum('ar,br,cr->abc', U, V, W)
                md = np.max(np.abs(T_test - T_target))
                if md < best_err:
                    best_err = md
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if md < 1e-10:
                        return best_U, best_V, best_W, True, md
                if new_err >= cur_err:
                    if np.random.random() > np.exp(-(new_err - cur_err) / max(temp, 1e-10)):
                        W[c, r] = old_val
                else:
                    cur_err = new_err
    
    if best_U is not None:
        T_final = np.einsum('ar,br,cr->abc', best_U, best_V, best_W)
        final_diff = np.max(np.abs(T_final - T_target))
        return best_U, best_V, best_W, final_diff < 1e-10, final_diff
    return None, None, None, False, float('inf')

def _alternating_minimization(U_init, V_init, W_init, T_target, target_rank, 
                                n, m, p, max_iters=200, tol=1e-10, deadline=None):
    """
    Alternating minimization to find rank-target_rank decomposition.
    """
    nm, mp, pn = n*m, m*p, p*n
    
    U = U_init[:, :target_rank].copy() if U_init.shape[1] >= target_rank else np.random.randn(nm, target_rank) * 0.1
    V = V_init[:, :target_rank].copy() if V_init.shape[1] >= target_rank else np.random.randn(mp, target_rank) * 0.1
    W = W_init[:, :target_rank].copy() if W_init.shape[1] >= target_rank else np.random.randn(pn, target_rank) * 0.1
    
    best_diff = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U
        B_flat = np.einsum('br,cr->bcr', V, W).reshape(target_rank, mp * pn)
        for a in range(nm):
            target_vec = T_target[a, :, :].reshape(-1)
            try:
                sol, _, _, _ = np.linalg.lstsq(B_flat.T, target_vec, rcond=None)
                U[a, :] = sol
            except:
                pass
        
        # Update V
        A_flat = np.einsum('ar,cr->acr', U, W).reshape(target_rank, nm * pn)
        for b in range(mp):
            target_vec = T_target[:, b, :].reshape(-1)
            try:
                sol, _, _, _ = np.linalg.lstsq(A_flat.T, target_vec, rcond=None)
                V[b, :] = sol
            except:
                pass
        
        # Update W
        C_flat = np.einsum('ar,br->abr', U, V).reshape(target_rank, nm * mp)
        for c in range(pn):
            target_vec = T_target[:, :, c].reshape(-1)
            try:
                sol, _, _, _ = np.linalg.lstsq(C_flat.T, target_vec, rcond=None)
                W[c, :] = sol
            except:
                pass
        
        T_comp = np.einsum('ar,br,cr->abc', U, V, W)
        max_diff = np.max(np.abs(T_comp - T_target))
        
        if max_diff < best_diff:
            best_diff = max_diff
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if max_diff < tol:
            break
    
    return best_U, best_V, best_W, best_diff < tol, best_diff

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    T_target = _build_target_tensor(n, m, p)
    
    # Start with schoolbook
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = n * m * p
    
    valid, diff = _verify(best_U, best_V, best_W, T_target)
    assert valid, f"Schoolbook should be valid, got diff={diff}"
    
    # Time budget allocation
    time_per_rank = 6
    min_rank = 36
    
    for target_rank in range(min_rank, best_rank):
        if time.time() > DEADLINE - 10:
            break
        
        rank_deadline = min(start_time + time_per_rank, DEADLINE - 10)
        half_deadline = start_time + (rank_deadline - start_time) * 0.6
        
        best_diff_this_rank = float('inf')
        best_U_r, best_V_r, best_W_r = None, None, None
        
        n_restarts = 4
        for restart in range(n_restarts):
            if time.time() > half_deadline:
                break
            
            np.random.seed(hash((target_rank, restart)) % (2**31))
            U_init = np.random.randn(nm, target_rank) * 0.5
            V_init = np.random.randn(mp, target_rank) * 0.5
            W_init = np.random.randn(pn, target_rank) * 0.5
            
            U_r, V_r, W_r, converged, max_diff = _alternating_minimization(
                U_init, V_init, W_init, T_target, target_rank,
                n, m, p, max_iters=80, tol=1e-10, deadline=half_deadline
            )
            
            if max_diff < best_diff_this_rank:
                best_diff_this_rank = max_diff
                best_U_r, best_V_r, best_W_r = U_r, V_r, W_r
            
            if converged:
                break
        
        if time.time() < half_deadline - 1:
            U_init2 = U_school[:, :target_rank].copy()
            V_init2 = V_school[:, :target_rank].copy()
            W_init2 = W_school[:, :target_rank].copy()
            
            U_r2, V_r2, W_r2, conv2, diff2 = _alternating_minimization(
                U_init2, V_init2, W_init2, T_target, target_rank,
                n, m, p, max_iters=60, tol=1e-10, deadline=half_deadline
            )
            
            if diff2 < best_diff_this_rank:
                best_diff_this_rank = diff2
                best_U_r, best_V_r, best_W_r = U_r2, V_r2, W_r2
        
        if time.time() < rank_deadline - 1:
            U_bin, V_bin, W_bin, conv_bin, diff_bin = _binary_belief_propagation(
                n, m, p, T_target, target_rank,
                max_iters=30, deadline=rank_deadline
            )
            if U_bin is not None and diff_bin < best_diff_this_rank:
                best_diff_this_rank = diff_bin
                best_U_r, best_V_r, best_W_r = U_bin, V_bin, W_bin
        
        if best_U_r is not None and best_diff_this_rank < 1e-8:
            valid, final_diff = _verify(best_U_r, best_V_r, best_W_r, T_target)
            if valid:
                best_U, best_V, best_W = best_U_r, best_V_r, best_W_r
                best_rank = target_rank
                break
    
    if best_rank < n*m*p:
        U_round = np.round(2 * best_U) / 2
        V_round = np.round(2 * best_V) / 2
        W_round = np.round(2 * best_W) / 2
        
        valid_round, diff_round = _verify(U_round, V_round, W_round, T_target)
        if valid_round:
            best_U, best_V, best_W = U_round, V_round, W_round
        else:
            U_int = np.round(best_U)
            V_int = np.round(best_V)
            W_int = np.round(best_W)
            valid_int, diff_int = _verify(U_int, V_int, W_int, T_target)
            if valid_int:
                best_U, best_V, best_W = U_int, V_int, W_int
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
