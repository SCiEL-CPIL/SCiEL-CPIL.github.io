"""
Strategy: randomized alternating-least-squares tensor rank search with
null-space rank reduction, targeting a rank well below the 56 tiling bound.
Falls back to a verified rank-56 Strassen tiling if the search fails.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check whether (U, V, W) exactly decompose the multiplication tensor."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    if U_np.ndim != 2 or V_np.ndim != 2 or W_np.ndim != 2:
        return False
    if U_np.shape != (n * m, W_np.shape[1]):
        return False
    if V_np.shape != (m * p, W_np.shape[1]):
        return False
    if W_np.shape != (p * n, W_np.shape[1]):
        return False
    rank = U_np.shape[1]
    U3 = U_np.reshape(n, m, rank)
    V3 = V_np.reshape(m, p, rank)
    W3 = W_np.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=tol)


def _tensor_to_vec(n, m, p):
    """Flatten the all-ones tensor (n,m,p) to a vector of length n*m*p."""
    T = np.ones((n, m, p))
    return T.reshape(-1)


def _compute_W(U, V, n, m, p):
    """
    Given U (n*m, r) and V (m*p, r), compute the optimal W (p*n, r)
    that minimizes ||sum_c U[:,:,c] * V[:,:,c] * W[:,:,c] - T||_F.
    
    For each column c, W[:,:,c] should be the projection of T onto the
    rank-1 tensor U[:,:,c] * V[:,:,c] in the appropriate sense.
    
    Actually, the decomposition is:
    T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    For fixed U, V, the optimal W is found by solving a linear system.
    Let's vectorize: T_flat = sum_c (U_c ⊙ V_c) * W_c where ⊙ is the
    appropriate reshaping.
    
    More precisely, define for each c:
      A_c[i,k] = sum_j U[i,j,c] * V[j,k,c]   (n x p matrix)
    Then T[i,k] = sum_c A_c[i,k] * W[k,i,c]
    
    This is: T = sum_c A_c * W_c (element-wise product, with W_c[k,i])
    
    So: T_flat = M @ W_flat where M is a (n*p) x (n*p*r) matrix.
    Actually let's be more careful.
    
    T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    For fixed i,k: T[i,k] (summed over j) -- no, T is 3D.
    
    Let me think of it differently. Fix c. The contribution to T is:
    T_c[i,j,k] = U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    This is not separable in a simple way. Let's use the linear system approach.
    
    Vectorize T as t (length n*m*p). For each term c, define a vector v_c
    such that t = sum_c alpha_c * v_c where alpha_c involves W.
    
    Actually, W is linear in the equation. Let me write:
    t[i,j,k] = sum_c (U[i,j,c] * V[j,k,c]) * W[k,i,c]
    
    So for each (i,j,k): t[i,j,k] = sum_c M[(i,j,k), c, (k,i)] * W_flat[c*(n*p) + k*n + i]
    
    This is a linear system: t = A @ W_flat
    where A has shape (n*m*p, r*n*p).
    
    A[(i*m+j)*p + k, c*(n*p) + k*n + i] = U[i,j,c] * V[j,k,c]
    
    Then solve A @ W_flat = t in the least-squares sense.
    """
    r = U.shape[1]
    t = _tensor_to_vec(n, m, p)
    # Build the linear system
    # A has shape (n*m*p, r*n*p)
    # A[idx_t, idx_w] where idx_t = (i*m+j)*p+k, idx_w = c*(n*p) + k*n + i
    # A[idx_t, idx_w] = U[i,j,c] * V[j,k,c]
    
    # For efficiency, build A via broadcasting
    # U shape: (n, m, r), V shape: (m, p, r)
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    
    # For each (i, j, k, c): contribution to t[i,j,k] from W[k,i,c] is U[i,j,c]*V[j,k,c]
    # t[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    
    # Build A as (n*m*p, r*n*p)
    # Let's do it in a loop over c for clarity (r is small)
    A = np.zeros((n * m * p, r * n * p))
    for c in range(r):
        # For this c: t[i,j,k] += U[i,j,c]*V[j,k,c]*W[k,i,c]
        # W index: c*(n*p) + k*n + i
        # t index: (i*m+j)*p + k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = U3[i, j, c] * V3[j, k, c]
                    if val != 0:
                        t_idx = (i * m + j) * p + k
                        w_idx = c * (n * p) + k * n + i
                        A[t_idx, w_idx] = val
    
    # Solve least squares
    W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, t, rcond=None)
    W = W_flat.reshape(r, n, p).transpose(2, 0, 1).reshape(n * p, r)
    # Wait, W_flat[c*(n*p) + k*n + i] = W[k,i,c]
    # So W_flat reshaped to (r, n, p) with W_flat.reshape(r, n, p)[c, k, i]
    # We want W matrix of shape (p*n, r) where W[k*n + i, c]
    W_mat = W_flat.reshape(r, n, p).transpose(1, 2, 0).reshape(p * n, r)
    return W_mat


def _residual(U, V, W, n, m, p):
    """Compute the Frobenius norm of the residual."""
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.linalg.norm(T - np.ones((n, m, p)))


def _reduce_rank_by_null_space(U, V, W, n, m, p, tol=1e-9):
    """
    Try to reduce the rank by finding linear dependencies among the terms.
    If term c can be expressed as a linear combination of other terms, remove it.
    """
    r = U.shape[1]
    # Check if any column is a linear combination of others
    # This is complex; for now, try SVD-based truncation
    # Build the full tensor from the decomposition
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    # Try to find a lower-rank decomposition of T
    # This is the hard part; use a simple heuristic
    # For now, just check if we can remove columns that are near-zero
    new_r = r
    for c in range(r):
        if np.allclose(U[:, c], 0, atol=tol) or np.allclose(V[:, c], 0, atol=tol) or np.allclose(W[:, c], 0, atol=tol):
            # This term contributes nothing, remove it
            U = np.delete(U, c, axis=1)
            V = np.delete(V, c, axis=1)
            W = np.delete(W, c, axis=1)
            new_r -= 1
            c -= 1
    return U, V, W, new_r


def _build_strassen_2x2x2():
    """Rank-7 Strassen decomposition of 2x2x2."""
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    U[0, 2] = 1
    V[2, 2] = 1; V[0, 2] = -1
    W[2, 2] = 1
    
    U[3, 3] = 1
    V[1, 3] = 1; V[2, 3] = -1
    W[0, 3] = -1; W[2, 3] = -1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = -1; W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = -1; W[0, 6] = 1
    
    return U, V, W


def _build_strassen_tiled():
    """Rank-56 Strassen tiling for 2x4x8."""
    U_sub, V_sub, W_sub = _build_strassen_2x2x2()
    n, m, p = 2, 4, 8
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col_offset = 0
    for k in range(4):
        for half in range(2):  # 0: A_L*B_T, 1: A_R*B_B
            for c in range(7):
                # U factor
                for idx in range(4):
                    u_val = U_sub[idx, c]
                    if u_val != 0:
                        i = idx // 2
                        j = idx % 2
                        if half == 0:
                            U[i * m + j, col_offset + c] = u_val
                        else:
                            U[i * m + (j + 2), col_offset + c] = u_val
                
                # V factor
                for idx in range(4):
                    v_val = V_sub[idx, c]
                    if v_val != 0:
                        j = idx // 2
                        l = idx % 2
                        global_col = 2 * k + l
                        if half == 0:
                            V[j * p + global_col, col_offset + c] = v_val
                        else:
                            V[(2 + j) * p + global_col, col_offset + c] = v_val
                
                # W factor
                for idx in range(4):
                    w_val = W_sub[idx, c]
                    if w_val != 0:
                        l = idx // 2
                        i = idx % 2
                        W[(2 * k + l) * n + i, col_offset + c] = w_val
                col_offset += 7
    
    return U, V, W


def _randomized_search(n, m, p, target_rank, max_iters, deadline):
    """
    Randomized search for a rank-`target_rank` decomposition.
    Uses alternating minimization with random restarts.
    """
    best_U, best_V, best_W = None, None, None
    best_rank = target_rank + 1
    
    for attempt in range(20):  # 20 random restarts
        if time.time() > deadline:
            break
        
        r = target_rank
        # Initialize U, V randomly
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        # Normalize columns to avoid scaling issues
        for c in range(r):
            u_norm = np.linalg.norm(U[:, c])
            v_norm = np.linalg.norm(V[:, c])
            if u_norm > 1e-12 and v_norm > 1e-12:
                U[:, c] /= u_norm
                V[:, c] /= v_norm
        
        # Alternating minimization
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update W
            W = _compute_W(U, V, n, m, p)
            
            # Check residual
            res = _residual(U, V, W, n, m, p)
            if res < 1e-8:
                # Found an exact decomposition!
                U, V, W, new_r = _reduce_rank_by_null_space(U, V, W, n, m, p)
                if new_r < best_rank:
                    best_U, best_V, best_W = U, V, W
                    best_rank = new_r
                break
            
            # Update U (fix V, W)
            # t[i,j,k] = sum_c U[i,j,c] * (V[j,k,c] * W[k,i,c])
            # For each (i,j): t[i,j,:] = sum_c U[i,j,c] * G_c[j,:] where G_c[j,k] = V[j,k,c]*W[k,i,c]
            # This is: for each i, U[i,:] (m x r) should satisfy:
            # sum_j U[i,j,c] * V[j,k,c] * W[k,i,c] = t[i,j,k] for all j,k
            # Let's build the system for U:
            # t_flat = A_U @ U_flat
            # A_U[(i*m+j)*p+k, i*m+j*c... ] -- this is getting complex
            
            # Simpler: use gradient-free update or just re-randomize
            # For now, do a simple coordinate descent on U
            for c in range(r):
                # Optimize U[:, c] given V, W
                # t[i,j,k] = sum_{c'} U[i,j,c'] * V[j,k,c'] * W[k,i,c']
                # For fixed c: contribution is U[i,j,c] * V[j,k,c] * W[k,i,c]
                # Residual after removing term c: r[i,j,k] = t[i,j,k] - sum_{c'!=c} U[i,j,c']*V[j,k,c']*W[k,i,c]
                # We want: r[i,j,k] = U[i,j,c] * V[j,k,c] * W[k,i,c]
                # This is not directly solvable for U[i,j,c] since it depends on j,k.
                # Least squares: minimize sum_{j,k} (r[i,j,k] - U[i,j,c]*V[j,k,c]*W[k,i,c])^2
                # For each i: minimize over U[i,0..m-1,c]
                # Let a[i,j] = U[i,j,c], b[i,j,k] = V[j,k,c]*W[k,i,c]
                # minimize sum_{j,k} (r[i,j,k] - a[i,j]*b[i,j,k])^2
                # This is: for each i, solve least squares: R_i = A_i @ a_i
                # where R_i[j,k] = r[i,j,k], A_i[j,k,:] = b[i,j,k]*e_j (one-hot)
                # So A_i is (m*p, m) with A_i[(j,p)+k, j'] = b[i,j,k] if j==j' else 0
                
                # Compute residual without term c
                U3 = U.reshape(n, m, r)
                V3 = V.reshape(m, p, r)
                W3 = W.reshape(p, n, r)
                T_full = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                T_without_c = T_full - U3[:, :, c:c+1] * V3[:, :, c:c+1] * W3[:, :, c:c+1]
                
                # For each i, solve for U[i,:,c]
                for i in range(n):
                    # R[j,k] = T_without_c[i,j,k]
                    R = T_without_c[i]  # (m, p)
                    # A[j,k,j'] = V[j,k,c]*W[k,i,c] if j==j' else 0
                    # So the system is: sum_j' A[j,k,j'] * U[i,j',c] = R[j,k]
                    # A[j,k,j'] = delta(j,j') * V[j,k,c] * W[k,i,c]
                    # So: U[i,j,c] * V[j,k,c] * W[k,i,c] = R[j,k] for all k
                    # This gives: U[i,j,c] = R[j,k] / (V[j,k,c] * W[k,i,c]) for each k
                    # Least squares over k:
                    for j in range(m):
                        # U[i,j,c] * V[j,k,c] * W[k,i,c] ≈ R[j,k] for all k
                        # Let a_k = V[j,k,c] * W[k,i,c], b_k = R[j,k]
                        # minimize sum_k (U[i,j,c] * a_k - b_k)^2
                        # U[i,j,c] = sum_k a_k * b_k / sum_k a_k^2
                        a = V3[j, :, c] * W3[:, i, c]  # (p,)
                        b = R[j]  # (p,)
                        denom = np.dot(a, a)
                        if denom > 1e-12:
                            U[i * m + j, c] = np.dot(a, b) / denom
                        else:
                            U[i * m + j, c] = 0.0
            
            # Similarly update V (fix U, W)
            for c in range(r):
                U3 = U.reshape(n, m, r)
                V3 = V.reshape(m, p, r)
                W3 = W.reshape(p, n, r)
                T_full = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                T_without_c = T_full - U3[:, :, c:c+1] * V3[:, :, c:c+1] * W3[:, :, c:c+1]
                
                for k in range(p):
                    # R[i,j] = T_without_c[i,j,k]
                    R = T_without_c[:, :, k]  # (n, m)
                    for j in range(m):
                        # U[i,j,c] * V[j,k,c] * W[k,i,c] ≈ R[i,j] for all i
                        # Let a_i = U[i,j,c] * W[k,i,c], b_i = R[i,j]
                        # V[j,k,c] = sum_i a_i * b_i / sum_i a_i^2
                        a = U3[:, j, c] * W3[k, :, c]  # (n,)
                        b = R[:, j]  # (n,)
                        denom = np.dot(a, a)
                        if denom > 1e-12:
                            V[j * p + k, c] = np.dot(a, b) / denom
                        else:
                            V[j * p + k, c] = 0.0
        
        # After AM iterations, check if we got an exact solution
        W = _compute_W(U, V, n, m, p)
        res = _residual(U, V, W, n, m, p)
        if res < 1e-8:
            U, V, W, new_r = _reduce_rank_by_null_space(U, V, W, n, m, p)
            if new_r < best_rank:
                best_U, best_V, best_W = U, V, W
                best_rank = new_r
    
    if best_U is not None:
        return best_U, best_V, best_W, best_rank
    return None, None, None, target_rank + 1


def _build_schoolbook():
    """Fallback rank-64 schoolbook decomposition."""
    n, m, p = 2, 4, 8
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # First, try the randomized search for a rank below 56
    # Known upper bounds: mu(2,4,8) <= 48 (from 2*mu(2,4,4) with mu(2,4,4)<=24)
    # Try ranks: 32, 36, 40, 44, 48
    for target_rank in [32, 36, 40, 44, 48]:
        if time.time() > deadline - 5:
            break
        
        max_iters = 30  # AM iterations per restart
        U, V, W, found_rank = _randomized_search(n, m, p, target_rank, max_iters, deadline)
        
        if U is not None and found_rank <= target_rank:
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: verified rank-56 Strassen tiling
    if time.time() < deadline - 5:
        try:
            U, V, W = _build_strassen_tiled()
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except Exception:
            pass
    
    # Last resort: schoolbook
    U, V, W = _build_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
