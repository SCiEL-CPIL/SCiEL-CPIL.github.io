"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def _build_target():
    """Build the target multiplication tensor once."""
    n, m, p = DIMENSIONS
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    return target

_TARGET = None
def _get_target():
    global _TARGET
    if _TARGET is None:
        _TARGET = _build_target()
    return _TARGET

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    target = _get_target()
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    err = np.max(np.abs(T - target))
    return err <= tol

def _verify_detailed(U, V, W, tol=1e-9):
    """Return max error for debugging."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    target = _get_target()
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    return np.max(np.abs(T - target))

def _round_and_verify(U, V, W, tol=1e-5):
    """Try to round U, V, W to integers or half-integers and verify."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Try half-integers
    U_half = np.round(U_arr * 2) / 2
    V_half = np.round(V_arr * 2) / 2
    W_half = np.round(W_arr * 2) / 2
    if _verify(U_half.tolist(), V_half.tolist(), W_half.tolist(), tol=tol):
        return U_half.tolist(), V_half.tolist(), W_half.tolist()
        
    # Try integers
    U_int = np.round(U_arr)
    V_int = np.round(V_arr)
    W_int = np.round(W_arr)
    if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), tol=tol):
        return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
    return None

def _build_rank44_decomposition():
    """
    Build explicit rank-44 decomposition for 2x3 * 3x8.
    Split into 4 blocks of 2x3 * 3x2, each using Strassen-7 + 4 outer products.
    """
    n, m, p = 2, 3, 8
    U = np.zeros((n*m, 44))
    V = np.zeros((m*p, 44))
    W = np.zeros((p*n, 44))
    
    col_offset = 0
    for block in range(4):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # Strassen rank-7 for 2x2*2x2 (A cols 0,1 × B rows 0,1)
        # m1 = (a00+a11)(b00+b11)
        t = col_offset
        U[0, t] += 1; U[4, t] += 1
        V[k0, t] += 1; V[8+k1, t] += 1
        W[2*k0, t] += 1; W[2*k1+1, t] += 1
        col_offset += 1
        
        # m2 = (a10+a11)b00
        t = col_offset
        U[3, t] += 1; U[4, t] += 1
        V[k0, t] += 1
        W[2*k0+1, t] += 1; W[2*k1+1, t] += -1
        col_offset += 1
        
        # m3 = a00(b01-b11)
        t = col_offset
        U[0, t] += 1
        V[k1, t] += 1; V[8+k1, t] += -1
        W[2*k1, t] += 1; W[2*k1+1, t] += 1
        col_offset += 1
        
        # m4 = a11(b10-b00)
        t = col_offset
        U[4, t] += 1
        V[8+k0, t] += 1; V[k0, t] += -1
        W[2*k0, t] += 1; W[2*k0+1, t] += 1
        col_offset += 1
        
        # m5 = (a00+a01)b11
        t = col_offset
        U[0, t] += 1; U[1, t] += 1
        V[8+k1, t] += 1
        W[2*k0, t] += -1; W[2*k1, t] += 1
        col_offset += 1
        
        # m6 = (a10-a00)(b00+b01)
        t = col_offset
        U[3, t] += 1; U[0, t] += -1
        V[k0, t] += 1; V[k1, t] += 1
        W[2*k1+1, t] += 1
        col_offset += 1
        
        # m7 = (a01-a11)(b10+b11)
        t = col_offset
        U[1, t] += 1; U[4, t] += -1
        V[8+k0, t] += 1; V[8+k1, t] += 1
        W[2*k0, t] += 1
        col_offset += 1
        
        # 4 outer product terms for A[:,2] * B[2,:]
        for ii in range(2):
            for kk in range(2):
                t = col_offset
                k_val = k0 if kk == 0 else k1
                U[ii*3+2, t] = 1
                V[16+k_val, t] = 1
                W[k_val*2+ii, t] = 1
                col_offset += 1
    
    assert col_offset == 44
    return U.tolist(), V.tolist(), W.tolist()

def _try_remove_term(U, V, W, term_idx, deadline):
    """
    Try removing term term_idx and see if we can adjust remaining terms
    to still satisfy the decomposition. Uses least-squares on the residual.
    Returns (U_new, V_new, W_new) if successful, else None.
    """
    if time.time() > deadline:
        return None
    
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Remove the term
    keep = [i for i in range(rank) if i != term_idx]
    U_keep = U_arr[:, keep]
    V_keep = V_arr[:, keep]
    W_keep = W_arr[:, keep]
    
    # Compute what the kept terms give
    T_kept = np.einsum('ar,br,cr->abc', U_keep, V_keep, W_keep)
    
    # Target
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    residual = target - T_kept
    
    # The removed term was: U[:,term_idx] ⊗ V[:,term_idx] ⊗ W[:,term_idx]
    # We need: sum_{r in keep} delta_U[r] ⊗ V_keep[:,r] ⊗ W_keep[:,r] 
    #           + sum_{r in keep} U_keep[:,r] ⊗ delta_V[r] ⊗ W_keep[:,r]
    #           + sum_{r in keep} U_keep[:,r] ⊗ V_keep[:,r] ⊗ delta_W[r]
    #         = residual
    # This is complex. Instead, just check if residual is close to zero.
    err = np.max(np.abs(residual))
    if err < 1e-10:
        # Perfect - just drop the term
        return U_keep.tolist(), V_keep.tolist(), W_keep.tolist()
    
    # Try to absorb into existing terms via small adjustments
    # For each kept term r, we can adjust U_keep[:,r], V_keep[:,r], W_keep[:,r]
    # Linearize: for small delta, the change in T is approximately:
    # dT = sum_r (dU_r ⊗ V_r ⊗ W_r + U_r ⊗ dV_r ⊗ W_r + U_r ⊗ V_r ⊗ dW_r)
    # This is a linear system in the deltas.
    
    num_keep = len(keep)
    # Unknowns: num_keep * (n*m + m*p + p*n)
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    total_unknowns = num_keep * (dim_u + dim_v + dim_w)
    
    # Build the linear system: A * x = residual.flatten()
    # This is too large for direct solve, use iterative approach instead.
    # Skip for now - just return None if not exact.
    
    return None

def _als_decompose_fast(rank, max_iters=100, deadline=None, init=None):
    """
    Fast ALS for tensor decomposition using vectorized operations.
    """
    n, m, p = DIMENSIONS
    
    if init is not None:
        U = np.array(init[0], dtype=float)
        V = np.array(init[1], dtype=float)
        W = np.array(init[2], dtype=float)
        if U.shape[1] != rank:
            U = U[:, :rank] if U.shape[1] >= rank else np.hstack([U, np.random.randn(n*m, rank-U.shape[1])])
        if V.shape[1] != rank:
            V = V[:, :rank] if V.shape[1] >= rank else np.hstack([V, np.random.randn(m*p, rank-V.shape[1])])
        if W.shape[1] != rank:
            W = W[:, :rank] if W.shape[1] >= rank else np.hstack([W, np.random.randn(p*n, rank-W.shape[1])])
    else:
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
    
    target = _get_target()
    
    # Precompute index mappings
    # For U update: for each (i,j), M[k,:] = V[j*p+k,:] * W[k*n+i,:]
    u_indices = []
    for i in range(n):
        for j in range(m):
            a = i * m + j
            vs = np.array([j*p+k for k in range(p)])
            ws = np.array([k*n+i for k in range(p)])
            u_indices.append((a, vs, ws))
    
    # For V update: for each (j,k), M has rows for each (i,kk): U[i*m+j,:] * W[kk*n+i,:]
    v_indices = []
    for j in range(m):
        for k in range(p):
            b = j * p + k
            rows = []
            for i in range(n):
                for kk in range(p):
                    a = i * m + j
                    c = kk * n + i
                    rows.append((a, c))
            v_indices.append((b, rows))
    
    # For W update: for each (k,i), M has rows for each (ii,j): U[ii*m+j,:] * V[j*p+k,:]
    w_indices = []
    for k in range(p):
        for i in range(n):
            c = k * n + i
            rows = []
            for ii in range(n):
                for j in range(m):
                    a = ii * m + j
                    b = j * p + k
                    rows.append((a, b))
            w_indices.append((c, rows))
    
    ones_p = np.ones(p)
    ones_np = np.ones(n*p)
    ones_nm = np.ones(n*m)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Update U
        for a, vs, ws in u_indices:
            M = V[vs] * W[ws]
            try:
                x, _, _, _ = np.linalg.lstsq(M, ones_p, rcond=None)
                U[a, :] = x
            except:
                pass
        
        # Update V
        for b, rows in v_indices:
            M = np.array([U[a] * W[c] for a, c in rows])
            try:
                x, _, _, _ = np.linalg.lstsq(M, ones_np, rcond=None)
                V[b, :] = x
            except:
                pass
        
        # Update W
        for c, rows in w_indices:
            M = np.array([U[a] * V[b] for a, b in rows])
            try:
                x, _, _, _ = np.linalg.lstsq(M, ones_nm, rcond=None)
                W[c, :] = x
            except:
                pass
        
        # Check convergence
        T = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T - target))
        if err < 1e-10:
            return U.tolist(), V.tolist(), W.tolist()
            
    return None

def _polish(U, V, W, deadline, max_iters=50):
    """
    High-precision polish: use Newton-like iterations to drive error to machine precision.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    target = _get_target()
    
    # Precompute index mappings
    u_indices = []
    for i in range(n):
        for j in range(m):
            a = i * m + j
            vs = np.array([j*p+k for k in range(p)])
            ws = np.array([k*n+i for k in range(p)])
            u_indices.append((a, vs, ws))
    
    v_indices = []
    for j in range(m):
        for k in range(p):
            b = j * p + k
            rows = []
            for i in range(n):
                for kk in range(p):
                    a = i * m + j
                    c = kk * n + i
                    rows.append((a, c))
            v_indices.append((b, rows))
    
    w_indices = []
    for k in range(p):
        for i in range(n):
            c = k * n + i
            rows = []
            for ii in range(n):
                for j in range(m):
                    a = ii * m + j
                    b = j * p + k
                    rows.append((a, b))
            w_indices.append((c, rows))
    
    ones_p = np.ones(p)
    ones_np = np.ones(n*p)
    ones_nm = np.ones(n*m)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        err = np.max(np.abs(T - target))
        if err < 1e-14:
            break
        
        for a, vs, ws in u_indices:
            M = V_arr[vs] * W_arr[ws]
            try:
                x, _, _, _ = np.linalg.lstsq(M, ones_p, rcond=None)
                U_arr[a, :] = x
            except:
                pass
        
        for b, rows in v_indices:
            M = np.array([U_arr[a] * W_arr[c] for a, c in rows])
            try:
                x, _, _, _ = np.linalg.lstsq(M, ones_np, rcond=None)
                V_arr[b, :] = x
            except:
                pass
        
        for c, rows in w_indices:
            M = np.array([U_arr[a] * V_arr[b] for a, b in rows])
            try:
                x, _, _, _ = np.linalg.lstsq(M, ones_nm, rcond=None)
                W_arr[c, :] = x
            except:
                pass
    
    return U_arr.tolist(), V_arr.tolist(), W_arr.tolist()

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    # Fallback: schoolbook
    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
                
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = schoolbook_rank
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Step 1: Build rank-44 decomposition
    try:
        U44, V44, W44 = _build_rank44_decomposition()
        if _verify(U44, V44, W44, tol=1e-6):
            best_U, best_V, best_W = U44, V44, W44
            best_rank = 44
        else:
            rounded = _round_and_verify(U44, V44, W44)
            if rounded:
                best_U, best_V, best_W = rounded
                best_rank = 44
    except Exception:
        pass
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Step 2: Try ALS for lower ranks with multiple restarts
    target_ranks = [42, 41, 40, 39, 38, 37, 36, 35]
    
    for rank in target_ranks:
        if time.time() > deadline - 20:
            break
        
        found = False
        num_trials = 5 if rank >= 40 else 3
        
        for trial in range(num_trials):
            if time.time() > deadline - 20:
                break
            
            # Initialize
            if rank < best_rank and best_rank <= 44:
                # Use truncated current best as init
                U_init = np.array(best_U, dtype=float)[:, :rank]
                V_init = np.array(best_V, dtype=float)[:, :rank]
                W_init = np.array(best_W, dtype=float)[:, :rank]
                # Add small random perturbation
                U_init = U_init + np.random.randn(n*m, rank) * 0.01
                V_init = V_init + np.random.randn(m*p, rank) * 0.01
                W_init = W_init + np.random.randn(p*n, rank) * 0.01
                init = (U_init.tolist(), V_init.tolist(), W_init.tolist())
            else:
                init = None
            
            max_iters = 150 if rank >= 40 else 200
            result = _als_decompose_fast(rank, max_iters=max_iters, deadline=deadline - 20, init=init)
            if result:
                U, V, W = result
                if _verify(U, V, W, tol=1e-6):
                    # Polish
                    U_pol, V_pol, W_pol = _polish(U, V, W, deadline - 5, max_iters=30)
                    if _verify(U_pol, V_pol, W_pol, tol=1e-8):
                        rounded = _round_and_verify(U_pol, V_pol, W_pol)
                        if rounded:
                            best_U, best_V, best_W = rounded
                        else:
                            best_U, best_V, best_W = U_pol, V_pol, W_pol
                    else:
                        best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    found = True
                    break
    
    # Step 3: Final polish and rounding
    if time.time() < deadline - 5:
        U_pol, V_pol, W_pol = _polish(best_U, best_V, best_W, deadline, max_iters=30)
        if _verify(U_pol, V_pol, W_pol, tol=1e-8):
            rounded = _round_and_verify(U_pol, V_pol, W_pol)
            if rounded:
                best_U, best_V, best_W = rounded
            else:
                best_U, best_V, best_W = U_pol, V_pol, W_pol
    
    return {"U": best_U, "V": best_V, "W": best_W}
