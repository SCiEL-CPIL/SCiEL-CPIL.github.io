"""Tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _validate(U, V, W, n=4, m=4, p=5, tol=1e-9):
    """Check that U,V,W form a valid tensor decomposition of matmul(n,m,p)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    
    M_decomp = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)
    
    if not np.allclose(M_decomp, 1.0, atol=tol):
        return False
    return True

def _construct_schoolbook(n=4, m=4, p=5):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _strassen_2x2x2_terms():
    """
    Return the 7 rank-1 tensors for 2x2x2 matrix multiplication.
    Each term is (u, v, w) where u is 4-vector (i*2+j), v is 4-vector (j*2+k), w is 4-vector (k*2+i)
    """
    # Standard Strassen 7 terms for 2x2x2
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Term 1: (A00+A11)(B00+B11) -> C00 += , C11 +=
    # Term 2: (A10+A11)B00 -> C10 +=
    # Term 3: A00(B10-B00) -> C01 -=, C11 += ... wait, let's use a standard verified set
    
    # Let's use the well-known Strassen decomposition:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B00)
    # M4 = A11 * (B01 - B11)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to express this as sum_r u_r[i,j] v_r[j,k] w_r[k,i] = delta_{i,k} for the tensor
    # Actually, the matrix multiplication tensor T[i,j,k] = 1 if the indices align properly.
    # For 2x2x2: T[i,j,k] = 1 for all i,j,k? No.
    # The condition is: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
    # This means the tensor is the identity in the sense that it computes matmul.
    # The tensor T has T[i,j,k] = 1 for all i,j,k in {0,1}.
    # Wait, no. The tensor for matmul is T[i,j,k] = 1 if we are computing C[i,k] += A[i,j]B[j,k].
    # So for each (i,j,k), the coefficient is 1.
    # So we need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
    
    # Let's verify the Strassen terms satisfy this.
    # Term 1: u = [1,0, 0,1] (A00, A11), v = [1,0, 0,1] (B00, B11), w = [1,0, 0,1] (C00, C11)
    #   U[i,j]: u[i*2+j]
    #   V[j,k]: v[j*2+k]
    #   W[k,i]: w[k*2+i]
    #   For i=0,j=0,k=0: U=1, V=1, W=1 -> 1
    #   For i=0,j=0,k=1: U=1, V=0, W=0 -> 0
    #   For i=0,j=1,k=0: U=0, V=1, W=1 -> 0
    #   For i=0,j=1,k=1: U=0, V=0, W=0 -> 0
    #   For i=1,j=0,k=0: U=0, V=1, W=0 -> 0
    #   For i=1,j=0,k=1: U=0, V=0, W=1 -> 0
    #   For i=1,j=1,k=0: U=1, V=0, W=0 -> 0
    #   For i=1,j=1,k=1: U=1, V=1, W=1 -> 1
    #   Sum so far: T[0,0,0]=1, T[1,1,1]=1, others 0.
    
    # This is getting complex. Let's just use a known valid set or construct one carefully.
    # Actually, the schoolbook is always valid. Let's stick with a robust approach.
    
    terms = []
    # Term 1
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1]
    terms.append((u1, v1, w1))
    
    # Term 2
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 1, 0, 0]
    terms.append((u2, v2, w2))
    
    # Term 3
    u3 = [1, 0, 0, 0]
    v3 = [0, 0, 1, -1]
    w3 = [0, 0, 1, 0]
    terms.append((u3, v3, w3))
    
    # Term 4
    u4 = [0, 0, 0, 1]
    v4 = [1, -1, 0, 0]
    w4 = [0, 0, 0, 1]
    terms.append((u4, v4, w4))
    
    # Term 5
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [0, 0, 1, 0]
    terms.append((u5, v5, w5))
    
    # Term 6
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [0, 1, 0, 0]
    terms.append((u6, v6, w6))
    
    # Term 7
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    w7 = [0, 0, 0, 1]
    terms.append((u7, v7, w7))
    
    return terms

def _embed_strassen_2x2x2(n=4, m=4, p=5):
    """Embed the 2x2x2 Strassen decomposition into the top-left 2x2x2 part of 4x4x5."""
    terms = _strassen_2x2x2_terms()
    
    U = np.zeros((n*m, 7))
    V = np.zeros((m*p, 7))
    W = np.zeros((p*n, 7))
    
    for r, (u, v, w) in enumerate(terms):
        for i in range(2):
            for j in range(2):
                U[i*m + j, r] = u[i*2 + j]
        for j in range(2):
            for k in range(2):
                V[j*p + k, r] = v[j*2 + k]
        for k in range(2):
            for i in range(2):
                W[k*n + i, r] = w[k*2 + i]
    
    return U, V, W

def _construct_hybrid(n=4, m=4, p=5):
    """
    Construct a hybrid decomposition:
    - Use Strassen for the 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1})
    - Use schoolbook for all other (i,j,k) triples
    
    This gives rank = 7 + (4*4*5 - 2*2*2) = 7 + (80 - 8) = 79
    """
    rank = 79
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Fill first 7 columns with Strassen
    U_str, V_str, W_str = _embed_strassen_2x2x2(n, m, p)
    U[:, :7] = U_str
    V[:, :7] = V_str
    W[:, :7] = W_str
    
    # Fill remaining columns with schoolbook for non-Strassen positions
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue  # Already handled by Strassen
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_merge_pair(U, V, W, a, b, n, m, p):
    """
    Try to merge columns a and b into a single column.
    Returns (U_new, V_new, W_new) if successful, else None.
    """
    # Compute the combined tensor contribution from columns a and b
    # T[i,j,k] = U[i*m+j,a]*V[j*p+k,a]*W[k*n+i,a] + U[i*m+j,b]*V[j*p+k,b]*W[k*n+i,b]
    # We want to check if T is rank-1, i.e., T[i,j,k] = u[i,j]*v[j,k]*w[k,i]
    
    # Build the 3D tensor
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i,j,k] = (U[i*m+j, a]*V[j*p+k, a]*W[k*n+i, a] + 
                           U[i*m+j, b]*V[j*p+k, b]*W[k*n+i, b])
    
    # Check if T is rank-1
    # A tensor is rank-1 iff all its 2x2 minors (in any unfolding) are zero
    # More precisely, T[i,j,k] = u[i,j] v[j,k] w[k,i]
    # This implies: T[i1,j1,k1]*T[i2,j2,k2] = T[i1,j2,k1]*T[i2,j1,k2] for all i1,i2,j1,j2,k1,k2
    # (since u[i1,j1]v[j1,k1]w[k1,i1] * u[i2,j2]v[j2,k2]w[k2,i2] = u[i1,j2]v[j2,k1]w[k1,i1] * u[i2,j1]v[j1,k2]w[k2,i2]
    #  only if the cross terms cancel, which they don't in general)
    
    # Actually, the correct condition for rank-1 is:
    # T[i,j,k] / T[i0,j0,k0] = (T[i,j,k0]/T[i0,j0,k0]) * (T[i0,j,k]/T[i0,j0,k0]) * (T[i0,j0,k]/T[i0,j0,k0]) / (T[i0,j0,k0])
    # This is still complex. Let's use a simpler check:
    # If T is rank-1, then for any fixed i0, the matrix T[i0,:,:] must be rank-1 (outer product)
    # and consistent across i.
    
    # Find a non-zero entry to use as anchor
    anchor = None
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i,j,k]) > 1e-12:
                    anchor = (i,j,k)
                    break
            if anchor:
                break
        if anchor:
            break
    
    if anchor is None:
        # T is all zeros, we can remove both columns
        U_new = np.delete(U, [a, b], axis=1)
        V_new = np.delete(V, [a, b], axis=1)
        W_new = np.delete(W, [a, b], axis=1)
        return U_new, V_new, W_new
    
    i0, j0, k0 = anchor
    T0 = T[i0, j0, k0]
    
    # Check rank-1 condition using the anchor
    # For rank-1: T[i,j,k] * T[i0,j0,k0] = T[i,j,k0] * T[i0,j0,k] * T[i0,j,k] / T[i0,j0,k0]
    # Let's verify this:
    # T[i,j,k] = u[i,j] v[j,k] w[k,i]
    # T[i,j,k0] = u[i,j] v[j,k0] w[k0,i]
    # T[i0,j0,k] = u[i0,j0] v[j0,k] w[k,i0]
    # T[i0,j,k] = u[i0,j] v[j,k] w[k,i0]
    # 
    # T[i,j,k] * T[i0,j0,k0] = u[i,j] v[j,k] w[k,i] * u[i0,j0] v[j0,k0] w[k0,i0]
    # T[i,j,k0] * T[i0,j0,k] * T[i0,j,k] / T[i0,j0,k0]
    # = [u[i,j] v[j,k0] w[k0,i]] * [u[i0,j0] v[j0,k] w[k,i0]] * [u[i0,j] v[j,k] w[k,i0]] / [u[i0,j0] v[j0,k0] w[k0,i0]]
    # This doesn't simplify to the same thing in general.
    
    # Let's use a different approach: check if all 2x2 minors of the matricizations are zero.
    # Matricize T as a matrix of shape (n*m, p) by flattening (i,j) -> i*m+j
    # For rank-1, this matrix must have rank <= 1, so all 2x2 minors must be zero.
    T_mat = T.reshape(n*m, p)
    
    # Check all 2x2 minors
    is_rank1 = True
    for i1 in range(n*m):
        for i2 in range(i1+1, n*m):
            for k1 in range(p):
                for k2 in range(k1+1, p):
                    minor = T_mat[i1, k1]*T_mat[i2, k2] - T_mat[i1, k2]*T_mat[i2, k1]
                    if abs(minor) > 1e-9 * max(1.0, abs(minor)):
                        is_rank1 = False
                        break
                if not is_rank1:
                    break
            if not is_rank1:
                break
        if not is_rank1:
            break
    
    if not is_rank1:
        return None
    
    # Extract u, v, w vectors
    u_vec = np.zeros(n*m)
    v_vec = np.zeros(m*p)
    w_vec = np.zeros(p*n)
    
    # Gauge choice: set w_vec[k0*n+i0] = 1
    w_vec[k0*n+i0] = 1.0
    
    # From T[i0,j0,k0] = u_vec[i0*m+j0] * v_vec[j0*p+k0] * w_vec[k0*n+i0] = u_vec[i0*m+j0] * v_vec[j0*p+k0]
    # So u_vec[i0*m+j0] * v_vec[j0*p+k0] = T0
    
    # From T[i,j,k0] = u_vec[i*m+j] * v_vec[j*p+k0] * w_vec[k0*n+i]
    # And T[i0,j,k] = u_vec[i0*m+j] * v_vec[j*p+k] * w_vec[k*n+i0]
    # And T[i0,j0,k] = u_vec[i0*m+j0] * v_vec[j0*p+k] * w_vec[k*n+i0]
    
    # Let's solve iteratively
    # Initialize
    u_vec[:] = 1.0
    v_vec[:] = 1.0
    w_vec[:] = 1.0
    w_vec[k0*n+i0] = 1.0
    
    converged = False
    for _ in range(200):
        # Update u_vec
        for i in range(n):
            for j in range(m):
                # Use T[i,j,k0] = u_vec[i*m+j] * v_vec[j*p+k0] * w_vec[k0*n+i]
                denom = v_vec[j*p+k0] * w_vec[k0*n+i]
                if abs(denom) > 1e-15:
                    u_vec[i*m+j] = T[i,j,k0] / denom
        
        # Update v_vec
        for j in range(m):
            for k in range(p):
                # Use T[i0,j,k] = u_vec[i0*m+j] * v_vec[j*p+k] * w_vec[k*n+i0]
                denom = u_vec[i0*m+j] * w_vec[k*n+i0]
                if abs(denom) > 1e-15:
                    v_vec[j*p+k] = T[i0,j,k] / denom
        
        # Update w_vec
        for k in range(p):
            for i in range(n):
                # Use T[i0,j0,k] = u_vec[i0*m+j0] * v_vec[j0*p+k] * w_vec[k*n+i0]
                if k == k0 and i == i0:
                    continue  # Keep gauge
                denom = u_vec[i0*m+j0] * v_vec[j0*p+k]
                if abs(denom) > 1e-15:
                    w_vec[k*n+i] = T[i0,j0,k] / denom
        
        # Check convergence
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    pred = u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i]
                    err = max(err, abs(pred - T[i,j,k]))
        if err < 1e-10:
            converged = True
            break
    
    if not converged:
        return None
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                pred = u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i]
                if abs(pred - T[i,j,k]) > 1e-9:
                    return None
    
    # Create new matrices
    U_new = np.delete(U, [a, b], axis=1)
    V_new = np.delete(V, [a, b], axis=1)
    W_new = np.delete(W, [a, b], axis=1)
    
    pos = min(a, b)
    U_new = np.insert(U_new, pos, u_vec, axis=1)
    V_new = np.insert(V_new, pos, v_vec, axis=1)
    W_new = np.insert(W_new, pos, w_vec, axis=1)
    
    return U_new, V_new, W_new

def _greedy_merge(U, V, W, n=4, m=4, p=5, max_iter=50, time_limit=None, start_time=None):
    """
    Try to reduce rank by finding pairs of rank-1 terms that can be merged.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if start_time is None:
        start_time = time.time()
    
    for _ in range(max_iter):
        if time_limit and time.time() - start_time > time_limit:
            break
        
        merged = False
        rank = U.shape[1]
        
        for a in range(rank):
            if time_limit and time.time() - start_time > time_limit:
                break
            for b in range(a+1, rank):
                if time_limit and time.time() - start_time > time_limit:
                    break
                result = _try_merge_pair(U, V, W, a, b, n, m, p)
                if result is not None:
                    U, V, W = result
                    merged = True
                    break
            if merged:
                break
        
        if not merged:
            break
    
    return U, V, W

def construct():
    """Construct a rank-minimal tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # 110 second limit to be safe
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 80
    
    # Start with schoolbook
    best_result = _construct_schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Try the hybrid Strassen + schoolbook approach
    try:
        hybrid_result = _construct_hybrid(n, m, p)
        rank_h = len(hybrid_result["U"][0]) if hybrid_result["U"] else 0
        if rank_h < best_rank:
            if _validate(hybrid_result["U"], hybrid_result["V"], hybrid_result["W"], n, m, p):
                best_result = hybrid_result
                best_rank = rank_h
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return best_result
    
    # Try greedy merging on the best result
    try:
        U = np.array(best_result["U"], dtype=float)
        V = np.array(best_result["V"], dtype=float)
        W = np.array(best_result["W"], dtype=float)
        
        U_merged, V_merged, W_merged = _greedy_merge(U, V, W, n, m, p, max_iter=50, 
                                                       time_limit=deadline - time.time() - 10,
                                                       start_time=start_time)
        new_rank = U_merged.shape[1]
        
        if new_rank < best_rank:
            result = {"U": U_merged.tolist(), "V": V_merged.tolist(), "W": W_merged.tolist()}
            if _validate(result["U"], result["V"], result["W"], n, m, p):
                best_result = result
                best_rank = new_rank
    except Exception:
        pass
    
    # Check time again
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return best_result
    
    return best_result

# EVOLVE-BLOCK-END
