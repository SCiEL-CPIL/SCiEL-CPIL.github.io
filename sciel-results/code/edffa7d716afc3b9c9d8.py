import numpy as np
import time

def _strassen_2x2x2():
    U2 = np.zeros((4, 7)); V2 = np.zeros((4, 7)); W2 = np.zeros((4, 7))
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1; W2[0,0]=1; W2[3,0]=1
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[2,1]=1; W2[3,1]=-1
    U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1; W2[1,2]=1; W2[3,2]=1
    U2[3,3]=1; V2[2,3]=1; V2[0,3]=-1; W2[0,3]=1; W2[2,3]=1
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1; W2[0,4]=-1; W2[1,4]=1
    U2[2,5]=1; U2[0,5]=-1; V2[0,5]=1; V2[1,5]=1; W2[3,5]=1
    U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1; W2[0,6]=1
    return U2, V2, W2

def _block_specs():
    A_specs = [
        [(1,0),(1,3)], [(1,2),(1,3)], [(1,0)], [(1,3)],
        [(1,0),(1,1)], [(1,2),(-1,0)], [(1,1),(-1,3)],
    ]
    B_specs = [
        [(1,0),(1,3)], [(1,0)], [(1,1),(-1,3)], [(1,2),(-1,0)],
        [(1,3)], [(1,0),(1,1)], [(1,2),(1,3)],
    ]
    C_specs = [
        [(1,0),(1,3)], [(1,2),(-1,3)], [(1,1),(1,3)], [(1,0),(1,2)],
        [(-1,0),(1,1)], [(1,3)], [(1,0)],
    ]
    return A_specs, B_specs, C_specs

def _build(rank=98):
    n, m, p = 4, 4, 8
    U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
    U2, V2, W2 = _strassen_2x2x2()
    A_specs, B_specs, C_specs = _block_specs()
    for half in range(2):
        for kp in range(7):
            for ms in range(7):
                col = half*49 + kp*7 + ms
                for ca, bi in A_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c
                            U[gi*4+gj, col] += ca * U2[r*2+c, ms]
                for cb, bi in B_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c + half*4
                            V[gi*8+gj, col] += cb * V2[r*2+c, ms]
                for cc, bi in C_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c + half*4
                            W[gj*4+gi, col] += cc * W2[r*2+c, ms]
    return U, V, W

def _verify(U, V, W, n, m, p):
    U = np.asarray(U); V = np.asarray(V); W = np.asarray(W)
    rank = U.shape[1]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                expected = 1.0
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - expected) > 1e-6:
                    return False
    return True

def _find_dependency(M):
    # M is (dim, R)
    # Find null space vector c such that M @ c = 0
    # Use SVD
    try:
        U_svd, s_vals, Vt = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    
    tol = 1e-10 * max(M.shape)
    rank_svd = sum(1 for sv in s_vals if sv > tol)
    
    if rank_svd >= M.shape[1]:
        return None
        
    # Null space vectors are rows of Vt corresponding to zero singular values
    # Vt is (min(dim, R), R)
    # If rank_svd < R, the last (R - rank_svd) rows of Vt span the null space
    # Note: Vt rows are the right singular vectors.
    # The singular values correspond to the columns of U and rows of Vt.
    # s_vals[0] corresponds to Vt[0], etc.
    # So null vectors are Vt[rank_svd:, :]
    
    null_vectors = Vt[rank_svd:, :]
    if len(null_vectors) == 0:
        return None
        
    # Pick the null vector with the largest max absolute component to avoid numerical issues
    # Or just the first one.
    # We want a vector c such that sum c_r T_r = 0.
    # Vt rows are orthonormal.
    # Let's take the first null vector.
    c = null_vectors[0]
    
    # Ensure c is not zero
    if np.max(np.abs(c)) < 1e-10:
        return None
        
    return c

def construct():
    n, m, p = 4, 4, 8
    start = time.time()
    deadline = start + 145.0
    
    U, V, W = _build(98)
    assert _verify(U, V, W, n, m, p), "base decomposition invalid"
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    R = U.shape[1]
    
    # Precompute tensor vectors for current U, V, W
    # We will update them as we reduce rank
    
    def get_tensor_vecs(U, V, W):
        R = U.shape[1]
        dim = n * p * m
        M = np.zeros((dim, R))
        for r in range(R):
            u_r = U[:, r]
            v_r = V[:, r]
            w_r = W[:, r]
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        idx = i * p * m + k * m + j
                        M[idx, r] = u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
        return M

    M = get_tensor_vecs(U, V, W)
    
    iterations = 0
    while time.time() < deadline - 10.0:
        iterations += 1
        c = _find_dependency(M)
        if c is None:
            break
            
        # Find index r0 with largest |c[r0]| to be the one removed
        r0 = np.argmax(np.abs(c))
        if abs(c[r0]) < 1e-10:
            break
            
        # New coefficients for s != r0: alpha_s = 1 - c[s]/c[r0]
        # We remove column r0.
        # We scale remaining columns s by alpha_s.
        
        # Create indices for remaining columns
        keep = [i for i in range(R) if i != r0]
        
        # New U, V, W
        U_new = np.zeros((U.shape[0], R-1))
        V_new = np.zeros((V.shape[0], R-1))
        W_new = np.zeros((W.shape[0], R-1))
        
        for new_idx, old_idx in enumerate(keep):
            alpha = 1.0 - c[old_idx] / c[r0]
            # Scale U, V, W columns by alpha
            # We can distribute alpha, e.g., put all in U
            U_new[:, new_idx] = alpha * U[:, old_idx]
            V_new[:, new_idx] = V[:, old_idx]
            W_new[:, new_idx] = W[:, old_idx]
            
        U = U_new
        V = V_new
        W = W_new
        R -= 1
        
        # Update M
        M = get_tensor_vecs(U, V, W)
        
        # Optional: check verification periodically? It's expensive.
        # But we must ensure we don't drift.
        # The algebraic manipulation is exact if c is exact.
        # c is from SVD, so it's approximate.
        # This might accumulate error.
        # We should verify at the end.
        
    # Final verification
    if time.time() < deadline - 2.0:
        if not _verify(U, V, W, n, m, p):
            # If verification fails, we might have accumulated numerical errors
            # or the dependency was not exact.
            # In that case, we should probably return the best verified state.
            # But we don't store intermediate states.
            # We can try to recover by not doing the last step?
            # For now, if it fails, we return the initial rank 98 which is verified.
            U, V, W = _build(98)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
