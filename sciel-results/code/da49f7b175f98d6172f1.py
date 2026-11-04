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
    
    # Build tensor vector matrix: each column r gives the tensor T_r flattened
    # T_r[i,k,j] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Flatten index: i*p*m + k*m + j  (length 128)
    dim = n * p * m  # 128
    M = np.zeros((dim, R))  # columns are tensor vectors
    
    for r in range(R):
        u_r = U[:, r]
        v_r = V[:, r]
        w_r = W[:, r]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    idx = i * p * m + k * m + j
                    M[idx, r] = u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    
    # Find linear dependencies among the R tensor vectors
    # Use SVD to find the null space of M (dim x R)
    # If rank(M) < R, there are dependencies
    
    removed_set = set()
    
    # Iteratively find and remove dependent vectors
    # Strategy: compute SVD of M, find null space vectors, 
    # identify which original indices have significant weight in null vectors
    
    active = list(range(R))
    
    while time.time() < deadline - 10.0:
        if not active:
            break
        
        # Build submatrix of active columns
        M_active = M[:, active]
        n_active = len(active)
        
        if n_active <= 1:
            break
        
        # Compute SVD
        try:
            U_svd, s_vals, Vt = np.linalg.svd(M_active, full_matrices=False)
        except np.linalg.LinAlgError:
            break
        
        tol = 1e-10 * max(M_active.shape)
        rank_svd = sum(1 for sv in s_vals if sv > tol)
        
        if rank_svd >= n_active:
            # No dependencies, we're done
            break
        
        # Find null space: right singular vectors with singular value <= tol
        # Vt has shape (min(dim, n_active), n_active)
        # Null space vectors are rows of Vt corresponding to zero singular values
        null_start = rank_svd
        if null_start >= Vt.shape[0]:
            # All singular values are nonzero but rank < n_active means 
            # we need to look at the full Vt
            # Actually Vt is (min(dim,R), n_active), so if rank_svd < n_active,
            # the last (n_active - rank_svd) rows of Vt span the null space
            null_vectors = Vt[null_start:, :]
        else:
            null_vectors = Vt[null_start:rank_svd, :] if null_start < rank_svd else np.array([])
        
        if len(null_vectors) == 0:
            # Try a slightly larger tolerance
            tol2 = 1e-8 * max(M_active.shape)
            rank_svd2 = sum(1 for sv in s_vals if sv > tol2)
            if rank_svd2 < n_active:
                null_vectors = Vt[rank_svd2:, :]
            else:
                break
        
        if len(null_vectors) == 0:
            break
        
        # For each null vector, find the index with the largest absolute weight
        # and remove it (it's in the span of others)
        found_removal = False
        for nv in null_vectors:
            # nv is a vector of length n_active
            # Find the index with max |nv[i]|
            max_idx = np.argmax(np.abs(nv))
            orig_idx = active[max_idx]
            removed_set.add(orig_idx)
            active.remove(orig_idx)
            found_removal = True
            break
        
        if not found_removal:
            break
    
    # Construct final matrices
    if removed_set:
        keep = [i for i in range(R) if i not in removed_set]
        U_final = U[:, keep]
        V_final = V[:, keep]
        W_final = W[:, keep]
    else:
        U_final = U
        V_final = V
        W_final = W
    
    # Final verification
    if time.time() < deadline - 2.0:
        if not _verify(U_final, V_final, W_final, n, m, p):
            # Fallback to original
            U_final = U
            V_final = V
            W_final = W
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
