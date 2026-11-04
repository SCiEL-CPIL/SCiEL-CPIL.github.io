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
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def construct():
    n, m, p = 4, 4, 8
    start = time.time()
    deadline = start + 145.0
    
    U, V, W = _build(98)
    assert _verify(U, V, W, n, m, p), "base decomposition invalid"
    
    U = U.copy(); V = V.copy(); W = W.copy()
    R = U.shape[1]
    
    # Build tensor matrix efficiently using einsum
    dim = n * p * m  # 128
    # M[idx, r] where idx = i*p*m + k*m + j
    # M[i*p*m + k*m + j, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape: U is (n*m, R), V is (m*p, R), W is (p*n, R)
    # Let's use broadcasting
    # U reshaped to (n, m, R), V to (m, p, R), W to (p, n, R)
    U_r = U.reshape(n, m, R)  # U_r[i,j,r]
    V_r = V.reshape(m, p, R)  # V_r[j,k,r]
    W_r = W.reshape(p, n, R)  # W_r[k,i,r]
    
    # M[i,k,j,r] = U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
    # Use einsum: 'ijr,jkr,kir->ikjr'
    M_4d = np.einsum('ijr,jkr,kir->ikjr', U_r, V_r, W_r, optimize=True)
    M = M_4d.reshape(dim, R)
    
    # Find null space of M (dim x R matrix)
    # We want to find the maximum set of linearly independent columns
    # Use SVD
    try:
        U_svd, s_vals, Vt = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    tol = 1e-10 * max(M.shape)
    rank_svd = sum(1 for sv in s_vals if sv > tol)
    
    if rank_svd >= R:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Null space: last (R - rank_svd) rows of Vt
    # Vt has shape (min(dim,R), R) = (128, 98) or (98, 98)
    # Since dim=128 > R=98, Vt is (98, 98)
    # rank_svd singular values, then (R - rank_svd) null vectors
    null_start = rank_svd
    if null_start >= Vt.shape[0]:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    null_vecs = Vt[null_start:, :]  # shape (R-rank_svd, R)
    
    # Greedy removal: for each null vector, remove the column with max weight
    removed_set = set()
    active = list(range(R))
    
    for nv in null_vecs:
        if time.time() > deadline - 10.0:
            break
        # Find max weight index among active
        nv_active = np.array([nv[i] for i in active])
        max_local = np.argmax(np.abs(nv_active))
        orig_idx = active[max_local]
        removed_set.add(orig_idx)
        active.remove(orig_idx)
    
    if removed_set:
        keep = [i for i in range(R) if i not in removed_set]
        U_final = U[:, keep]
        V_final = V[:, keep]
        W_final = W[:, keep]
    else:
        U_final = U; V_final = V; W_final = W
    
    if time.time() < deadline - 2.0:
        if not _verify(U_final, V_final, W_final, n, m, p):
            U_final = U; V_final = V; W_final = W
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
