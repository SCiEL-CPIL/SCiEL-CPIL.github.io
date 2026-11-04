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

def _build_base():
    n, m, p = 4, 4, 8
    rank = 98
    U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
    U2, V2, W2 = _strassen_2x2x2()
    
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

def _verify(U, V, W, n, m, p, tol=1e-6):
    rank = U.shape[1]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """Try to reduce rank by finding integer linear dependencies."""
    R = U.shape[1]
    
    # Build the tensor matrix: for each (i,j,k), the vector of products
    # T[i,j,k,r] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # We need sum_r T[i,j,k,r] = 1 for all (i,j,k)
    
    def get_tensor(U, V, W):
        n, m, p = 4, 4, 8
        R = U.shape[1]
        T = np.zeros((n*m*p, R))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(R):
                        T[idx, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    idx += 1
        return T
    
    def verify_full(U, V, W):
        return _verify(U, V, W, n, m, p)
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = R
    
    # Try to find a column that can be eliminated
    # For each column r0, check if there exist coefficients c_r such that
    # sum_{r != r0} c_r * (U[:,r]*V[:,r]*W[:,r]) = -1 * (U[:,r0]*V[:,r0]*W[:,r0])
    # This is a linear system in the c_r's
    
    T = get_tensor(U, V, W)
    
    for r0 in range(R):
        if time.time() > deadline - 10:
            break
        
        # Try to express column r0 as combination of others
        # We need: sum_{r != r0} c_r * T[:,r] = -T[:,r0]
        # This is underdetermined (128 equations, R-1 unknowns)
        
        # Use least squares to find c
        T_other = T[:, [r for r in range(R) if r != r0]]
        target = -T[:, r0]
        
        try:
            c, residuals, rank_ls, sv = np.linalg.lstsq(T_other, target, rcond=1e-10)
        except:
            continue
        
        # Check if the solution is exact
        residual = np.linalg.norm(T_other @ c + T[:, r0])
        if residual > 1e-5:
            continue
        
        # Check if c has small integer values (within tolerance)
        # If c_r are integers, we can scale and eliminate
        c_rounded = np.round(c)
        if np.linalg.norm(c - c_rounded) < 1e-3:
            # Check if we can scale to make it work
            # New decomposition: remove r0, add c_r to column r
            new_U = U.copy(); new_V = V.copy(); new_W = W.copy()
            keep = [r for r in range(R) if r != r0]
            for idx, r in enumerate(keep):
                new_U[:, r] += c_rounded[idx] * U[:, r0]
                new_V[:, r] += c_rounded[idx] * V[:, r0]
                new_W[:, r] += c_rounded[idx] * W[:, r0]
            new_U = new_U[:, keep]
            new_V = new_V[:, keep]
            new_W = new_W[:, keep]
            
            if verify_full(new_U, new_V, new_W):
                best_U, best_V, best_W = new_U, new_V, new_W
                best_rank = R - 1
                U, V, W = new_U, new_V, new_W
                R = best_rank
                T = get_tensor(U, V, W)
    
    return best_U, best_V, best_W

def construct():
    n, m, p = 4, 4, 8
    start = time.time()
    deadline = start + 130.0
    
    U, V, W = _build_base()
    assert _verify(U, V, W, n, m, p), "base decomposition invalid"
    
    # Try to reduce rank
    if time.time() < deadline - 20:
        U, V, W = _try_reduce_rank(U, V, W, n, m, p, deadline)
    
    # Final verification
    if not _verify(U, V, W, n, m, p):
        U, V, W = _build_base()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
