# EVOLVE-BLOCK-START
import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 iff C[i,k] uses A[i,j]*B[j,k]."""
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _als_step_U(T, V, W, n, m, p, r):
    """Solve for U given V, W."""
    U = np.zeros((n, m, r), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            # T[i,j,k] = sum_s U[i,j,s] * V[j,k,s] * W[k,i,s]
            # For fixed i,j: vector over k: T[i,j,:] = U[i,j,:] * diag(V[j,:,s]*W[:,i,s]) summed over s
            # Build linear system: sum_s U[i,j,s] * (V[j,k,s] * W[k,i,s]) = T[i,j,k] for all k
            # This is: (V[j,:,s] * W[:,i,s]) dot U[i,j,:] = T[i,j,:]
            # Shape: (p, r) matrix times (r,) vector = (p,) vector
            A = V[j, :, :] * W[:, i, :]  # (p, r)
            b = T[i, j, :]  # (p,)
            # Solve A @ x = b in least squares
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            U[i, j, :] = x
    return U

def _als_step_V(T, U, W, n, m, p, r):
    """Solve for V given U, W."""
    V = np.zeros((m, p, r), dtype=np.float64)
    for j in range(m):
        for k in range(p):
            # T[i,j,k] = sum_s U[i,j,s] * V[j,k,s] * W[k,i,s]
            # For fixed j,k: T[:,j,k] = sum_s V[j,k,s] * (U[:,j,s] * W[k,:]) 
            # vector over i: T[:,j,k] = V[j,k,:] * (U[:,j,:] * W[k,:])^T ... 
            # sum_s V[j,k,s] * U[i,j,s] * W[k,i] = T[i,j,k]
            # For fixed j,k: (U[:,j,:] * W[k,:]) @ V[j,k,:] = T[:,j,k]
            A = U[:, j, :] * W[k, :]  # (n, r)
            b = T[:, j, k]  # (n,)
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            V[j, k, :] = x
    return V

def _als_step_W(T, U, V, n, m, p, r):
    """Solve for W given U, V."""
    W = np.zeros((p, n, r), dtype=np.float64)
    for k in range(p):
        for i in range(n):
            # T[i,j,k] = sum_s U[i,j,s] * V[j,k,s] * W[k,i,s]
            # For fixed k,i: T[i,:,k] = sum_s W[k,i,s] * (U[i,:]*V[:,k,s])
            # sum_s W[k,i,s] * U[i,j,s] * V[j,k,s] = T[i,j,k]
            # For fixed k,i: (U[i,:]*V[:,k,:]) @ W[k,i,:] = T[i,:,k]
            A = U[i, :, :] * V[:, k, :]  # (m, r)
            b = T[i, :, k]  # (m,)
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            W[k, i, :] = x
    return W

def _residual(T, U, V, W):
    """Compute ||sum_s U[i,j,s]*V[j,k,s]*W[k,i,s] - T[i,j,k]||_F"""
    result = np.einsum('ijs,jks,ksi->ijk', U, V, W)
    return np.linalg.norm(result - T)

def _strassen_2x2_factors():
    """Return the 7 rank-1 factors for 2x2*2x2."""
    factors = []
    u = [1, 0, 0, 1]; v = [1, 0, 0, 1]; w = [1, 0, 0, 1]
    factors.append((u, v, w))
    u = [0, 0, 1, 1]; v = [1, 0, 0, 0]; w = [0, 0, 1, -1]
    factors.append((u, v, w))
    u = [1, 0, 0, 0]; v = [0, 1, 0, -1]; w = [0, 1, 0, 1]
    factors.append((u, v, w))
    u = [0, 0, 0, 1]; v = [-1, 0, 1, 0]; w = [1, 0, 1, 0]
    factors.append((u, v, w))
    u = [1, 1, 0, 0]; v = [0, 0, 0, 1]; w = [-1, 1, 0, 0]
    factors.append((u, v, w))
    u = [-1, 0, 1, 0]; v = [1, 1, 0, 0]; w = [0, 0, 0, 1]
    factors.append((u, v, w))
    u = [0, 1, 0, -1]; v = [0, 0, 1, 1]; w = [1, 0, 0, 0]
    factors.append((u, v, w))
    return factors

def _strassen_4x4_4x8():
    """Construct rank-98 Strassen decomposition."""
    n, m, p = DIMENSIONS
    s2 = _strassen_2x2_factors()
    rank = 98
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for t in range(2):
        for s in range(7):
            for r in range(7):
                col = t * 49 + s * 7 + r
                su = s2[s][0]; sv = s2[s][1]; sw = s2[s][2]
                ru = s2[r][0]; rv = s2[r][1]; rw = s2[r][2]
                for bi in range(2):
                    for bj in range(2):
                        cb = su[bi*2+bj]
                        if cb == 0: continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = ru[ei*2+ej]
                                if cs == 0: continue
                                i = 2*bi+ei; j = 2*bj+ej
                                U[i*m+j, col] += cb * cs
                for bi in range(2):
                    for bj in range(2):
                        cb = sv[bi*2+bj]
                        if cb == 0: continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = rv[ei*2+ej]
                                if cs == 0: continue
                                j = 2*bi+ei
                                k = 4*t+2*bj+ej
                                V[j*p+k, col] += cb * cs
                for ki in range(2):
                    for ii in range(2):
                        cb = sw[ki*2+ii]
                        if cb == 0: continue
                        for ek in range(2):
                            for ei in range(2):
                                cs = rw[ek*2+ei]
                                if cs == 0: continue
                                k = 4*t+2*ki+ek
                                i = 2*ii+ei
                                W[k*n+i, col] += cb * cs
    return U, V, W

def _verify(U, V, W, n=4, m=4, p=8, atol=1e-9):
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    r = U3.shape[2]
    result = np.einsum('ijs,jks,ksi->ijk', U3, V3, W3)
    T = np.ones((n, m, p))
    return np.allclose(result, T, atol=atol)

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _try_als(n, m, p, r, max_iter=200, n_restarts=5, time_limit=120):
    """Try ALS to find rank-r decomposition."""
    T = _build_tensor(n, m, p)
    start = time.time()
    best_residual = float('inf')
    best = None
    
    for restart in range(n_restarts):
        if time.time() - start > time_limit - 10:
            break
        rng = np.random.RandomState(42 + restart)
        U = rng.randn(n, m, r)
        V = rng.randn(m, p, r)
        W = rng.randn(p, n, r)
        # Normalize
        for s in range(r):
            nrm = np.linalg.norm(U[:,:,s]) * np.linalg.norm(V[:,:,s]) * np.linalg.norm(W[:,:,s])
            if nrm > 1e-10:
                U[:,:,s] /= nrm**(1/3)
                V[:,:,s] /= nrm**(1/3)
                W[:,:,s] /= nrm**(1/3)
        
        for it in range(max_iter):
            if time.time() - start > time_limit - 10:
                break
            U = _als_step_U(T, V, W, n, m, p, r)
            V = _als_step_V(T, U, W, n, m, p, r)
            W = _als_step_W(T, U, V, n, m, p, r)
            res = _residual(T, U, V, W)
            if res < best_residual:
                best_residual = res
                best = (U.copy(), V.copy(), W.copy())
            if res < 1e-9:
                break
        if best_residual < 1e-9:
            break
    
    if best is None:
        return None
    U, V, W = best
    if _verify(U.reshape(n*m, -1), V.reshape(m*p, -1), W.reshape(p*n, -1)):
        return U.reshape(n*m, -1), V.reshape(m*p, -1), W.reshape(p*n, -1)
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Try ALS for rank 56
    try:
        result = _try_als(n, m, p, r=56, max_iter=300, n_restarts=10, time_limit=100)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Try ALS for rank 64
    if time.time() < deadline - 20:
        try:
            result = _try_als(n, m, p, r=64, max_iter=300, n_restarts=5, time_limit=60)
            if result is not None:
                U, V, W = result
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except Exception:
            pass
    
    # Fall back to Strassen rank-98
    try:
        U, V, W = _strassen_4x4_4x8()
        if _verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Final fallback: schoolbook
    U, V, W = _schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
