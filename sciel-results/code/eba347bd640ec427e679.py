"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 5]
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin for the 150s limit

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=2, m=2, p=5):
    """
    Verify if the given U, V, W correctly decompose the matrix multiplication tensor.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                computed = 0.0
                for r in range(rank):
                    computed += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(computed - target) > 1e-9:
                    return False
    return True

def _construct_rank18(n=2, m=2, p=5):
    """
    Construct a rank-18 decomposition using Strassen for 2x2x2 (cols 0-1),
    Strassen for 2x2x2 (cols 2-3), and Schoolbook for 2x2x1 (col 4).
    """
    rank = 18
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    def add_strassen_block(col0, col1, base):
        r = base
        U[0][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1
        W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
        
        r = base + 2
        U[0][r] = 1
        V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
        W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 3
        U[3][r] = 1
        V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
        W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
        
        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[1*p+col1][r] = 1
        W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
        
        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
        W[col1*n+1][r] = 1
        
        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1
    
    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    # Schoolbook for column 4
    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    return None

def _construct_schoolbook(n=2, m=2, p=5):
    """
    Construct the standard schoolbook decomposition of rank n*m*p.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _als_refine(U, V, W, T, R, n, m, p, max_iter=500, tol=1e-15):
    """Run ALS iterations to refine the decomposition."""
    prev_err = np.inf
    for it in range(max_iter):
        # Solve for U
        VvW = np.einsum('vr,wr->vwr', V, W)
        VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
        for u in range(n*m):
            b = T[u].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(VvW_flat, b, rcond=None)
            U[u] = sol
        
        # Solve for V
        Uw = np.einsum('ur,wr->uwr', U, W)
        Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
        for v in range(m*p):
            b = T[:, v, :].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(Uw_flat, b, rcond=None)
            V[v] = sol
        
        # Solve for W
        UV = np.einsum('ur,vr->uvr', U, V)
        UV_flat = UV.reshape(-1, R)
        for w in range(p*n):
            b = T[:, :, w].reshape(-1)
            sol, res, rank_s, sv = np.linalg.lstsq(UV_flat, b, rcond=None)
            W[w] = sol
        
        # Compute error
        err = 0.0
        for r in range(R):
            err += np.sum((np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r]) - T)**2)
        err = np.sqrt(err / (n*m * m*p * p*n))
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return U, V, W, err

def _check_and_rationalize(U, V, W, T, R, n, m, p):
    """Check if decomposition is exact and try to rationalize entries."""
    result = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r])
    
    if not np.allclose(result, T, atol=1e-8):
        return None
    
    # Try to rationalize with various denominators
    for denom in [2, 4, 8, 16, 32, 64]:
        U_s = np.round(U * denom) / denom
        V_s = np.round(V * denom) / denom
        W_s = np.round(W * denom) / denom
        
        result_s = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(R):
            result_s += np.einsum('u,v,w->uvw', U_s[:,r], V_s[:,r], W_s[:,r])
        
        if np.allclose(result_s, T, atol=1e-7):
            U_list = [[float(x) for x in row] for row in U_s]
            V_list = [[float(x) for x in row] for row in V_s]
            W_list = [[float(x) for x in row] for row in W_s]
            
            for mat in [U_list, V_list, W_list]:
                for i in range(len(mat)):
                    for j in range(len(mat[i])):
                        if abs(mat[i][j] - round(mat[i][j])) < 1e-9:
                            mat[i][j] = int(round(mat[i][j]))
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return as floats if valid
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _get_strassen_block_np(n=2, m=2, p=5, col0=0, col1=1):
    """Get Strassen block for two columns as numpy arrays."""
    U = np.zeros((n*m, 7))
    V = np.zeros((m*p, 7))
    W = np.zeros((p*n, 7))
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0*p+col0, 0] = 1; V[1*p+col1, 0] = 1
    W[col0*n+0, 0] = 1; W[col1*n+1, 0] = 1
    
    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0*p+col0, 1] = 1
    W[col0*n+1, 1] = 1; W[col1*n+1, 1] = -1
    
    # M3: A00(B01-B11)
    U[0, 2] = 1
    V[0*p+col1, 2] = 1; V[1*p+col1, 2] = -1
    W[col1*n+0, 2] = 1; W[col1*n+1, 2] = 1
    
    # M4: A11(B10-B00)
    U[3, 3] = 1
    V[1*p+col0, 3] = 1; V[0*p+col0, 3] = -1
    W[col0*n+0, 3] = 1; W[col0*n+1, 3] = 1
    
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[1*p+col1, 4] = 1
    W[col0*n+0, 4] = -1; W[col1*n+0, 4] = 1
    
    # M6: (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0*p+col0, 5] = 1; V[0*p+col1, 5] = 1
    W[col1*n+1, 5] = 1
    
    # M7: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1*p+col0, 6] = 1; V[1*p+col1, 6] = 1
    W[col0*n+0, 6] = 1
    
    return U, V, W

def _search_rank16(n=2, m=2, p=5, timeout=120):
    """
    Search for a rank-16 decomposition.
    Uses multiple initialization strategies: Strassen + random, structured patterns.
    """
    start_time = time.time()
    
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1.0
    
    R = 16
    R_strassen = 7
    
    for trial in range(100):
        if time.time() - start_time > timeout - 5:
            break
        
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        strategy = trial % 4
        
        if strategy == 0:
            # Strassen for cols 0,1 + random
            U_str, V_str, W_str = _get_strassen_block_np(n, m, p, 0, 1)
            U[:, :R_strassen] = U_str
            V[:, :R_strassen] = V_str
            W[:, :R_strassen] = W_str
            noise_scale = 0.3 + 0.1 * (trial % 7)
            U[:, R_strassen:] = np.random.randn(n*m, R - R_strassen) * noise_scale
            V[:, R_strassen:] = np.random.randn(m*p, R - R_strassen) * noise_scale
            W[:, R_strassen:] = np.random.randn(p*n, R - R_strassen) * noise_scale
        elif strategy == 1:
            # Strassen for cols 0,2 + random
            U_str, V_str, W_str = _get_strassen_block_np(n, m, p, 0, 2)
            U[:, :R_strassen] = U_str
            V[:, :R_strassen] = V_str
            W[:, :R_strassen] = W_str
            noise_scale = 0.3 + 0.1 * (trial % 7)
            U[:, R_strassen:] = np.random.randn(n*m, R - R_strassen) * noise_scale
            V[:, R_strassen:] = np.random.randn(m*p, R - R_strassen) * noise_scale
            W[:, R_strassen:] = np.random.randn(p*n, R - R_strassen) * noise_scale
        elif strategy == 2:
            # All random with moderate scale
            noise_scale = 0.2 + 0.1 * (trial % 10)
            U[:, :] = np.random.randn(n*m, R) * noise_scale
            V[:, :] = np.random.randn(m*p, R) * noise_scale
            W[:, :] = np.random.randn(p*n, R) * noise_scale
        else:
            # Structured: use schoolbook-like patterns for some columns
            # First 4 columns: schoolbook for (i,j,k) in [(0,0,0),(0,0,1),(0,1,0),(0,1,1)]
            for idx, (ii, jj, kk) in enumerate([(0,0,0),(0,0,1),(0,1,0),(0,1,1)]):
                if idx < R:
                    U[ii*m+jj, idx] = 1
                    V[jj*p+kk, idx] = 1
                    W[kk*n+ii, idx] = 1
            # Remaining: random
            for r in range(4, R):
                U[:, r] = np.random.randn(n*m) * 0.3
                V[:, r] = np.random.randn(m*p) * 0.3
                W[:, r] = np.random.randn(p*n) * 0.3
        
        U, V, W, err = _als_refine(U, V, W, T, R, n, m, p, max_iter=300)
        
        result = _check_and_rationalize(U, V, W, T, R, n, m, p)
        if result is not None:
            return result
    
    return None

def _search_rank17(n=2, m=2, p=5, timeout=120):
    """
    Search for a rank-17 decomposition.
    Uses multiple initialization strategies.
    """
    start_time = time.time()
    
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1.0
    
    R = 17
    R_strassen = 7
    
    for trial in range(80):
        if time.time() - start_time > timeout - 5:
            break
        
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        strategy = trial % 4
        
        if strategy == 0:
            # Strassen for cols 0,1 + random
            U_str, V_str, W_str = _get_strassen_block_np(n, m, p, 0, 1)
            U[:, :R_strassen] = U_str
            V[:, :R_strassen] = V_str
            W[:, :R_strassen] = W_str
            noise_scale = 0.3 + 0.1 * (trial % 7)
            U[:, R_strassen:] = np.random.randn(n*m, R - R_strassen) * noise_scale
            V[:, R_strassen:] = np.random.randn(m*p, R - R_strassen) * noise_scale
            W[:, R_strassen:] = np.random.randn(p*n, R - R_strassen) * noise_scale
        elif strategy == 1:
            # Strassen for cols 2,3 + random
            U_str, V_str, W_str = _get_strassen_block_np(n, m, p, 2, 3)
            U[:, :R_strassen] = U_str
            V[:, :R_strassen] = V_str
            W[:, :R_strassen] = W_str
            noise_scale = 0.3 + 0.1 * (trial % 7)
            U[:, R_strassen:] = np.random.randn(n*m, R - R_strassen) * noise_scale
            V[:, R_strassen:] = np.random.randn(m*p, R - R_strassen) * noise_scale
            W[:, R_strassen:] = np.random.randn(p*n, R - R_strassen) * noise_scale
        elif strategy == 2:
            # All random
            noise_scale = 0.2 + 0.1 * (trial % 10)
            U[:, :] = np.random.randn(n*m, R) * noise_scale
            V[:, :] = np.random.randn(m*p, R) * noise_scale
            W[:, :] = np.random.randn(p*n, R) * noise_scale
        else:
            # Structured: schoolbook for 5 columns + random
            for idx, (ii, jj, kk) in enumerate([(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0)]):
                if idx < R:
                    U[ii*m+jj, idx] = 1
                    V[jj*p+kk, idx] = 1
                    W[kk*n+ii, idx] = 1
            for r in range(5, R):
                U[:, r] = np.random.randn(n*m) * 0.3
                V[:, r] = np.random.randn(m*p) * 0.3
                W[:, r] = np.random.randn(p*n) * 0.3
        
        U, V, W, err = _als_refine(U, V, W, T, R, n, m, p, max_iter=300)
        
        result = _check_and_rationalize(U, V, W, T, R, n, m, p)
        if result is not None:
            return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT
    
    # Try rank 16 search with more trials
    if time.time() < deadline - 60:
        result = _search_rank16(n, m, p, timeout=min(35, deadline - time.time() - 60))
        if result is not None:
            return result
    
    # Try rank 17 search
    if time.time() < deadline - 25:
        result = _search_rank17(n, m, p, timeout=min(30, deadline - time.time() - 25))
        if result is not None:
            return result
    
    # Try rank 18 (constructive)
    result = _construct_rank18(n, m, p)
    if result is not None:
        return result
    
    # Fallback: Schoolbook (rank 20)
    return _construct_schoolbook(n, m, p)

# EVOLVE-BLOCK-END
