import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 15]
TIME_LIMIT = 110

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _build_strassen_2x2x2():
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    
    # M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[1,1]=1; W[3,1]=-1
    
    # M3 = A00(B01-B11)
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[2,2]=1; W[3,2]=1
    
    # M4 = A11(B10-B00)
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[0,3]=1; W[1,3]=1
    
    # M5 = (A00+A01)B11
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1; W[2,4]=1
    
    # M6 = (A10-A00)(B00+B01)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[3,5]=1
    
    # M7 = (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1
    
    return U, V, W

def _build_rank53():
    n, m, p = 2, 2, 15
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
    
    # 7 blocks of 2, 1 block of 1
    # Indices k: 0-1 (strassen), 2-3 (strassen), 4-5 (strassen), 6-7 (strassen), 8-9 (strassen), 10-11 (strassen), 12-13 (strassen), 14 (schoolbook)
    total_rank = 7 * 7 + 4
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for b in range(7):
        start_k = b * 2
        U[:, col_offset:col_offset+7] = U_str
        for j in range(m):
            for k_l in range(2):
                V[j*p + start_k + k_l, col_offset:col_offset+7] = V_str[j*2 + k_l, :]
        for k_l in range(2):
            for i in range(n):
                W[(start_k + k_l)*n + i, col_offset:col_offset+7] = W_str[k_l*2 + i, :]
        col_offset += 7
        
    start_k = 14
    U[:, col_offset:col_offset+4] = U_sb
    for j in range(m):
        V[j*p + start_k, col_offset:col_offset+4] = V_sb[j*1 + 0, :]
    for i in range(n):
        W[start_k*n + i, col_offset:col_offset+4] = W_sb[0*2 + i, :]
        
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :])
                val = np.dot(val, W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _find_rank10_block_223(timeout=20):
    start = time.time()
    n, m, p = 2, 2, 3
    R = 10
    # Target tensor T[i*m+j, j*p+k, k*n+i] = 1
    # We flatten U, V, W into one vector x of size R*(n*m + m*p + p*n)
    size_U = n*m
    size_V = m*p
    size_W = p*n
    total_size = R * (size_U + size_V + size_W)
    
    def objective(x):
        if time.time() - start > timeout:
            return np.inf
        x = x.reshape(R, -1)
        U = x[:, :size_U].T # R x size_U -> transpose for convenience? No, U is size_U x R.
        # x is R x (size_U + size_V + size_W)
        # U_r is x[r, 0:size_U]
        # We need U (size_U x R), V (size_V x R), W (size_W x R)
        U = x[:, :size_U].T
        V = x[:, size_U:size_U+size_V].T
        W = x[:, size_U+size_V:].T
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    prod = U[i*m+j, :] * V[j*p+k, :]
                    s = np.dot(prod, W[k*n+i, :])
                    err += (s - 1.0)**2
        return err

    best_err = np.inf
    best_x = None
    
    # Try a few random restarts
    for _ in range(5):
        if time.time() - start > timeout - 2:
            break
        # Initialize with random values
        x0 = np.random.normal(0, 0.5, total_size)
        
        # We can also try initializing with a perturbed rank 11 solution
        # But random might work for small cases
        
        res = minimize(objective, x0, method='BFGS', options={'maxiter': 1000, 'disp': False})
        
        if res.success and res.fun < best_err:
            best_err = res.fun
            best_x = res.x
            
        if best_err < 1e-8:
            break

    if best_err < 1e-8:
        # Snap to half-integers
        x = best_x.reshape(R, -1)
        U = x[:, :size_U].T
        V = x[:, size_U:size_U+size_V].T
        W = x[:, size_U+size_V:].T
        
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        if _verify_decomposition(U, V, W, n, m, p):
            return U, V, W
            
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Baseline rank 53
    U53, V53, W53 = _build_rank53()
    if not _verify_decomposition(U53, V53, W53, n, m, p):
         # Fallback to schoolbook if 53 fails (shouldn't happen)
         U53, V53, W53 = _build_schoolbook(n, m, p)
         
    best_rank = 53
    best_U, best_V, best_W = U53, V53, W53
    
    # Try to find rank 50 solution
    if time.time() - start_time < TIME_LIMIT - 30:
        block_223 = _find_rank10_block_223(timeout=min(30, TIME_LIMIT - (time.time() - start_time) - 10))
        if block_223 is not None:
            U223, V223, W223 = block_223
            # Assemble 5 blocks
            total_rank = 5 * 10
            U_full = np.zeros((n*m, total_rank))
            V_full = np.zeros((m*p, total_rank))
            W_full = np.zeros((p*n, total_rank))
            
            for b in range(5):
                start_k = b * 3
                col_offset = b * 10
                U_full[:, col_offset:col_offset+10] = U223
                for j in range(m):
                    for k_l in range(3):
                        V_full[j*p + start_k + k_l, col_offset:col_offset+10] = V223[j*3 + k_l, :]
                for k_l in range(3):
                    for i in range(n):
                        W_full[(start_k + k_l)*n + i, col_offset:col_offset+10] = W223[k_l*2 + i, :]
                        
            if _verify_decomposition(U_full, V_full, W_full, n, m, p):
                best_rank = 50
                best_U, best_V, best_W = U_full, V_full, W_full
                
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
