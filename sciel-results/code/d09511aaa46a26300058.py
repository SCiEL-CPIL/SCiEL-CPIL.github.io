import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _build_rank33_decomposition():
    """
    Constructs a rank-33 decomposition for 2x3 by 3x6 matrix multiplication.
    Decomposes B into three 3x2 blocks, using rank-11 for each 2x3x2 multiplication.
    """
    n, m, p = 2, 3, 6
    rank = 33
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # 11 terms for a 2x3 x 3x2 multiplication block.
    terms = []
    
    # Strassen 7 terms for 2x2 x 2x2 (A[:,0:2] * B[0:2,:])
    terms.append(({0:1, 4:1}, {0:1, 3:1}, {0:1, 3:1}))
    terms.append(({3:1, 4:1}, {0:1}, {2:1, 3:-1}))
    terms.append(({0:1}, {1:1, 3:-1}, {1:1, 3:1}))
    terms.append(({4:1}, {2:1, 0:-1}, {0:1, 2:1}))
    terms.append(({0:1, 1:1}, {3:1}, {0:-1, 1:1}))
    terms.append(({3:1, 0:-1}, {0:1, 1:1}, {3:1}))
    terms.append(({1:1, 4:-1}, {2:1, 3:1}, {0:1}))
    
    # 4 scalar terms for A[:,2] * B[2,:]
    terms.append(({2:1}, {4:1}, {0:1}))
    terms.append(({2:1}, {5:1}, {1:1}))
    terms.append(({5:1}, {4:1}, {2:1}))
    terms.append(({5:1}, {5:1}, {3:1}))
    
    for block in range(3):
        for t_idx, (u_dict, v_dict, w_dict) in enumerate(terms):
            col = block * 11 + t_idx
            for idx, val in u_dict.items():
                U[idx, col] = val
            for l_idx, val in v_dict.items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx, col] = val
            for l_idx, val in w_dict.items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx, col] = val
                
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify exact decomposition of the matrix multiplication tensor."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_find_rank13_2x3x3(max_time=30):
    """Try to find a rank-13 decomposition for 2x3 x 3x3 with integer/half-integer entries."""
    start_time = time.time()
    n, m, p = 2, 3, 3
    rank = 13
    dim_u = n * m  # 6
    dim_v = m * p  # 9
    dim_w = p * n  # 6
    
    def residuals(params):
        U = params[:dim_u*rank].reshape(dim_u, rank)
        V = params[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
        W = params[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    res.append(s - 1.0)
        return np.array(res)
    
    best_solution = None
    best_cost = np.inf
    
    for seed in range(300):
        if time.time() - start_time > max_time:
            break
        
        np.random.seed(seed)
        # Initialize with a structured guess based on Strassen + scalar terms
        x0 = np.zeros(dim_u*rank + dim_v*rank + dim_w*rank)
        
        # Place Strassen terms in first 7 columns
        strassen_U = np.zeros((dim_u, 7))
        strassen_V = np.zeros((dim_v, 7))
        strassen_W = np.zeros((dim_w, 7))
        
        # M1
        strassen_U[0, 0] = 1; strassen_U[4, 0] = 1
        strassen_V[0, 0] = 1; strassen_V[4, 0] = 1
        strassen_W[0, 0] = 1; strassen_W[4, 0] = 1
        # M2
        strassen_U[3, 1] = 1; strassen_U[4, 1] = 1
        strassen_V[0, 1] = 1
        strassen_W[3, 1] = 1; strassen_W[4, 1] = -1
        # M3
        strassen_U[0, 2] = 1
        strassen_V[1, 2] = 1; strassen_V[4, 2] = -1
        strassen_W[1, 2] = 1; strassen_W[4, 2] = 1
        # M4
        strassen_U[4, 3] = 1
        strassen_V[3, 3] = 1; strassen_V[0, 3] = -1
        strassen_W[0, 3] = 1; strassen_W[3, 3] = 1
        # M5
        strassen_U[0, 4] = 1; strassen_U[1, 4] = 1
        strassen_V[4, 4] = 1
        strassen_W[0, 4] = -1; strassen_W[1, 4] = 1
        # M6
        strassen_U[3, 5] = 1; strassen_U[0, 5] = -1
        strassen_V[0, 5] = 1; strassen_V[1, 5] = 1
        strassen_W[4, 5] = 1
        # M7
        strassen_U[1, 6] = 1; strassen_U[4, 6] = -1
        strassen_V[3, 6] = 1; strassen_V[4, 6] = 1
        strassen_W[0, 6] = 1
        
        x0[:dim_u*7] = strassen_U.flatten()
        x0[dim_u*7:dim_u*7+dim_v*7] = strassen_V.flatten()
        x0[dim_u*7+dim_v*7:dim_u*7+dim_v*7+dim_w*7] = strassen_W.flatten()
        
        # Add small random perturbation
        x0 += np.random.randn(len(x0)) * 0.3
        
        try:
            result = least_squares(residuals, x0, max_nfev=2000, ftol=1e-12)
            if result.cost < best_cost:
                best_cost = result.cost
                best_solution = result.x
                if best_cost < 1e-14:
                    break
        except:
            continue
    
    if best_solution is not None and best_cost < 1e-14:
        U = best_solution[:dim_u*rank].reshape(dim_u, rank)
        V = best_solution[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
        W = best_solution[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
        
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return U_r, V_r, W_r
    
    return None

def _build_rank26_decomposition(U13, V13, W13):
    """Build a rank-26 decomposition using two rank-13 2x3x3 blocks."""
    n, m, p = 2, 3, 6
    rank = 26
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for block in range(2):
        for c in range(13):
            col = block * 13 + c
            U[:, col] = U13[:, c]
            for j in range(m):
                for k in range(3):
                    V[j*p + (block*3 + k), col] = V13[j*3 + k, c]
            for k in range(3):
                for i in range(n):
                    W[(block*3 + k)*n + i, col] = W13[k*n + i, c]
    
    return U, V, W

def _try_find_rank10_2x3x2(max_time=20):
    """Try to find a rank-10 decomposition for 2x3 x 3x2 with integer/half-integer entries."""
    start_time = time.time()
    n, m, p = 2, 3, 2
    rank = 10
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    def residuals(params):
        U = params[:dim_u*rank].reshape(dim_u, rank)
        V = params[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
        W = params[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    res.append(s - 1.0)
        return np.array(res)
    
    best_solution = None
    best_cost = np.inf
    
    for seed in range(200):
        if time.time() - start_time > max_time:
            break
        
        np.random.seed(seed)
        x0 = np.random.randn(dim_u*rank + dim_v*rank + dim_w*rank) * 0.5
        try:
            result = least_squares(residuals, x0, max_nfev=1000)
            if result.cost < best_cost:
                best_cost = result.cost
                best_solution = result.x
                if best_cost < 1e-14:
                    break
        except:
            continue
    
    if best_solution is not None and best_cost < 1e-14:
        U = best_solution[:dim_u*rank].reshape(dim_u, rank)
        V = best_solution[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
        W = best_solution[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
        
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return U_r, V_r, W_r
    
    return None

def _build_rank30_decomposition(U10, V10, W10):
    """Build a rank-30 decomposition using rank-10 2x3x2 blocks."""
    n, m, p = 2, 3, 6
    rank = 30
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for block in range(3):
        for c in range(10):
            col = block * 10 + c
            U[:, col] = U10[:, c]
            for j in range(m):
                for k in range(2):
                    V[j*p + (block*2 + k), col] = V10[j*2 + k, c]
            for k in range(2):
                for i in range(n):
                    W[(block*2 + k)*n + i, col] = W10[k*n + i, c]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Try to find rank-13 decomposition for 2x3x3 (gives rank-26 total)
    if time.time() < deadline - 35:
        result = _try_find_rank13_2x3x3(max_time=30)
        
        if result is not None and time.time() < deadline:
            U13, V13, W13 = result
            U, V, W = _build_rank26_decomposition(U13, V13, W13)
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find rank-10 decomposition for 2x3x2 (gives rank-30 total)
    if time.time() < deadline - 20:
        result = _try_find_rank10_2x3x2(max_time=15)
        
        if result is not None and time.time() < deadline:
            U10, V10, W10 = result
            U, V, W = _build_rank30_decomposition(U10, V10, W10)
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 33
    U, V, W = _build_rank33_decomposition()
    
    if time.time() < deadline:
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
