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
    # A is 2x3 (flat: 0..5), B is 3x2 (flat: 0..5), C is 2x2 (flat: 0..3)
    # A: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1), 4:(2,0), 5:(2,1)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    terms = []
    
    # Strassen 7 terms for 2x2 x 2x2 (A[:,0:2] * B[0:2,:])
    # A sub: 0,1,3,4; B sub: 0,1,2,3; C sub: 0,1,2,3
    
    # M1 = (A00+A11)(B00+B11)
    terms.append(({0:1, 4:1}, {0:1, 3:1}, {0:1, 3:1}))
    # M2 = (A10+A11)B00
    terms.append(({3:1, 4:1}, {0:1}, {2:1, 3:-1}))
    # M3 = A00(B01-B11)
    terms.append(({0:1}, {1:1, 3:-1}, {1:1, 3:1}))
    # M4 = A11(B10-B00)
    terms.append(({4:1}, {2:1, 0:-1}, {0:1, 2:1}))
    # M5 = (A00+A01)B11
    terms.append(({0:1, 1:1}, {3:1}, {0:-1, 1:1}))
    # M6 = (A10-A00)(B00+B01)
    terms.append(({3:1, 0:-1}, {0:1, 1:1}, {3:1}))
    # M7 = (A01-A11)(B10+B11)
    terms.append(({1:1, 4:-1}, {2:1, 3:1}, {0:1}))
    
    # 4 scalar terms for A[:,2] * B[2,:]
    # K1 = A02 * B20 -> C00
    terms.append(({2:1}, {4:1}, {0:1}))
    # K2 = A02 * B21 -> C01
    terms.append(({2:1}, {5:1}, {1:1}))
    # K3 = A12 * B20 -> C10
    terms.append(({5:1}, {4:1}, {2:1}))
    # K4 = A12 * B21 -> C11
    terms.append(({5:1}, {5:1}, {3:1}))
    
    # Fill global matrices for 3 blocks
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

def _try_find_rank_R_2x3x3(R, max_time=20):
    """Try to find a rank-R decomposition for 2x3 x 3x3 with integer/half-integer entries."""
    start_time = time.time()
    n, m, p = 2, 3, 3
    rank = R
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
    
    # Random restarts
    for seed in range(500):
        if time.time() - start_time > max_time:
            break
        
        np.random.seed(seed)
        # Scale initial guess
        # For rank 10/11, values are often small
        scale = 0.5 + 0.5 * (seed % 10) / 10.0
        x0 = np.random.randn(dim_u*rank + dim_v*rank + dim_w*rank) * scale
        
        try:
            result = least_squares(residuals, x0, max_nfev=2000, ftol=1e-15, xtol=1e-15, gtol=1e-15)
            if result.cost < best_cost:
                best_cost = result.cost
                best_solution = result.x
                if best_cost < 1e-14:
                    break
        except:
            continue
    
    if best_solution is not None and best_cost < 1e-10:
        U = best_solution[:dim_u*rank].reshape(dim_u, rank)
        V = best_solution[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
        W = best_solution[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
        
        # Round to nearest half-integer and verify
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Verify the rounded solution
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
            
        # If direct rounding fails, try to optimize from the rounded point?
        # Or try different rounding strategies?
        # For now, return None to try other strategies.
    
    return None

def _build_rank22_decomposition(U11, V11, W11):
    """Build a rank-22 decomposition using rank-11 2x3x3 blocks."""
    n, m, p = 2, 3, 6
    rank = 22
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Block 0: k=0,1,2
    # Block 1: k=3,4,5
    for block in range(2):
        for c in range(11):
            col = block * 11 + c
            U[:, col] = U11[:, c]
            for j in range(m):
                for k in range(3):
                    V[j*p + (block*3 + k), col] = V11[j*3 + k, c]
            for k in range(3):
                for i in range(n):
                    W[(block*3 + k)*n + i, col] = W11[k*n + i, c]
    
    return U, V, W

def _build_rank20_decomposition(U10, V10, W10):
    """Build a rank-20 decomposition using rank-10 2x3x3 blocks."""
    n, m, p = 2, 3, 6
    rank = 20
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for block in range(2):
        for c in range(10):
            col = block * 10 + c
            U[:, col] = U10[:, c]
            for j in range(m):
                for k in range(3):
                    V[j*p + (block*3 + k), col] = V10[j*3 + k, c]
            for k in range(3):
                for i in range(n):
                    W[(block*3 + k)*n + i, col] = W10[k*n + i, c]
    
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
    
    # Strategy 1: Start from Strassen-like structure with 7 terms + 3 extra
    base_terms = [
        ({0:1, 4:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({3:1, 4:1}, {0:1}, {2:1, 3:-1}),
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        ({4:1}, {2:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        ({3:1, 0:-1}, {0:1, 1:1}, {3:1}),
        ({1:1, 4:-1}, {2:1, 3:1}, {0:1}),
        ({2:1}, {4:1}, {0:1}),
        ({2:1}, {5:1}, {1:1}),
        ({5:1}, {4:1}, {2:1}),
    ]
    
    U_init = np.zeros((dim_u, rank))
    V_init = np.zeros((dim_v, rank))
    W_init = np.zeros((dim_w, rank))
    
    for c, (u_dict, v_dict, w_dict) in enumerate(base_terms):
        for idx, val in u_dict.items():
            U_init[idx, c] = val
        for idx, val in v_dict.items():
            V_init[idx, c] = val
        for idx, val in w_dict.items():
            W_init[idx, c] = val
    
    x0_strassen = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    try:
        result = least_squares(residuals, x0_strassen, max_nfev=2000)
        if result.cost < best_cost:
            best_cost = result.cost
            best_solution = result.x
    except:
        pass
    
    # Strategy 2: Random restarts
    for seed in range(300):
        if time.time() - start_time > max_time:
            break
        
        np.random.seed(seed)
        scale = 0.3 + 0.4 * (seed % 5) / 5.0
        x0 = np.random.randn(dim_u*rank + dim_v*rank + dim_w*rank) * scale
        try:
            result = least_squares(residuals, x0, max_nfev=1500)
            if result.cost < best_cost:
                best_cost = result.cost
                best_solution = result.x
                if best_cost < 1e-14:
                    break
        except:
            continue
    
    if best_solution is not None and best_cost < 1e-10:
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
    deadline = start_time + 145.0
    
    # 1. Try Rank 10 for 2x3x3 (Global Rank 20)
    remaining = deadline - time.time() - 5.0
    if remaining > 10.0:
        result = _try_find_rank_R_2x3x3(10, max_time=min(remaining, 40.0))
        if result is not None and time.time() < deadline:
            U10, V10, W10 = result
            U, V, W = _build_rank20_decomposition(U10, V10, W10)
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # 2. Try Rank 11 for 2x3x3 (Global Rank 22)
    remaining = deadline - time.time() - 5.0
    if remaining > 10.0:
        result = _try_find_rank_R_2x3x3(11, max_time=min(remaining, 40.0))
        if result is not None and time.time() < deadline:
            U11, V11, W11 = result
            U, V, W = _build_rank22_decomposition(U11, V11, W11)
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # 3. Try Rank 10 for 2x3x2 (Global Rank 30)
    remaining = deadline - time.time() - 5.0
    if remaining > 10.0:
        result = _try_find_rank10_2x3x2(max_time=min(remaining, 40.0))
        if result is not None and time.time() < deadline:
            U10, V10, W10 = result
            U, V, W = _build_rank30_decomposition(U10, V10, W10)
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # 4. Fallback to Rank 33
    U, V, W = _build_rank33_decomposition()
    
    if time.time() < deadline:
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
