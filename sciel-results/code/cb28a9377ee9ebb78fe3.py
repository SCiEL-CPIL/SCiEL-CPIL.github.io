import numpy as np
import time
import sys
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 4]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1] if hasattr(U, 'shape') else len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if U_arr.shape != (n * m, rank) or V_arr.shape != (m * p, rank) or W_arr.shape != (p * n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _check_and_round(U, V, W, tol=1e-4):
    """Check if U, V, W are close to half-integers and return rounded versions if so."""
    Uf = U.flatten()
    Vf = V.flatten()
    Wf = W.flatten()
    
    dist_U = np.abs(Uf - np.round(2 * Uf) / 2.0)
    dist_V = np.abs(Vf - np.round(2 * Vf) / 2.0)
    dist_W = np.abs(Wf - np.round(2 * Wf) / 2.0)
    
    if np.max(dist_U) < tol and np.max(dist_V) < tol and np.max(dist_W) < tol:
        U_r = np.round(2 * Uf) / 2.0
        V_r = np.round(2 * Vf) / 2.0
        W_r = np.round(2 * Wf) / 2.0
        
        n, m, p = DIMENSIONS
        U_3d = U_r.reshape(n*m, -1)
        V_3d = V_r.reshape(m*p, -1)
        W_3d = W_r.reshape(p*n, -1)
        
        if _verify_decomposition(U_3d, V_3d, W_3d, n, m, p, tol=1e-8):
            return True, U_r.reshape(U.shape), V_r.reshape(V.shape), W_r.reshape(W.shape)
        
    return False, None, None, None

def _strassen_2x2x2():
    """Return rank-7 Strassen decomposition for 2x2 * 2x2."""
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01-B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B10)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[2][5] = 1
    
    # M7 = (A01-A11)(B01+B11)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[3][6] = 1
    
    return {"U": U, "V": V, "W": W}

def _build_strassen_based_init(n, m, p, rank, seed=0):
    """Build initialization using Strassen blocks where applicable."""
    rng = np.random.RandomState(seed)
    
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    total_size = size_U + size_V + size_W
    
    x0 = np.zeros(total_size)
    
    strassen = _strassen_2x2x2()
    sU = np.array(strassen['U'])  # 4x7
    sV = np.array(strassen['V'])  # 4x7
    sW = np.array(strassen['W'])  # 4x7
    
    # Embed Strassen terms in columns 0-6
    for c in range(7):
        for i in range(2):
            for j in range(2):
                s_idx = i * 2 + j
                x0[(i * m + j) * rank + c] = sU[s_idx, c]
        
        for j in range(2):
            for k in range(2):
                s_idx = j * 2 + k
                x0[(n * m) + (j * p + k) * rank + c] = sV[s_idx, c]
        
        for k in range(2):
            for i in range(2):
                s_idx = k * 2 + i
                x0[(n * m + m * p) + (k * n + i) * rank + c] = sW[s_idx, c]
    
    # For remaining terms, use random or structured init
    all_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                all_triples.append((i, j, k))
    
    if len(all_triples) > 0 and rank > 7:
        rng.shuffle(all_triples)
        remaining_cols = rank - 7
        for r in range(remaining_cols):
            start = r * (len(all_triples) // remaining_cols)
            end = (r + 1) * (len(all_triples) // remaining_cols) if r < remaining_cols - 1 else len(all_triples)
            
            for (i, j, k) in all_triples[start:end]:
                col = 7 + r
                x0[(i * m + j) * rank + col] = 1.0
                x0[(n * m) + (j * p + k) * rank + col] = 1.0
                x0[(n * m + m * p) + (k * n + i) * rank + col] = 1.0
    
    x0 += rng.normal(0, 0.01, total_size)
    return x0

def _build_random_init(n, m, p, rank, seed=0):
    """Build random initialization."""
    rng = np.random.RandomState(seed)
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    total_size = size_U + size_V + size_W
    x0 = rng.normal(0, 1.0, total_size)
    return x0

def _residual(x, n, m, p, rank):
    """Compute residual for the decomposition."""
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    
    Uf = x[:size_U].reshape(n*m, rank)
    Vf = x[size_U:size_U+size_V].reshape(m*p, rank)
    Wf = x[size_U+size_V:].reshape(p*n, rank)
    
    res = np.zeros(n*m*p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = Uf[i*m+j, :]
                v_row = Vf[j*p+k, :]
                w_row = Wf[k*n+i, :]
                val = np.dot(u_row, v_row * w_row)
                res[idx] = val - 1.0
                idx += 1
    return res

def _try_find_decomposition(n, m, p, rank, max_restarts=5, deadline=None, use_strassen=True):
    """Try to find a rank-r decomposition using multiple restarts."""
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    total_size = size_U + size_V + size_W
    
    def residual(x):
        return _residual(x, n, m, p, rank)
    
    best_result = None
    best_cost = float('inf')
    
    init_strategies = []
    if use_strassen and rank >= 7:
        init_strategies.append(('strassen', _build_strassen_based_init))
    init_strategies.append(('random', _build_random_init))
    
    for restart in range(max_restarts):
        if deadline is not None and time.time() > deadline:
            break
        
        strategy_name, init_func = init_strategies[restart % len(init_strategies)]
        seed = restart * 1000 + 42
        
        try:
            x0 = init_func(n, m, p, rank, seed=seed)
            
            res = least_squares(residual, x0, method='lm', xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=3000)
            
            if res.cost < best_cost:
                best_cost = res.cost
                best_x = res.x.copy()
            
            if res.cost < 1e-10:
                x_sol = res.x
                U_sol = x_sol[:size_U].reshape(n*m, rank)
                V_sol = x_sol[size_U:size_U+size_V].reshape(m*p, rank)
                W_sol = x_sol[size_U+size_V:].reshape(p*n, rank)
                
                valid, U_r, V_r, W_r = _check_and_round(U_sol, V_sol, W_sol)
                if valid:
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                
                # Polish step
                try:
                    res2 = least_squares(residual, x_sol, method='lm', xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=1000)
                    if res2.cost < 1e-12:
                        U_pol = res2.x[:size_U].reshape(n*m, rank)
                        V_pol = res2.x[size_U:size_U+size_V].reshape(m*p, rank)
                        W_pol = res2.x[size_U+size_V:].reshape(p*n, rank)
                        
                        valid2, U_r2, V_r2, W_r2 = _check_and_round(U_pol, V_pol, W_pol, tol=1e-3)
                        if valid2:
                            return {"U": U_r2.tolist(), "V": V_r2.tolist(), "W": W_r2.tolist()}
                except:
                    pass
                    
        except Exception as e:
            pass
    
    if best_cost < 1e-6 and best_cost > 1e-10:
        try:
            res_final = least_squares(residual, best_x, method='lm', xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=2000)
            if res_final.cost < 1e-10:
                U_sol = res_final.x[:size_U].reshape(n*m, rank)
                V_sol = res_final.x[size_U:size_U+size_V].reshape(m*p, rank)
                W_sol = res_final.x[size_U+size_V:].reshape(p*n, rank)
                
                valid, U_r, V_r, W_r = _check_and_round(U_sol, V_sol, W_sol, tol=1e-3)
                if valid:
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        except:
            pass
    
    return None

def _schoolbook(n, m, p):
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    for rank in range(20, 37):
        if time.time() > deadline:
            break
        
        if rank <= 24:
            max_restarts = 10
        elif rank <= 28:
            max_restarts = 7
        else:
            max_restarts = 4
        
        result = _try_find_decomposition(n, m, p, rank, max_restarts=max_restarts, deadline=deadline, use_strassen=True)
        if result:
            return result
            
    return _schoolbook(n, m, p)
