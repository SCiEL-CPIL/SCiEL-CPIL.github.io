import numpy as np
import time
from fractions import Fraction
import itertools

DIMENSIONS = [2, 3, 4]

def _verify_solution(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for jp in range(m):
                    s = 0.0
                    for c in range(rank):
                        s += W[k * n + i][c] * U[i * m + j][c] * V[jp * p + k][c]
                    target = 1.0 if j == jp else 0.0
                    if abs(s - target) > 1e-9:
                        return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _snap_to_half_int(x):
    return np.round(x * 2) / 2

def _snap_to_rational(x, denom=12):
    for d in [1, 2, 3, 4, 6, 12]:
        snapped = np.round(x * d) / d
        if np.allclose(snapped, x, atol=1e-8):
            return snapped
    return np.round(x * 2) / 2

def _polish(U, V, W, n, m, p, rank, max_nfev=5000):
    from scipy.optimize import least_squares
    U_flat = U.flatten()
    V_flat = V.flatten()
    W_flat = W.flatten()
    x0 = np.concatenate([U_flat, V_flat, W_flat])
    
    def residual(x):
        U_res = x[:n*m*rank].reshape(n*m, rank)
        V_res = x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
        W_res = x[n*m*rank + m*p*rank :].reshape(p*n, rank)
        
        res = []
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = np.dot(W_res[k*n+i], U_res[i*m+j] * V_res[jp*p+k])
                        target = 1.0 if j == jp else 0.0
                        res.append(s - target)
        return np.array(res)
        
    try:
        res = least_squares(residual, x0, method='trf', max_nfev=max_nfev, ftol=1e-14, xtol=1e-14, gtol=1e-14)
        x_opt = res.x
        U_opt = x_opt[:n*m*rank].reshape(n*m, rank)
        V_opt = x_opt[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
        W_opt = x_opt[n*m*rank + m*p*rank :].reshape(p*n, rank)
        
        snap_candidates = [
            _snap_to_half_int(U_opt), _snap_to_half_int(V_opt), _snap_to_half_int(W_opt),
            np.round(U_opt), np.round(V_opt), np.round(W_opt),
            _snap_to_rational(U_opt), _snap_to_rational(V_opt), _snap_to_rational(W_opt),
        ]
        
        for Uc, Vc, Wc in zip([snap_candidates[0], snap_candidates[3], snap_candidates[6]],
                              [snap_candidates[1], snap_candidates[4], snap_candidates[7]],
                              [snap_candidates[2], snap_candidates[5], snap_candidates[8]]):
            if _verify_solution(Uc.tolist(), Vc.tolist(), Wc.tolist(), n, m, p):
                return Uc.tolist(), Vc.tolist(), Wc.tolist()
            
        if np.linalg.norm(res.fun) < 1e-12:
            return U_opt.tolist(), V_opt.tolist(), W_opt.tolist()
    except Exception:
        pass
    return None

def _structured_init(n, m, p, rank, variant=0):
    rng = np.random.RandomState(variant * 1000 + 42)
    
    if variant == 0:
        U = rng.randn(n*m, rank) * 0.5
        V = rng.randn(m*p, rank) * 0.5
        W = rng.randn(p*n, rank) * 0.5
    elif variant == 1:
        U = rng.randn(n*m, rank) * 0.5
        V = rng.randn(m*p, rank) * 0.5
        W = rng.randn(p*n, rank) * 0.5
        for mat in [U, V, W]:
            mask = rng.rand(*mat.shape) < 0.3
            mat[mask] = 0
    elif variant == 2:
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for c in range(rank):
            U[:, c] = rng.randn(n*m) * 0.4
            V[:, c] = rng.randn(m*p) * 0.4
            W[:, c] = rng.randn(p*n) * 0.4
    elif variant == 3:
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for c in range(rank):
            U[:, c] = rng.choice([-1, 0, 1], size=n*m) * 0.5
            V[:, c] = rng.choice([-1, 0, 1], size=m*p) * 0.5
            W[:, c] = rng.choice([-1, 0, 1], size=p*n) * 0.5
    else:
        U = rng.randn(n*m, rank) * 0.3
        V = rng.randn(m*p, rank) * 0.3
        W = rng.randn(p*n, rank) * 0.3
    
    return U, V, W

def _als_solve(rank, n, m, p, max_iters=300, restarts=50, timeout=30):
    start_time = time.time()
    
    for restart in range(restarts):
        if time.time() - start_time > timeout:
            break
            
        U, V, W = _structured_init(n, m, p, rank, variant=restart % 5)
        
        for it in range(max_iters):
            if time.time() - start_time > timeout:
                return None
                
            for i in range(n):
                for j in range(m):
                    A_rows = []
                    b_vals = []
                    for k in range(p):
                        for jp in range(m):
                            A_rows.append(W[k * n + i] * V[jp * p + k])
                            b_vals.append(1.0 if j == jp else 0.0)
                    A = np.array(A_rows)
                    b = np.array(b_vals)
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[i * m + j] = x
                    
            for j in range(m):
                for k in range(p):
                    A_rows = []
                    b_vals = []
                    for i in range(n):
                        for jp in range(m):
                            A_rows.append(W[k * n + i] * U[i * m + jp])
                            b_vals.append(1.0 if j == jp else 0.0)
                    A = np.array(A_rows)
                    b = np.array(b_vals)
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[j * p + k] = x

            for k in range(p):
                for i in range(n):
                    A_rows = []
                    b_vals = []
                    for j in range(m):
                        for jp in range(m):
                            A_rows.append(U[i * m + j] * V[jp * p + k])
                            b_vals.append(1.0 if j == jp else 0.0)
                    A = np.array(A_rows)
                    b = np.array(b_vals)
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[k * n + i] = x

            error = 0.0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            s = np.dot(W[k * n + i], U[i * m + j] * V[jp * p + k])
                            target = 1.0 if j == jp else 0.0
                            error += (s - target) ** 2
            
            if error < 1e-12:
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if _verify_solution(U_list, V_list, W_list, n, m, p):
                    return U_list, V_list, W_list
            
            if error < 1e-8 and it > 10:
                polished = _polish(U, V, W, n, m, p, rank, max_nfev=2000)
                if polished:
                    U_list, V_list, W_list = polished
                    if _verify_solution(U_list, V_list, W_list, n, m, p):
                        return U_list, V_list, W_list

    return None

def _try_direct_rank12(n, m, p):
    rank = 12
    best_error = float('inf')
    best_solution = None
    
    for seed in range(20):
        U, V, W = _structured_init(n, m, p, rank, variant=seed)
        
        for it in range(50):
            for i in range(n):
                for j in range(m):
                    A_rows = []
                    b_vals = []
                    for k in range(p):
                        for jp in range(m):
                            A_rows.append(W[k * n + i] * V[jp * p + k])
                            b_vals.append(1.0 if j == jp else 0.0)
                    A = np.array(A_rows)
                    b = np.array(b_vals)
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[i * m + j] = x
                    
            for j in range(m):
                for k in range(p):
                    A_rows = []
                    b_vals = []
                    for i in range(n):
                        for jp in range(m):
                            A_rows.append(W[k * n + i] * U[i * m + jp])
                            b_vals.append(1.0 if j == jp else 0.0)
                    A = np.array(A_rows)
                    b = np.array(b_vals)
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[j * p + k] = x

            for k in range(p):
                for i in range(n):
                    A_rows = []
                    b_vals = []
                    for j in range(m):
                        for jp in range(m):
                            A_rows.append(U[i * m + j] * V[jp * p + k])
                            b_vals.append(1.0 if j == jp else 0.0)
                    A = np.array(A_rows)
                    b = np.array(b_vals)
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[k * n + i] = x
            
            error = 0.0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            s = np.dot(W[k * n + i], U[i * m + j] * V[jp * p + k])
                            target = 1.0 if j == jp else 0.0
                            error += (s - target) ** 2
            
            if error < best_error:
                best_error = error
                best_solution = (U.copy(), V.copy(), W.copy())
            
            if error < 1e-10:
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if _verify_solution(U_list, V_list, W_list, n, m, p):
                    return U_list, V_list, W_list
        
        if best_error < 1e-6 and best_solution is not None:
            U_b, V_b, W_b = best_solution
            polished = _polish(U_b, V_b, W_b, n, m, p, rank, max_nfev=3000)
            if polished:
                U_list, V_list, W_list = polished
                if _verify_solution(U_list, V_list, W_list, n, m, p):
                    return U_list, V_list, W_list
    
    return None

def _winograd_style_init(n, m, p, rank, seed=0):
    rng = np.random.RandomState(seed)
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for c in range(rank):
        scale = rng.uniform(0.3, 0.7)
        U[:, c] = rng.randn(n*m) * scale
        V[:, c] = rng.randn(m*p) * scale
        W[:, c] = rng.randn(p*n) * scale
    
    return U, V, W

def _als_solve_v2(rank, n, m, p, max_iters=300, restarts=50, timeout=30, use_winograd_init=False):
    start_time = time.time()
    
    for restart in range(restarts):
        if time.time() - start_time > timeout:
            break
            
        if use_winograd_init and restart % 2 == 0:
            U, V, W = _winograd_style_init(n, m, p, rank, seed=restart)
        else:
            U, V, W = _structured_init(n, m, p, rank, variant=restart % 5)
        
        for it in range(max_iters):
            if time.time() - start_time > timeout:
                return None
                
            for i in range(n):
                for j in range(m):
                    A_rows = []
                    b_vals = []
                    for k in range(p):
                        for jp in range(m):
                            A_rows.append(W[k * n + i] * V[jp * p + k])
                            b_vals.append(1.0 if j == jp else 0.0)
                    A = np.array(A_rows)
                    b = np.array(b_vals)
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[i * m + j] = x
                    
            for j in range(m):
                for k in range(p):
                    A_rows = []
                    b_vals = []
                    for i in range(n):
                        for jp in range(m):
                            A_rows.append(W[k * n + i] * U[i * m + jp])
                            b_vals.append(1.0 if j == jp else 0.0)
                    A = np.array(A_rows)
                    b = np.array(b_vals)
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[j * p + k] = x

            for k in range(p):
                for i in range(n):
                    A_rows = []
                    b_vals = []
                    for j in range(m):
                        for jp in range(m):
                            A_rows.append(U[i * m + j] * V[jp * p + k])
                            b_vals.append(1.0 if j == jp else 0.0)
                    A = np.array(A_rows)
                    b = np.array(b_vals)
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[k * n + i] = x

            error = 0.0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            s = np.dot(W[k * n + i], U[i * m + j] * V[jp * p + k])
                            target = 1.0 if j == jp else 0.0
                            error += (s - target) ** 2
            
            if error < 1e-12:
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if _verify_solution(U_list, V_list, W_list, n, m, p):
                    return U_list, V_list, W_list
            
            if error < 1e-8 and it > 10:
                polished = _polish(U, V, W, n, m, p, rank, max_nfev=2000)
                if polished:
                    U_list, V_list, W_list = polished
                    if _verify_solution(U_list, V_list, W_list, n, m, p):
                        return U_list, V_list, W_list

    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = 145
    
    if time.time() < deadline - 30:
        result = _als_solve_v2(12, n, m, p, max_iters=200, restarts=30, timeout=25, use_winograd_init=True)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    ranks_to_try = [12, 13, 14, 15, 16, 17, 18, 19, 20]
    
    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
            
        remaining = deadline - time.time()
        if remaining < 5:
            break
            
        if r <= 13:
            t_limit = min(25, remaining - 2)
        elif r <= 15:
            t_limit = min(15, remaining - 2)
        elif r <= 18:
            t_limit = min(10, remaining - 2)
        else:
            t_limit = min(5, remaining - 2)
            
        if t_limit < 1:
            continue

        result = _als_solve_v2(r, n, m, p, max_iters=200, restarts=30, timeout=t_limit, use_winograd_init=True)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    return _schoolbook()
