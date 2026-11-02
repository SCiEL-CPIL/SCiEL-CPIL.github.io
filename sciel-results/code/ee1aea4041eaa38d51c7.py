import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def _build_target_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _check_exact(U, V, W, n, m, p):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                val = np.dot(U[u, :], V[v, :] * W[w, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    for u in range(n * m):
        for v in range(m * p):
            for w in range(p * n):
                i1 = u // m
                j1 = u % m
                j2 = v // p
                k2 = v % p
                k3 = w // n
                i2 = w % n
                if i1 == i2 and j1 == j2 and k2 == k3:
                    continue
                val = np.dot(U[u, :], V[v, :] * W[w, :])
                if abs(val) > 1e-9:
                    return False
    return True

def _get_schoolbook(n, m, p):
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

def _precompute_indices(n, m, p):
    # For U update: for each u, list of (v, w, t_val)
    u_indices = []
    for u in range(n * m):
        i = u // m
        j = u % m
        pairs = []
        for v in range(m * p):
            for w in range(p * n):
                j_prime = v // p
                k_prime = v % p
                k_double_prime = w // n
                i_double_prime = w % n
                if i == i_double_prime and j == j_prime and k_prime == k_double_prime:
                    t_val = 1.0
                else:
                    t_val = 0.0
                pairs.append((v, w, t_val))
        u_indices.append(pairs)
        
    # For V update: for each v, list of (u, w, t_val)
    v_indices = []
    for v in range(m * p):
        j = v // p
        k = v % p
        pairs = []
        for u in range(n * m):
            for w in range(p * n):
                i1 = u // m
                j1 = u % m
                k2 = w // n
                i2 = w % n
                if i1 == i2 and j1 == j and k2 == k:
                    t_val = 1.0
                else:
                    t_val = 0.0
                pairs.append((u, w, t_val))
        v_indices.append(pairs)
        
    # For W update: for each w, list of (u, v, t_val)
    w_indices = []
    for w in range(p * n):
        k = w // n
        i = w % n
        pairs = []
        for u in range(n * m):
            for v in range(m * p):
                i1 = u // m
                j1 = u % m
                j2 = v // p
                k2 = v % p
                if i1 == i and j1 == j2 and k2 == k:
                    t_val = 1.0
                else:
                    t_val = 0.0
                pairs.append((u, v, t_val))
        w_indices.append(pairs)
        
    return u_indices, v_indices, w_indices

def _try_rank_alternating(R, n, m, p, time_limit, u_indices, v_indices, w_indices):
    start_time = time.time()
    init_values = [-2, -1, 0, 1, 2]
    
    best_U = None
    best_V = None
    best_W = None
    best_error = float('inf')
    
    num_restarts = 50
    max_iter = 50
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        V = np.random.choice(init_values, size=(m * p, R))
        W = np.random.choice(init_values, size=(p * n, R))
        
        current_error = float('inf')
        
        for it in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            
            # Update U
            U = np.zeros((n * m, R))
            for u in range(n * m):
                pairs = u_indices[u]
                A = np.zeros((len(pairs), R))
                b = np.zeros(len(pairs))
                for idx, (v, w, t_val) in enumerate(pairs):
                    b[idx] = t_val
                    A[idx, :] = V[v, :] * W[w, :]
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    U[u, :] = sol
                except Exception:
                    pass
            
            U = np.round(U)
            
            # Update V
            V_new = np.zeros((m * p, R))
            for v in range(m * p):
                pairs = v_indices[v]
                A = np.zeros((len(pairs), R))
                b = np.zeros(len(pairs))
                for idx, (u, w, t_val) in enumerate(pairs):
                    b[idx] = t_val
                    A[idx, :] = U[u, :] * W[w, :]
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    V_new[v, :] = sol
                except Exception:
                    pass
            V = np.round(V_new)
            
            # Update W
            W_new = np.zeros((p * n, R))
            for w in range(p * n):
                pairs = w_indices[w]
                A = np.zeros((len(pairs), R))
                b = np.zeros(len(pairs))
                for idx, (u, v, t_val) in enumerate(pairs):
                    b[idx] = t_val
                    A[idx, :] = U[u, :] * V[v, :]
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    W_new[w, :] = sol
                except Exception:
                    pass
            W = np.round(W_new)
            
            # Check error
            T_target = _build_target_tensor(n, m, p)
            T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
            error = np.linalg.norm(T_recon - T_target)
            
            if error < 1e-9:
                if _check_exact(U, V, W, n, m, p):
                    return U, V, W
            
            if error < current_error:
                current_error = error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            if current_error == 0.0:
                break

    if best_error < 1e-9:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    U_sb, V_sb, W_sb = _get_schoolbook(n, m, p)
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = n * m * p
    
    u_indices, v_indices, w_indices = _precompute_indices(n, m, p)
    
    start_time = time.time()
    deadline = 140.0
    
    for R in range(12, best_rank):
        if time.time() - start_time > deadline:
            break
        
        time_budget = min(10.0, (deadline - (time.time() - start_time)) / max(best_rank - R, 1))
        if time_budget < 1.0:
            break
        
        result = _try_rank_alternating(R, n, m, p, time_budget, u_indices, v_indices, w_indices)
        
        if result is not None:
            U, V, W = result
            if _check_exact(U, V, W, n, m, p):
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = R
                break
    
    return best_result
