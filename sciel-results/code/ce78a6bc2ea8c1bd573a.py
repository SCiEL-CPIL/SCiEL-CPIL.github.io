import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def _build_tensor(n, m, p):
    return np.ones((n, m, p))

def _reconstruct(U, V, W, n, m, p, rank):
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return T

def _validate(U, V, W, n, m, p, rank, tol=1e-9):
    T = _reconstruct(U, V, W, n, m, p, rank)
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=tol)

def _snap_to_half_integers(U, V, W):
    return np.round(U * 2) / 2.0, np.round(V * 2) / 2.0, np.round(W * 2) / 2.0

def _coordinate_descent(U, V, W, n, m, p, rank, max_steps=10000, time_limit=10.0, start_time=0):
    T_target = np.ones((n, m, p))
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    err = np.sum((T - T_target)**2)
    
    best_err = err
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    steps_since_improve = 0
    rng = np.random.RandomState(42)
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
            
        if err == 0:
            break
            
        # Pick a random variable
        # 1/3 chance for U, 1/3 for V, 1/3 for W
        var_type = rng.randint(3)
        
        if var_type == 0:
            idx = rng.randint(n * m)
            c = rng.randint(rank)
            i, j = divmod(idx, m)
            # Try changing U[i, j, c]
            current_val = U_r[i, j, c]
            # Try neighbors
            for delta in [-1.0, -0.5, 0.5, 1.0, -2.0, 2.0]:
                new_val = current_val + delta
                # Optimization: only consider half-integers
                if abs(new_val * 2 - round(new_val * 2)) > 1e-9:
                    continue
                U_r[i, j, c] = new_val
                T_new = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                err_new = np.sum((T_new - T_target)**2)
                if err_new < err:
                    err = err_new
                    if err < best_err:
                        best_err = err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        steps_since_improve = 0
                    else:
                        steps_since_improve += 1
                else:
                    U_r[i, j, c] = current_val
        elif var_type == 1:
            idx = rng.randint(m * p)
            c = rng.randint(rank)
            j, k = divmod(idx, p)
            current_val = V_r[j, k, c]
            for delta in [-1.0, -0.5, 0.5, 1.0, -2.0, 2.0]:
                new_val = current_val + delta
                if abs(new_val * 2 - round(new_val * 2)) > 1e-9:
                    continue
                V_r[j, k, c] = new_val
                T_new = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                err_new = np.sum((T_new - T_target)**2)
                if err_new < err:
                    err = err_new
                    if err < best_err:
                        best_err = err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        steps_since_improve = 0
                    else:
                        steps_since_improve += 1
                else:
                    V_r[j, k, c] = current_val
        else:
            idx = rng.randint(p * n)
            c = rng.randint(rank)
            k, i = divmod(idx, n)
            current_val = W_r[k, i, c]
            for delta in [-1.0, -0.5, 0.5, 1.0, -2.0, 2.0]:
                new_val = current_val + delta
                if abs(new_val * 2 - round(new_val * 2)) > 1e-9:
                    continue
                W_r[k, i, c] = new_val
                T_new = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                err_new = np.sum((T_new - T_target)**2)
                if err_new < err:
                    err = err_new
                    if err < best_err:
                        best_err = err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        steps_since_improve = 0
                    else:
                        steps_since_improve += 1
                else:
                    W_r[k, i, c] = current_val
                    
        if steps_since_improve > 200:
            # Restart with random perturbation
            U_r += rng.randn(n, m, rank) * 0.1
            V_r += rng.randn(m, p, rank) * 0.1
            W_r += rng.randn(p, n, rank) * 0.1
            U_r = np.round(U_r * 2) / 2.0
            V_r = np.round(V_r * 2) / 2.0
            W_r = np.round(W_r * 2) / 2.0
            T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            err = np.sum((T - T_target)**2)
            steps_since_improve = 0

    return best_U, best_V, best_W, best_err

def _try_rank(n, m, p, rank, seed, time_limit):
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    # Sparse initialization
    for attempt in range(20):
        if time.time() - start_time > time_limit:
            break
            
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Random sparse half-integers
        for c in range(rank):
            # Pick a few non-zeros
            nz_u = rng.randint(0, n*m, size=rng.randint(1, 3))
            nz_v = rng.randint(0, m*p, size=rng.randint(1, 3))
            nz_w = rng.randint(0, p*n, size=rng.randint(1, 3))
            
            U[nz_u, c] = rng.choice([0.5, -0.5, 1.0, -1.0])
            V[nz_v, c] = rng.choice([0.5, -0.5, 1.0, -1.0])
            W[nz_w, c] = rng.choice([0.5, -0.5, 1.0, -1.0])
            
        U_s, V_s, W_s, err = _coordinate_descent(U, V, W, n, m, p, rank, 
                                                 max_steps=5000, 
                                                 time_limit=time_limit - (time.time() - start_time),
                                                 start_time=start_time)
        if err < 1e-9:
            U_s, V_s, W_s = _snap_to_half_integers(U_s, V_s, W_s)
            if _validate(U_s, V_s, W_s, n, m, p, rank):
                return U_s, V_s, W_s
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = schoolbook_rank
    
    start_time = time.time()
    time_limit = 145
    
    lower_bound = max(n * m, m * p, p * n)
    
    for rank in range(lower_bound, schoolbook_rank):
        if time.time() - start_time > time_limit - 10:
            break
        
        rank_time_limit = min(10, time_limit - (time.time() - start_time))
        if rank_time_limit <= 2:
            break
            
        result = _try_rank(n, m, p, rank, seed=rank * 1000 + 42, time_limit=rank_time_limit)
        if result is not None:
            U_s, V_s, W_s = result
            U_list = [[val for val in row] for row in U_s.tolist()]
            V_list = [[val for val in row] for row in V_s.tolist()]
            W_list = [[val for val in row] for row in W_s.tolist()]
            
            if _validate(np.array(U_list), np.array(V_list), np.array(W_list), n, m, p, rank):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = rank
                break
                
    return best_result
