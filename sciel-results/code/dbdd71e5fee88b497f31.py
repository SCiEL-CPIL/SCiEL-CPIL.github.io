import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    cons_list = []
    for i in range(n):
        for k in range(p):
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        for d in range(p):
                            target = 1.0 if (b == c and a == i and d == k) else 0.0
                            cons_list.append((i, k, a, b, c, d, target))
    
    cons_i = np.array([c[0] for c in cons_list])
    cons_k = np.array([c[1] for c in cons_list])
    cons_a = np.array([c[2] for c in cons_list])
    cons_b = np.array([c[3] for c in cons_list])
    cons_c = np.array([c[4] for c in cons_list])
    cons_d = np.array([c[5] for c in cons_list])
    cons_target = np.array([c[6] for c in cons_list])
    
    U_idx = cons_a * m + cons_b
    V_idx = cons_c * p + cons_d
    W_idx = cons_k * n + cons_i
    
    def get_initial_solution_rank36():
        rank = 36
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        def add_strassen_222(i0, j0, k0):
            nonlocal col
            U[i0*m + j0][col] += 1; U[(i0+1)*m + (j0+1)][col] += 1
            V[j0*p + k0][col] += 1; V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] += 1; W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            U[(i0+1)*m + j0][col] += 1; U[(i0+1)*m + (j0+1)][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + (i0+1)][col] += 1; W[(k0+1)*n + (i0+1)][col] -= 1
            col += 1
            U[i0*m + j0][col] += 1
            V[j0*p + (k0+1)][col] += 1; V[(j0+1)*p + (k0+1)][col] -= 1
            W[(k0+1)*n + i0][col] += 1; W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            U[(i0+1)*m + (j0+1)][col] += 1
            V[(j0+1)*p + k0][col] += 1; V[j0*p + k0][col] -= 1
            W[k0*n + i0][col] += 1; W[k0*n + (i0+1)][col] += 1
            col += 1
            U[i0*m + j0][col] += 1; U[i0*m + (j0+1)][col] += 1
            V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] -= 1; W[(k0+1)*n + i0][col] += 1
            col += 1
            U[(i0+1)*m + j0][col] += 1; U[i0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1; V[j0*p + (k0+1)][col] += 1
            W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            U[i0*m + (j0+1)][col] += 1; U[(i0+1)*m + (j0+1)][col] -= 1
            V[(j0+1)*p + k0][col] += 1; V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] += 1
            col += 1

        def add_schoolbook_2x4x1(i0, j0, k0):
            nonlocal col
            for di in range(2):
                for dj in range(4):
                    U[(i0 + di)*m + (j0 + dj)][col] = 1
                    V[(j0 + dj)*p + k0][col] = 1
                    W[k0*n + (i0 + di)][col] = 1
                    col += 1

        add_strassen_222(0, 0, 0)
        add_strassen_222(0, 2, 0)
        add_strassen_222(0, 0, 2)
        add_strassen_222(0, 2, 2)
        add_schoolbook_2x4x1(0, 0, 4)
        
        return U, V, W

    U36, V36, W36 = get_initial_solution_rank36()
    
    def verify_solution(U, V, W, tol=1e-9):
        R = U.shape[1]
        vals = np.sum(U[U_idx] * V[V_idx] * W[W_idx], axis=1)
        return np.all(np.abs(vals - cons_target) < tol)

    def snap_to_half_integers(U, V, W):
        return np.round(U * 2) / 2.0, np.round(V * 2) / 2.0, np.round(W * 2) / 2.0

    def residual_vector_fast(x, R):
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = x[0:size_U]
        V_flat = x[size_U:size_U+size_V]
        W_flat = x[size_U+size_V:size_U+size_V+size_W]
        
        U = U_flat.reshape(n*m, R)
        V = V_flat.reshape(m*p, R)
        W = W_flat.reshape(p*n, R)
        
        vals = np.sum(U[U_idx] * V[V_idx] * W[W_idx], axis=1)
        return vals - cons_target

    def optimize_rank(R, U_init, V_init, W_init, max_nfev=20000):
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        try:
            result = opt.least_squares(residual_vector_fast, x0, args=(R,), method='lm', max_nfev=max_nfev)
            x_opt = result.x
            size_U = n * m * R
            size_V = m * p * R
            U_opt = x_opt[0:size_U].reshape(n*m, R)
            V_opt = x_opt[size_U:size_U+size_V].reshape(m*p, R)
            W_opt = x_opt[size_U+size_V:size_U+size_V+size_W].reshape(p*n, R)
            res = residual_vector_fast(x_opt, R)
            return U_opt, V_opt, W_opt, np.max(np.abs(res))
        except:
            return None, None, None, np.inf

    start_time = time.time()
    time_limit = 140
    
    best_U, best_V, best_W = U36.copy(), V36.copy(), W36.copy()
    best_rank = 36
    
    # Try to reduce rank
    for R in range(best_rank - 1, 19, -1):
        if time.time() - start_time > time_limit:
            break
        
        found = False
        num_attempts = 40 if R >= 33 else 20
        
        for attempt in range(num_attempts):
            if time.time() - start_time > time_limit:
                break
            
            if attempt == 0:
                U_init = best_U[:, :R]
                V_init = best_V[:, :R]
                W_init = best_W[:, :R]
            elif attempt == 1:
                U_init = best_U[:, :R] + np.random.normal(0, 0.05, (n*m, R))
                V_init = best_V[:, :R] + np.random.normal(0, 0.05, (m*p, R))
                W_init = best_W[:, :R] + np.random.normal(0, 0.05, (p*n, R))
            elif attempt == 2:
                U_init = best_U[:, :R] + np.random.normal(0, 0.1, (n*m, R))
                V_init = best_V[:, :R] + np.random.normal(0, 0.1, (m*p, R))
                W_init = best_W[:, :R] + np.random.normal(0, 0.1, (p*n, R))
            elif attempt == 3:
                U_init = best_U[:, :R] + np.random.normal(0, 0.2, (n*m, R))
                V_init = best_V[:, :R] + np.random.normal(0, 0.2, (m*p, R))
                W_init = best_W[:, :R] + np.random.normal(0, 0.2, (p*n, R))
            elif attempt == 4:
                U_init = best_U[:, :R] + np.random.normal(0, 0.5, (n*m, R))
                V_init = best_V[:, :R] + np.random.normal(0, 0.5, (m*p, R))
                W_init = best_W[:, :R] + np.random.normal(0, 0.5, (p*n, R))
            else:
                scale = 0.3 + 0.1 * (attempt % 7)
                U_init = np.random.normal(0, scale, (n*m, R))
                V_init = np.random.normal(0, scale, (m*p, R))
                W_init = np.random.normal(0, scale, (p*n, R))
            
            U_opt, V_opt, W_opt, max_res = optimize_rank(R, U_init, V_init, W_init)
            
            if U_opt is not None and max_res < 1e-6:
                U_snap, V_snap, W_snap = snap_to_half_integers(U_opt, V_opt, W_opt)
                if verify_solution(U_snap, V_snap, W_snap):
                    best_U, best_V, best_W = U_snap, V_snap, W_snap
                    best_rank = R
                    found = True
                    break
    
    U_snap, V_snap, W_snap = snap_to_half_integers(best_U, best_V, best_W)
    if verify_solution(U_snap, V_snap, W_snap):
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    U_snap, V_snap, W_snap = snap_to_half_integers(U36, V36, W36)
    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
