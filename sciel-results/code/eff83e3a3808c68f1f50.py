import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # Precompute constraint structure
    all_i = np.repeat(np.arange(n), p * n * m * m * p)
    all_k = np.tile(np.repeat(np.arange(p), n * m * m * p), n)
    all_a = np.tile(np.repeat(np.arange(n), m * m * p), n * p)
    all_b = np.tile(np.repeat(np.arange(m), m * p), n * p * n)
    all_c = np.tile(np.repeat(np.arange(m), p), n * p * n * m)
    all_d = np.tile(np.arange(p), n * p * n * m * m)
    
    cons_target = np.where(
        (all_i == all_a) & (all_b == all_c) & (all_d == all_k), 
        1.0, 0.0
    )
    
    n_cons = len(cons_target)
    
    idx_a = all_a * m + all_b
    idx_c = all_c * p + all_d
    idx_w = all_k * n + all_i
    
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
        U_sel = U[idx_a]
        V_sel = V[idx_c]
        W_sel = W[idx_w]
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        residuals = vals - cons_target
        return np.max(np.abs(residuals)) < tol

    def snap_to_half_integers(U, V, W):
        return (np.round(U * 2) / 2.0, np.round(V * 2) / 2.0, np.round(W * 2) / 2.0)

    def residual_vector(x, R):
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U = x[0:size_U].reshape(n*m, R)
        V = x[size_U:size_U+size_V].reshape(m*p, R)
        W = x[size_U+size_V:].reshape(p*n, R)
        
        U_sel = U[idx_a]
        V_sel = V[idx_c]
        W_sel = W[idx_w]
        
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        return vals - cons_target

    def try_optimize(U_init, V_init, W_init, R, max_nfev=10000):
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = opt.least_squares(residual_vector, x0, args=(R,), method='lm', max_nfev=max_nfev)
            
            x_opt = result.x
            size_U = n * m * R
            size_V = m * p * R
            U_opt = x_opt[0:size_U].reshape(n*m, R)
            V_opt = x_opt[size_U:size_U+size_V].reshape(m*p, R)
            W_opt = x_opt[size_U+size_V:].reshape(p*n, R)
            
            if verify_solution(U_opt, V_opt, W_opt, tol=1e-6):
                U_snap, V_snap, W_snap = snap_to_half_integers(U_opt, V_opt, W_opt)
                if verify_solution(U_snap, V_snap, W_snap, tol=1e-9):
                    return U_snap, V_snap, W_snap
        except Exception:
            pass
        
        return None, None, None

    start_time = time.time()
    time_limit = 140
    
    # Verify rank-36 solution
    U_snap, V_snap, W_snap = snap_to_half_integers(U36, V36, W36)
    if verify_solution(U_snap, V_snap, W_snap, tol=1e-9):
        best_U, best_V, best_W = U_snap, V_snap, W_snap
    else:
        best_U, best_V, best_W = U36, V36, W36
    
    # Try lower ranks
    for R in range(35, 28, -1):
        if time.time() - start_time > time_limit:
            break
        
        inits = []
        inits.append((U36[:, :R], V36[:, :R], W36[:, :R]))
        
        for scale in [0.5, 1.0]:
            inits.append((
                np.random.normal(0, scale, (n*m, R)),
                np.random.normal(0, scale, (m*p, R)),
                np.random.normal(0, scale, (p*n, R))
            ))
        
        for _ in range(2):
            inits.append((
                U36[:, :R].copy() + np.random.normal(0, 0.1, (n*m, R)),
                V36[:, :R].copy() + np.random.normal(0, 0.1, (m*p, R)),
                W36[:, :R].copy() + np.random.normal(0, 0.1, (p*n, R))
            ))
        
        found = False
        for U_init, V_init, W_init in inits:
            if time.time() - start_time > time_limit:
                break
            U_opt, V_opt, W_opt = try_optimize(U_init, V_init, W_init, R)
            if U_opt is not None:
                best_U, best_V, best_W = U_opt, V_opt, W_opt
                found = True
                break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
