import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    nm, mp, pn = n * m, m * p, p * n  # 8, 20, 10
    
    start_time = time.time()
    time_limit = 145  # Leave 5s margin for 150s limit
    
    # Precompute all constraint tuples
    cons_list = []
    for i in range(n):
        for k in range(p):
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        for d in range(p):
                            target = 1.0 if (b == c and a == i and d == k) else 0.0
                            cons_list.append((a * m + b, c * p + d, k * n + i, target))
    
    # Fast residual: vectorized
    def residual_vector(x, R):
        size_U = nm * R
        size_V = mp * R
        U = x[0:size_U].reshape(nm, R)
        V = x[size_U:size_U + size_V].reshape(mp, R)
        W = x[size_U + size_V:].reshape(pn, R)
        
        # For each constraint: sum_mu U[ai,mu] * V[vi,mu] * W[wi,mu] - target
        # Vectorize over constraints
        U_rows = np.array([x[0] for x in cons_list])
        V_rows = np.array([x[1] for x in cons_list])
        W_rows = np.array([x[2] for x in cons_list])
        targets = np.array([x[3] for x in cons_list])
        
        # U_rows: (N, R), V_rows: (N, R), W_rows: (N, R)
        # For each constraint i: sum_mu U[i,mu]*V[i,mu]*W[i,mu]
        vals = np.einsum('im,im,im->i', U_rows, V_rows, W_rows)
        return vals - targets
    
    # Build rank-36 solution
    def get_initial_solution_rank36():
        rank = 36
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        
        def add_strassen_222(i0, j0, k0):
            nonlocal col
            # M1: (A00+A11)(B00+B11)
            U[i0*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M2: (A10+A11)B00
            U[(i0+1)*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + (i0+1), col] += 1; W[(k0+1)*n + (i0+1), col] -= 1
            col += 1
            # M3: A00(B10-B11)
            U[i0*m + j0, col] += 1
            V[j0*p + (k0+1), col] += 1; V[(j0+1)*p + (k0+1), col] -= 1
            W[(k0+1)*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M4: A11(B01-B00)
            U[(i0+1)*m + (j0+1), col] += 1
            V[(j0+1)*p + k0, col] += 1; V[j0*p + k0, col] -= 1
            W[k0*n + i0, col] += 1; W[k0*n + (i0+1), col] += 1
            col += 1
            # M5: (A00+A01)B11
            U[i0*m + j0, col] += 1; U[i0*m + (j0+1), col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] -= 1; W[(k0+1)*n + i0, col] += 1
            col += 1
            # M6: (A10-A00)(B00+B01)
            U[(i0+1)*m + j0, col] += 1; U[i0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1; V[j0*p + (k0+1), col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M7: (A01-A11)(B10+B11)
            U[i0*m + (j0+1), col] += 1; U[(i0+1)*m + (j0+1), col] -= 1
            V[(j0+1)*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1
            col += 1
            return col
        
        def add_schoolbook_2x2x1(i0, j0, k0):
            nonlocal col
            for di in range(2):
                for dj in range(2):
                    U[(i0 + di)*m + (j0 + dj), col] = 1
                    V[(j0 + dj)*p + k0, col] = 1
                    W[k0*n + (i0 + di), col] = 1
                    col += 1
            return col
        
        add_strassen_222(0, 0, 0)
        add_strassen_222(0, 2, 0)
        add_strassen_222(0, 0, 2)
        add_strassen_222(0, 2, 2)
        add_schoolbook_2x2x1(0, 0, 4)
        add_schoolbook_2x2x1(0, 2, 4)
        
        return U, V, W
    
    # Verify solution
    def verify_solution(U, V, W):
        R = U.shape[1]
        for (ui, vi, wi, target) in cons_list:
            val = np.dot(U[ui], np.multiply(V[vi], W[wi]))
            if abs(val - target) > 1e-9:
                return False
        return True
    
    # Snap to half-integers
    def snap_to_half_integers(U, V, W):
        return (np.round(U * 2) / 2.0, np.round(V * 2) / 2.0, np.round(W * 2) / 2.0)
    
    U36, V36, W36 = get_initial_solution_rank36()
    
    # Verify rank-36
    if not verify_solution(U36, V36, W36):
        # Fallback to schoolbook
        rank = nm * mp
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U_snap, V_snap, W_snap = snap_to_half_integers(U36, V36, W36)
    if not verify_solution(U_snap, V_snap, W_snap):
        return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}
    
    best_rank = 36
    best_solution = (U_snap, V_snap, W_snap)
    
    # Try to reduce rank
    for R in range(35, 20, -1):
        if time.time() - start_time > time_limit:
            break
        
        found = False
        for attempt in range(5):
            if time.time() - start_time > time_limit:
                break
            
            if attempt == 0:
                U_init = U36[:, :R]
                V_init = V36[:, :R]
                W_init = W36[:, :R]
            elif attempt == 1:
                # Perturbed truncation
                U_init = U36[:, :R] + np.random.normal(0, 0.1, (nm, R))
                V_init = V36[:, :R] + np.random.normal(0, 0.1, (mp, R))
                W_init = W36[:, :R] + np.random.normal(0, 0.1, (pn, R))
            else:
                U_init = np.random.normal(0, 0.5, (nm, R))
                V_init = np.random.normal(0, 0.5, (mp, R))
                W_init = np.random.normal(0, 0.5, (pn, R))
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = opt.least_squares(residual_vector, x0, args=(R,), method='lm', max_nfev=10000)
                if result.success or np.max(np.abs(result.fun)) < 1e-8:
                    x_opt = result.x
                    size_U = nm * R
                    size_V = mp * R
                    U_opt = x_opt[0:size_U].reshape(nm, R)
                    V_opt = x_opt[size_U:size_U + size_V].reshape(mp, R)
                    W_opt = x_opt[size_U + size_V:].reshape(pn, R)
                    
                    res = residual_vector(x_opt, R)
                    if np.max(np.abs(res)) < 1e-7:
                        U_s, V_s, W_s = snap_to_half_integers(U_opt, V_opt, W_opt)
                        if verify_solution(U_s, V_s, W_s):
                            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                        else:
                            # Try to refine
                            x_refine = np.concatenate([U_s.flatten(), V_s.flatten(), W_s.flatten()])
                            result2 = opt.least_squares(residual_vector, x_refine, args=(R,), method='lm', max_nfev=20000)
                            x_ref = result2.x
                            U_r = x_ref[0:size_U].reshape(nm, R)
                            V_r = x_ref[size_U:size_U + size_V].reshape(mp, R)
                            W_r = x_ref[size_U + size_V:].reshape(pn, R)
                            U_s2, V_s2, W_s2 = snap_to_half_integers(U_r, V_r, W_r)
                            if verify_solution(U_s2, V_s2, W_s2):
                                return {"U": U_s2.tolist(), "V": V_s2.tolist(), "W": W_s2.tolist()}
            except Exception:
                pass
        
        if found:
            break
    
    return {"U": best_solution[0].tolist(), "V": best_solution[1].tolist(), "W": best_solution[2].tolist()}
