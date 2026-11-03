import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    time_limit = 115

    # Precompute constraint structure
    cons_list = []
    for i in range(n):
        for k in range(p):
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        for d in range(p):
                            target = 1.0 if (b == c and a == i and d == k) else 0.0
                            cons_list.append((i, k, a, b, c, d, target))
    
    cons_idx = np.array(cons_list)
    cons_i = cons_idx[:, 0].astype(int)
    cons_k = cons_idx[:, 1].astype(int)
    cons_a = cons_idx[:, 2].astype(int)
    cons_b = cons_idx[:, 3].astype(int)
    cons_c = cons_idx[:, 4].astype(int)
    cons_d = cons_idx[:, 5].astype(int)
    cons_target = cons_idx[:, 6]
    N_cons = len(cons_target)

    def verify_solution(U, V, W):
        R = U.shape[1]
        U_sel = U[cons_a * m + cons_b, :]
        V_sel = V[cons_c * p + cons_d, :]
        W_sel = W[cons_k * n + cons_i, :]
        vals = np.einsum('ir,ir,ir->i', U_sel, V_sel, W_sel)
        return np.max(np.abs(vals - cons_target)) < 1e-9

    def snap_and_verify(U, V, W):
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        if verify_solution(U_s, V_s, W_s):
            return U_s, V_s, W_s
        # Try rounding to integers
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if verify_solution(U_i, V_i, W_i):
            return U_i, V_i, W_i
        return None

    def residual_fast(x, R):
        size_U = nm * R
        size_V = mp * R
        U = x[0:size_U].reshape(nm, R)
        V = x[size_U:size_U+size_V].reshape(mp, R)
        W = x[size_U+size_V:].reshape(pn, R)
        U_sel = U[cons_a * m + cons_b, :]
        V_sel = V[cons_c * p + cons_d, :]
        W_sel = W[cons_k * n + cons_i, :]
        return np.einsum('ir,ir,ir->i', U_sel, V_sel, W_sel) - cons_target

    def build_strassen_rank36():
        rank = 36
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1: (A00+A11)(B00+B11)
            U[i0*m+j0, col] += 1; U[(i0+1)*m+(j0+1), col] += 1
            V[j0*p+k0, col] += 1; V[(j0+1)*p+(k0+1), col] += 1
            W[k0*n+i0, col] += 1; W[(k0+1)*n+(i0+1), col] += 1
            col += 1
            # M2: (A10+A11)B00
            U[(i0+1)*m+j0, col] += 1; U[(i0+1)*m+(j0+1), col] += 1
            V[j0*p+k0, col] += 1
            W[k0*n+(i0+1), col] += 1; W[(k0+1)*n+(i0+1), col] -= 1
            col += 1
            # M3: A00(B10-B11)
            U[i0*m+j0, col] += 1
            V[j0*p+(k0+1), col] += 1; V[(j0+1)*p+(k0+1), col] -= 1
            W[(k0+1)*n+i0, col] += 1; W[(k0+1)*n+(i0+1), col] += 1
            col += 1
            # M4: A11(B01-B00)
            U[(i0+1)*m+(j0+1), col] += 1
            V[(j0+1)*p+k0, col] += 1; V[j0*p+k0, col] -= 1
            W[k0*n+i0, col] += 1; W[k0*n+(i0+1), col] += 1
            col += 1
            # M5: (A00+A01)B11
            U[i0*m+j0, col] += 1; U[i0*m+(j0+1), col] += 1
            V[(j0+1)*p+(k0+1), col] += 1
            W[k0*n+i0, col] -= 1; W[(k0+1)*n+i0, col] += 1
            col += 1
            # M6: (A10-A00)(B00+B01)
            U[(i0+1)*m+j0, col] += 1; U[i0*m+j0, col] -= 1
            V[j0*p+k0, col] += 1; V[j0*p+(k0+1), col] += 1
            W[(k0+1)*n+(i0+1), col] += 1
            col += 1
            # M7: (A01-A11)(B10+B11)
            U[i0*m+(j0+1), col] += 1; U[(i0+1)*m+(j0+1), col] -= 1
            V[(j0+1)*p+k0, col] += 1; V[(j0+1)*p+(k0+1), col] += 1
            W[k0*n+i0, col] += 1
            col += 1

        def add_schoolbook(i0, j0, k0):
            nonlocal col
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m+(j0+dj), col] = 1
                    V[(j0+dj)*p+k0, col] = 1
                    W[k0*n+(i0+di), col] = 1
                    col += 1

        add_strassen(0, 0, 0)
        add_strassen(0, 2, 0)
        add_strassen(0, 0, 2)
        add_strassen(0, 2, 2)
        add_schoolbook(0, 0, 4)
        add_schoolbook(0, 2, 4)
        return U, V, W

    def build_schoolbook():
        rank = nm * mp
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    U36, V36, W36 = build_strassen_rank36()
    
    if not verify_solution(U36, V36, W36):
        U36, V36, W36 = build_schoolbook()
    
    snapped = snap_and_verify(U36, V36, W36)
    if snapped:
        U36, V36, W36 = snapped
    
    best_rank = 36
    best_sol = (U36.copy(), V36.copy(), W36.copy())

    # Try to reduce rank from 35 down
    for R in range(35, 24, -1):
        if time.time() - start_time > time_limit - 5:
            break
        
        found = False
        n_attempts = 8 if R > 30 else 12
        
        for attempt in range(n_attempts):
            if time.time() - start_time > time_limit - 3:
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
                # Random with structured initialization
                scale = 0.5
                U_init = np.random.normal(0, scale, (nm, R))
                V_init = np.random.normal(0, scale, (mp, R))
                W_init = np.random.normal(0, scale, (pn, R))
                # Add some structure: ensure some entries are non-zero in expected places
                for ii in range(n):
                    for jj in range(m):
                        U_init[ii*m+jj, 0] += 0.3
                    for kk in range(p):
                        W_init[kk*n+ii, 0] += 0.3
                for jj in range(m):
                    for kk in range(p):
                        V_init[jj*p+kk, 0] += 0.3

            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = opt.least_squares(residual_fast, x0, args=(R,), method='lm', max_nfev=3000)
                if result.cost < 1e-8:
                    x_opt = result.x
                    size_U = nm * R
                    size_V = mp * R
                    U_opt = x_opt[0:size_U].reshape(nm, R)
                    V_opt = x_opt[size_U:size_U+size_V].reshape(mp, R)
                    W_opt = x_opt[size_U+size_V:].reshape(pn, R)
                    
                    # Polish
                    if time.time() - start_time < time_limit - 5:
                        try:
                            result2 = opt.least_squares(residual_fast, x_opt, args=(R,), method='lm', max_nfev=1000, xtol=1e-14, ftol=1e-14)
                            if result2.cost < 1e-10:
                                x_opt = result2.x
                                U_opt = x_opt[0:size_U].reshape(nm, R)
                                V_opt = x_opt[size_U:size_U+size_V].reshape(mp, R)
                                W_opt = x_opt[size_U+size_V:].reshape(pn, R)
                        except:
                            pass
                    
                    res = residual_fast(x_opt, R)
                    if np.max(np.abs(res)) < 1e-7:
                        snapped = snap_and_verify(U_opt, V_opt, W_opt)
                        if snapped:
                            U_s, V_s, W_s = snapped
                            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                        else:
                            # Check if close enough to half-integers with tighter snap
                            U_close = np.round(U_opt * 4) / 4.0
                            V_close = np.round(V_opt * 4) / 4.0
                            W_close = np.round(W_opt * 4) / 4.0
                            if verify_solution(U_close, V_close, W_close):
                                return {"U": U_close.tolist(), "V": V_close.tolist(), "W": W_close.tolist()}
                    found = True
            except:
                pass
        
        if not found and R <= 30:
            # Try a few more random restarts for lower ranks
            for extra in range(4):
                if time.time() - start_time > time_limit - 3:
                    break
                U_init = np.random.normal(0, 1.0, (nm, R))
                V_init = np.random.normal(0, 1.0, (mp, R))
                W_init = np.random.normal(0, 1.0, (pn, R))
                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                try:
                    result = opt.least_squares(residual_fast, x0, args=(R,), method='lm', max_nfev=2000)
                    if result.cost < 1e-8:
                        x_opt = result.x
                        size_U = nm * R
                        size_V = mp * R
                        U_opt = x_opt[0:size_U].reshape(nm, R)
                        V_opt = x_opt[size_U:size_U+size_V].reshape(mp, R)
                        W_opt = x_opt[size_U+size_V:].reshape(pn, R)
                        res = residual_fast(x_opt, R)
                        if np.max(np.abs(res)) < 1e-7:
                            snapped = snap_and_verify(U_opt, V_opt, W_opt)
                            if snapped:
                                U_s, V_s, W_s = snapped
                                return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                except:
                    pass

    return {"U": best_sol[0].tolist(), "V": best_sol[1].tolist(), "W": best_sol[2].tolist()}
