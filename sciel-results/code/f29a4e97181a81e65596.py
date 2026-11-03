import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # Build the target tensor T[a, b, c, d, k, i] = delta_b_c * delta_a_i * delta_d_k
    # Shape: (n*m, m*p, p*n) -> (8, 20, 10)
    # T[a*m+b, c*p+d, k*n+i] = 1 if b==c and a==i and d==k, else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(p):
            for b in range(m):
                a = i  # delta_a_i
                d = k  # delta_d_k
                c = b  # delta_b_c
                T[a*m+b, c*p+d, k*n+i] = 1.0
    
    def get_initial_solution_rank36():
        rank = 36
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        def add_strassen_222(i0, j0, k0):
            nonlocal col
            # M1
            U[i0*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M2
            U[(i0+1)*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + (i0+1), col] += 1; W[(k0+1)*n + (i0+1), col] -= 1
            col += 1
            # M3
            U[i0*m + j0, col] += 1
            V[j0*p + (k0+1), col] += 1; V[(j0+1)*p + (k0+1), col] -= 1
            W[(k0+1)*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M4
            U[(i0+1)*m + (j0+1), col] += 1
            V[(j0+1)*p + k0, col] += 1; V[j0*p + k0, col] -= 1
            W[k0*n + i0, col] += 1; W[k0*n + (i0+1), col] += 1
            col += 1
            # M5
            U[i0*m + j0, col] += 1; U[i0*m + (j0+1), col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] -= 1; W[(k0+1)*n + i0, col] += 1
            col += 1
            # M6
            U[(i0+1)*m + j0, col] += 1; U[i0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1; V[j0*p + (k0+1), col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M7
            U[i0*m + (j0+1), col] += 1; U[(i0+1)*m + (j0+1), col] -= 1
            V[(j0+1)*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1
            col += 1

        def add_schoolbook_2x4x1(i0, j0, k0):
            nonlocal col
            for di in range(2):
                for dj in range(4):
                    U[(i0 + di)*m + (j0 + dj), col] = 1
                    V[(j0 + dj)*p + k0, col] = 1
                    W[k0*n + (i0 + di), col] = 1
                    col += 1

        add_strassen_222(0, 0, 0)
        add_strassen_222(0, 2, 0)
        add_strassen_222(0, 0, 2)
        add_strassen_222(0, 2, 2)
        add_schoolbook_2x4x1(0, 0, 4)
        
        return U, V, W

    U36, V36, W36 = get_initial_solution_rank36()
    
    def verify_solution_copy(U, V, W, tol=1e-9):
        T_copy = T.copy()
        R = U.shape[1]
        for mu in range(R):
            T_copy -= np.einsum('a,b,c->abc', U[:,mu], V[:,mu], W[:,mu])
        return np.max(np.abs(T_copy)) < tol

    def snap_to_half_integers(U, V, W):
        U_snapped = np.round(U * 2) / 2.0
        V_snapped = np.round(V * 2) / 2.0
        W_snapped = np.round(W * 2) / 2.0
        return U_snapped, V_snapped, W_snapped

    def residual_vector_vectorized(x, R):
        """Compute residual using vectorized tensor operations."""
        size_U = n * m * R
        size_V = m * p * R
        
        U = x[0:size_U].reshape(n*m, R)
        V = x[size_U:size_U+size_V].reshape(m*p, R)
        W = x[size_U+size_V:].reshape(p*n, R)
        
        T_rec = np.einsum('am,bm,cm->abc', U, V, W, optimize=True)
        return (T_rec - T).flatten()
    
    def gradient_residual_vectorized(x, R):
        """Compute gradient of 0.5*||residual||^2 using vectorized operations."""
        size_U = n * m * R
        size_V = m * p * R
        
        U = x[0:size_U].reshape(n*m, R)
        V = x[size_U:size_U+size_V].reshape(m*p, R)
        W = x[size_U+size_V:].reshape(p*n, R)
        
        T_rec = np.einsum('am,bm,cm->abc', U, V, W, optimize=True)
        residual = T_rec - T
        
        grad_U = np.einsum('abc,bm,cm->am', residual, V, W, optimize=True)
        grad_V = np.einsum('abc,am,cm->bm', residual, U, W, optimize=True)
        grad_W = np.einsum('abc,am,bm->cm', residual, U, V, optimize=True)
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    def solve_for_W(U, V, R):
        """Solve for W given U, V: T[ai,bi,ci] = sum_r U[ai,r] V[bi,r] W[ci,r]"""
        mp = m * p
        # M[ai*mp+bi, r] = U[ai,r] * V[bi,r]
        M = np.einsum('ar,br->abr', U, V).reshape(n*m*mp, R)
        T_flat = T.reshape(n*m*mp, p*n)
        W_sol = np.zeros((R, p*n))
        # Solve M @ W[:,ci] = T_flat[:,ci] for each ci
        W_sol = np.linalg.lstsq(M, T_flat, rcond=None)[0].T
        return W_sol

    def solve_for_U(V, W, R):
        """Solve for U given V, W: T[ai,bi,ci] = sum_r U[ai,r] V[bi,r] W[ci,r]"""
        pn = p * n
        # M[bi*pn+ci, r] = V[bi,r] * W[ci,r]
        M = np.einsum('br,cr->bcr', V, W).reshape(m*p*pn, R)
        T_flat = T.reshape(n*m, m*p*pn)
        U_sol = np.zeros((n*m, R))
        for ai in range(n*m):
            U_sol[ai, :] = np.linalg.lstsq(M, T_flat[ai, :], rcond=None)[0]
        return U_sol

    def solve_for_V(U, W, R):
        """Solve for V given U, W: T[ai,bi,ci] = sum_r U[ai,r] V[bi,r] W[ci,r]"""
        pn = p * n
        # M[ai*pn+ci, r] = U[ai,r] * W[ci,r]
        M = np.einsum('ar,cr->acr', U, W).reshape(n*m*pn, R)
        T_flat = T.reshape(n*m, m*p*pn)
        V_sol = np.zeros((m*p, R))
        for bi in range(m*p):
            V_sol[bi, :] = np.linalg.lstsq(M, T_flat[:, bi*pn:(bi+1)*pn].flatten(), rcond=None)[0]
        return V_sol

    def alternating_linear_solve(U, V, W, R, n_iters=50):
        """Alternating linear solve: fix two factors, solve for third."""
        for it in range(n_iters):
            W = solve_for_W(U, V, R)
            U = solve_for_U(V, W, R)
            V = solve_for_V(U, W, R)
        return U, V, W

    def compute_residual(U, V, W):
        T_rec = np.einsum('am,bm,cm->abc', U, V, W, optimize=True)
        return np.max(np.abs(T_rec - T))

    start_time = time.time()
    time_limit = 135
    
    best_rank = 36
    best_result = None
    
    for R in range(35, 20, -1):
        if time.time() - start_time > time_limit:
            break
            
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        n_attempts = 30 if R >= 32 else 20
        
        for attempt in range(n_attempts):
            if time.time() - start_time > time_limit:
                break
            
            if attempt == 0 and R <= 36:
                U_init = U36[:, :R]
                V_init = V36[:, :R]
                W_init = W36[:, :R]
            elif attempt == 1:
                U_init = np.random.normal(0, 1.0, (n*m, R))
                V_init = np.random.normal(0, 1.0, (m*p, R))
                W_init = np.random.normal(0, 1.0, (p*n, R))
            elif attempt == 2:
                U_init = np.random.normal(0, 0.5, (n*m, R))
                V_init = np.random.normal(0, 0.5, (m*p, R))
                W_init = np.random.normal(0, 0.5, (p*n, R))
            elif attempt == 3:
                U_init = np.random.choice([-1, 0, 0, 0, 1], (n*m, R))
                V_init = np.random.choice([-1, 0, 0, 0, 1], (m*p, R))
                W_init = np.random.choice([-1, 0, 0, 0, 1], (p*n, R))
            elif attempt == 4:
                U_init = np.random.normal(0, 2.0, (n*m, R))
                V_init = np.random.normal(0, 2.0, (m*p, R))
                W_init = np.random.normal(0, 2.0, (p*n, R))
            elif attempt == 5:
                # Sparse random initialization
                U_init = np.zeros((n*m, R)); U_init[np.random.randint(0, n*m, R), np.random.randint(0, R, R)] = np.random.choice([-1, 1], R)
                V_init = np.zeros((m*p, R)); V_init[np.random.randint(0, m*p, R), np.random.randint(0, R, R)] = np.random.choice([-1, 1], R)
                W_init = np.zeros((p*n, R)); W_init[np.random.randint(0, p*n, R), np.random.randint(0, R, R)] = np.random.choice([-1, 1], R)
            else:
                scale = 0.3 + 0.2 * (attempt % 7)
                U_init = np.random.normal(0, scale, (n*m, R))
                V_init = np.random.normal(0, scale, (m*p, R))
                W_init = np.random.normal(0, scale, (p*n, R))
            
            # Phase 1: Alternating linear solve (fast, exploits linearity)
            try:
                U_alt, V_alt, W_alt = alternating_linear_solve(
                    U_init.copy(), V_init.copy(), W_init.copy(), R, 
                    n_iters=30
                )
                res_alt = compute_residual(U_alt, V_alt, W_alt)
                
                if res_alt < 1e-6:
                    U_snap, V_snap, W_snap = snap_to_half_integers(U_alt, V_alt, W_alt)
                    if compute_residual(U_snap, V_snap, W_snap) < 1e-9:
                        best_rank = R
                        best_result = (U_snap, V_snap, W_snap)
                        break
                elif res_alt < 1e-3:
                    # Phase 2: Polish with gradient-based optimization
                    x0 = np.concatenate([U_alt.flatten(), V_alt.flatten(), W_alt.flatten()])
                    result = opt.minimize(
                        lambda x: 0.5 * np.sum(residual_vector_vectorized(x, R)**2),
                        x0,
                        method='L-BFGS-B',
                        jac=lambda x: gradient_residual_vectorized(x, R),
                        maxiter=5000,
                        maxfun=10000,
                        options={'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
                    )
                    x_opt = result.x
                    U_opt = x_opt[0:size_U].reshape(n*m, R)
                    V_opt = x_opt[size_U:size_U+size_V].reshape(m*p, R)
                    W_opt = x_opt[size_U+size_V:].reshape(p*n, R)
                    max_res = compute_residual(U_opt, V_opt, W_opt)
                    
                    if max_res < 1e-8:
                        U_snap, V_snap, W_snap = snap_to_half_integers(U_opt, V_opt, W_opt)
                        if compute_residual(U_snap, V_snap, W_snap) < 1e-9:
                            best_rank = R
                            best_result = (U_snap, V_snap, W_snap)
                            break
                else:
                    # Try direct gradient optimization from original init
                    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                    result = opt.minimize(
                        lambda x: 0.5 * np.sum(residual_vector_vectorized(x, R)**2),
                        x0,
                        method='L-BFGS-B',
                        jac=lambda x: gradient_residual_vectorized(x, R),
                        maxiter=3000,
                        maxfun=6000,
                        options={'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                    )
                    x_opt = result.x
                    U_opt = x_opt[0:size_U].reshape(n*m, R)
                    V_opt = x_opt[size_U:size_U+size_V].reshape(m*p, R)
                    W_opt = x_opt[size_U+size_V:].reshape(p*n, R)
                    max_res = compute_residual(U_opt, V_opt, W_opt)
                    
                    if max_res < 1e-6:
                        U_snap, V_snap, W_snap = snap_to_half_integers(U_opt, V_opt, W_opt)
                        if compute_residual(U_snap, V_snap, W_snap) < 1e-9:
                            best_rank = R
                            best_result = (U_snap, V_snap, W_snap)
                            break
            except Exception:
                pass
    
    if best_result is not None:
        U_snap, V_snap, W_snap = best_result
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    U_snap, V_snap, W_snap = snap_to_half_integers(U36, V36, W36)
    T_rec = np.einsum('am,bm,cm->abc', U_snap, V_snap, W_snap, optimize=True)
    if np.max(np.abs(T_rec - T)) < 1e-9:
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}
