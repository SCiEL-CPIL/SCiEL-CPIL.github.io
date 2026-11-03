import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for k in range(p):
            for b in range(m):
                T[i*m+b, b*p+k, k*n+i] = 1.0

    start_time = time.time()
    time_limit = 135  # Safety margin before 150s kill

    def check_and_snap(U, V, W, tol=1e-9):
        U_snap = np.round(U * 2) / 2.0
        V_snap = np.round(V * 2) / 2.0
        W_snap = np.round(W * 2) / 2.0
        T_rec_snap = np.einsum('am,bm,cm->abc', U_snap, V_snap, W_snap, optimize=True)
        if np.max(np.abs(T_rec_snap - T)) < tol:
            return U_snap, V_snap, W_snap
        return None

    # --- Phase 1: Construct schoolbook rank-40 solution ---
    def get_schoolbook_rank40():
        rank = 40
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for k in range(p):
                for b in range(m):
                    U[i*m+b, col] = 1
                    V[b*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank

    U40, V40, W40, _ = get_schoolbook_rank40()
    best_rank = 40
    best_result = (U40, V40, W40)

    # --- Phase 2: Optimization to find lower rank ---
    def try_reduce_rank(U_init, V_init, W_init, R, maxiter=5000, maxfun=10000):
        size_U = nm * R
        size_V = mp * R
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def objective(x):
            U = x[0:size_U].reshape(nm, R)
            V = x[size_U:size_U+size_V].reshape(mp, R)
            W = x[size_U+size_V:].reshape(pn, R)
            T_rec = np.einsum('am,bm,cm->abc', U, V, W, optimize=True)
            return 0.5 * np.sum((T_rec - T)**2)
        
        def gradient(x):
            U = x[0:size_U].reshape(nm, R)
            V = x[size_U:size_U+size_V].reshape(mp, R)
            W = x[size_U+size_V:].reshape(pn, R)
            T_rec = np.einsum('am,bm,cm->abc', U, V, W, optimize=True)
            residual = T_rec - T
            grad_U = np.einsum('abc,bm,cm->am', residual, V, W, optimize=True)
            grad_V = np.einsum('abc,am,cm->bm', residual, U, W, optimize=True)
            grad_W = np.einsum('abc,am,bm->cm', residual, U, V, optimize=True)
            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        try:
            result = opt.minimize(
                objective, x0, method='L-BFGS-B', jac=gradient,
                maxiter=maxiter, maxfun=maxfun,
                options={'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
            )
            x_opt = result.x
            U_opt = x_opt[0:size_U].reshape(nm, R)
            V_opt = x_opt[size_U:size_U+size_V].reshape(mp, R)
            W_opt = x_opt[size_U+size_V:].reshape(pn, R)
            T_rec = np.einsum('am,bm,cm->abc', U_opt, V_opt, W_opt, optimize=True)
            max_res = np.max(np.abs(T_rec - T))
            return U_opt, V_opt, W_opt, max_res
        except Exception:
            return None, None, None, np.inf

    # --- Phase 3: Search for lower rank ---
    # Main Search Loop
    for R in range(39, 24, -1):
        if time.time() - start_time > time_limit - 5:
            break
        
        found = False
        
        # Try multiple initializations
        inits = []
        
        # Init 1: Truncated schoolbook
        if R <= 40:
            inits.append((U40[:, :R].copy(), V40[:, :R].copy(), W40[:, :R].copy()))
        
        # Init 2: Random Gaussian
        np.random.seed(42 + R)
        inits.append((
            np.random.normal(0, 1, (nm, R)),
            np.random.normal(0, 1, (mp, R)),
            np.random.normal(0, 1, (pn, R))
        ))
        
        # Init 3: Sparse random
        U_s = np.zeros((nm, R))
        V_s = np.zeros((mp, R))
        W_s = np.zeros((pn, R))
        for r in range(R):
            idx_u = np.random.choice(nm, size=min(3, nm), replace=False)
            idx_v = np.random.choice(mp, size=min(3, mp), replace=False)
            idx_w = np.random.choice(pn, size=min(3, pn), replace=False)
            U_s[idx_u, r] = np.random.choice([-1, 1], size=len(idx_u))
            V_s[idx_v, r] = np.random.choice([-1, 1], size=len(idx_v))
            W_s[idx_w, r] = np.random.choice([-1, 1], size=len(idx_w))
        inits.append((U_s, V_s, W_s))
        
        for U_init, V_init, W_init in inits:
            if time.time() - start_time > time_limit - 5:
                break
            
            U_opt, V_opt, W_opt, max_res = try_reduce_rank(U_init, V_init, W_init, R, maxiter=3000, maxfun=6000)
            
            if U_opt is not None and max_res < 1e-7:
                res = check_and_snap(U_opt, V_opt, W_opt)
                if res is not None:
                    best_rank = R
                    best_result = res
                    found = True
                    break
        
        if found:
            break

    # Final result
    if best_result is not None:
        U_snap, V_snap, W_snap = best_result
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    return {"U": U40.tolist(), "V": V40.tolist(), "W": W40.tolist()}
