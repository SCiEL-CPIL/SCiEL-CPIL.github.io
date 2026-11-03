import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # Build the target tensor T[a, b, c, d, k, i] = delta_b_c * delta_a_i * delta_d_k
    # Shape: (n*m, m*p, p*n) -> (8, 20, 10)
    # Indexing:
    # U index: i*m + j  (i in 0..n-1, j in 0..m-1)
    # V index: j*p + k  (j in 0..m-1, k in 0..p-1)
    # W index: k*n + i  (k in 0..p-1, i in 0..n-1)
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i for some i,j,k.
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def get_residual(x, R):
        size_U = n * m * R
        size_V = m * p * R
        U = x[0:size_U].reshape(n*m, R)
        V = x[size_U:size_U+size_V].reshape(m*p, R)
        W = x[size_U+size_V:].reshape(p*n, R)
        
        # Reconstruct tensor
        # T_rec[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return T_rec - T

    def get_objective_and_grad(x, R):
        residual = get_residual(x, R)
        obj = 0.5 * np.sum(residual**2)
        
        size_U = n * m * R
        size_V = m * p * R
        U = x[0:size_U].reshape(n*m, R)
        V = x[size_U:size_U+size_V].reshape(m*p, R)
        W = x[size_U+size_V:].reshape(p*n, R)
        
        # Gradient of 0.5 * ||T_rec - T||^2
        # d/dU[a,r] = sum_{b,c} (T_rec[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
        # grad_U[a,r] = einsum('abc,br,cr->ar', residual, V, W)
        grad_U = np.einsum('abc,br,cr->ar', residual, V, W, optimize=True)
        grad_V = np.einsum('abc,ar,cr->br', residual, U, W, optimize=True)
        grad_W = np.einsum('abc,ar,br->cr', residual, U, V, optimize=True)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return obj, grad

    def snap_to_half_integers(U, V, W):
        U_snapped = np.round(U * 2) / 2.0
        V_snapped = np.round(V * 2) / 2.0
        W_snapped = np.round(W * 2) / 2.0
        return U_snapped, V_snapped, W_snapped

    def verify_solution(U, V, W, tol=1e-9):
        T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.max(np.abs(T_rec - T)) < tol

    start_time = time.time()
    time_limit = 115  # Safety margin
    
    best_rank = 40
    best_result = None
    
    # Schoolbook solution for fallback and initialization
    def get_schoolbook_solution(R):
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= R:
                        break
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
                if col >= R:
                    break
            if col >= R:
                break
        return U, V, W

    # Try ranks from 32 down to 28
    # 32 is a strong candidate based on 2*rank(2x2x5) = 2*16 = 32
    for R in [32, 31, 30, 29, 28]:
        if time.time() - start_time > time_limit:
            break
            
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        # Multiple attempts for each rank
        n_attempts = 15 if R <= 30 else 5
        
        for attempt in range(n_attempts):
            if time.time() - start_time > time_limit - 5:
                break
                
            # Initialization strategy
            if attempt == 0:
                # Initialize with schoolbook truncated to R, then randomize the rest?
                # Or just random.
                # For R=32, we can try to initialize with two blocks of 2x2x5 if we had them.
                # Since we don't, we use random.
                np.random.seed(42 + R * 100 + attempt)
                x0 = np.random.normal(0, 1.0, (size_U + size_V + size_W))
            elif attempt == 1:
                # Sparse random
                np.random.seed(1000 + R * 100 + attempt)
                x0 = np.random.choice([-1, 0, 0, 0, 1], (size_U + size_V + size_W))
            else:
                # Varying scale
                scale = 0.5 + 0.2 * (attempt % 3)
                np.random.seed(2000 + R * 100 + attempt)
                x0 = np.random.normal(0, scale, (size_U + size_V + size_W))
            
            try:
                # Use L-BFGS-B
                result = opt.minimize(
                    lambda x: get_objective_and_grad(x, R)[0],
                    x0,
                    jac=lambda x: get_objective_and_grad(x, R)[1],
                    method='L-BFGS-B',
                    maxiter=5000,
                    maxfun=10000,
                    options={'ftol': 1e-12, 'gtol': 1e-6, 'maxls': 50}
                )
                
                x_opt = result.x
                U_opt = x_opt[0:size_U].reshape(n*m, R)
                V_opt = x_opt[size_U:size_U+size_V].reshape(m*p, R)
                W_opt = x_opt[size_U+size_V:].reshape(p*n, R)
                
                T_rec = np.einsum('ar,br,cr->abc', U_opt, V_opt, W_opt, optimize=True)
                max_res = np.max(np.abs(T_rec - T))
                
                if max_res < 1e-4:
                    # Polish
                    result2 = opt.minimize(
                        lambda x: get_objective_and_grad(x, R)[0],
                        x_opt,
                        jac=lambda x: get_objective_and_grad(x, R)[1],
                        method='L-BFGS-B',
                        maxiter=10000,
                        maxfun=20000,
                        options={'ftol': 1e-15, 'gtol': 1e-8, 'maxls': 100}
                    )
                    
                    x_pol = result2.x
                    U_pol = x_pol[0:size_U].reshape(n*m, R)
                    V_pol = x_pol[size_U:size_U+size_V].reshape(m*p, R)
                    W_pol = x_pol[size_U+size_V:].reshape(p*n, R)
                    
                    T_rec_pol = np.einsum('ar,br,cr->abc', U_pol, V_pol, W_pol, optimize=True)
                    max_res_pol = np.max(np.abs(T_rec_pol - T))
                    
                    if max_res_pol < 1e-7:
                        # Try snapping to half-integers
                        U_snap, V_snap, W_snap = snap_to_half_integers(U_pol, V_pol, W_pol)
                        T_rec_snap = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)
                        if np.max(np.abs(T_rec_snap - T)) < 1e-9:
                            best_rank = R
                            best_result = (U_snap, V_snap, W_snap)
                            break
                        else:
                            # Maybe the solution is integer?
                            U_int = np.round(U_pol)
                            V_int = np.round(V_pol)
                            W_int = np.round(W_pol)
                            T_rec_int = np.einsum('ar,br,cr->abc', U_int, V_int, W_int, optimize=True)
                            if np.max(np.abs(T_rec_int - T)) < 1e-9:
                                best_rank = R
                                best_result = (U_int, V_int, W_int)
                                break
            except Exception:
                pass
                
        if best_result is not None:
            break

    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank 40
    U, V, W = get_schoolbook_solution(40)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
