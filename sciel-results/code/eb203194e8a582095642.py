import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # Wait, the indices for W are (k*n + i).
    # The tensor entry T[a, b, c] is 1 if a=(i,j), b=(j,k), c=(k,i)
    # So T[i*m+j, j*p+k, k*n+i] = 1.
    
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0

    def tensor_product(U, V, W):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Result: (n*m, m*p, p*n)
        # R[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        # This can be done efficiently with einsum
        return np.einsum('ar,br,cr->abc', U, V, W)

    def objective(params, r):
        # params is a flat vector of U, V, W
        U = params[0:n*m*r].reshape(n*m, r)
        V = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
        W = params[n*m*r + m*p*r:].reshape(p*n, r)
        R = tensor_product(U, V, W)
        return np.sum((R - T)**2)

    def snap_to_half_integer(x, tol=1e-6):
        # Check if x is close to half-integers
        # Half integers are k/2 for integer k
        # So 2*x should be close to integer
        x2 = 2 * x
        rounded = np.round(x2)
        if np.max(np.abs(x2 - rounded)) < tol:
            return rounded / 2.0
        return None

    def verify(U, V, W):
        R = tensor_product(U, V, W)
        return np.allclose(R, T, atol=1e-9)

    start_time = time.time()
    deadline = 110.0  # 10s margin
    
    best_rank = n * m * p
    best_decomp = None
    
    # Schoolbook solution
    U_s = np.zeros((n*m, best_rank))
    V_s = np.zeros((m*p, best_rank))
    W_s = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1
                V_s[j*p+k, col] = 1
                W_s[k*n+i, col] = 1
                col += 1
    best_decomp = (U_s, V_s, W_s)

    # Try to find lower rank
    # Known rank for 2x3x5 is 28.
    # We try 28, then 29.
    for r in range(28, best_rank):
        if time.time() > deadline:
            break
            
        found = False
        # Number of restarts
        num_restarts = 50
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Scale matters. Start with small values.
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            # Simple gradient descent with momentum
            # Or use scipy.optimize
            # Since we need speed, let's try a few iterations of a simple solver
            # Actually, for exactness, random search might be too slow.
            # Let's use scipy.optimize.minimize with L-BFGS-B
            
            from scipy.optimize import minimize
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def obj(p):
                U_ = p[0:n*m*r].reshape(n*m, r)
                V_ = p[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_ = p[n*m*r + m*p*r:].reshape(p*n, r)
                R = tensor_product(U_, V_, W_)
                return np.sum((R - T)**2)
                
            def grad(p):
                U_ = p[0:n*m*r].reshape(n*m, r)
                V_ = p[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_ = p[n*m*r + m*p*r:].reshape(p*n, r)
                R = tensor_product(U_, V_, W_)
                diff = R - T
                # Gradient of sum(diff^2) wrt U[a, r]
                # d/du sum_{b,c} (sum_r' u[a, r'] v[b, r'] w[c, r'] - T[a,b,c])^2
                # = 2 * sum_{b,c} diff[a,b,c] * v[b, r] * w[c, r]
                gU = 2 * np.einsum('abc,br,cr->ar', diff, V_, W_)
                gV = 2 * np.einsum('abc,ar,cr->br', diff, U_, W_)
                gW = 2 * np.einsum('abc,ar,br->cr', diff, U_, V_)
                return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
            
            res = minimize(obj, params, jac=grad, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success or res.fun < 1e-8:
                U_opt = res.x[0:n*m*r].reshape(n*m, r)
                V_opt = res.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_opt = res.x[n*m*r + m*p*r:].reshape(p*n, r)
                
                # Try to snap
                U_snap = snap_to_half_integer(U_opt)
                V_snap = snap_to_half_integer(V_opt)
                W_snap = snap_to_half_integer(W_opt)
                
                if U_snap is not None and V_snap is not None and W_snap is not None:
                    if verify(U_snap, V_snap, W_snap):
                        best_rank = r
                        best_decomp = (U_snap, V_snap, W_snap)
                        found = True
                        break
                else:
                    # Maybe it's close enough to be considered valid if we just return floats?
                    # No, task requires integer/half-integer.
                    # But if verify passes on the raw floats, and they are close to half-integers,
                    # we might need to adjust.
                    if verify(U_opt, V_opt, W_opt):
                         # Check if they are effectively half-integers
                         if snap_to_half_integer(U_opt, 1e-5) is not None:
                             best_rank = r
                             best_decomp = (snap_to_half_integer(U_opt, 1e-5), 
                                            snap_to_half_integer(V_opt, 1e-5),
                                            snap_to_half_integer(W_opt, 1e-5))
                             found = True
                             break

        if found:
            break

    # Convert numpy arrays to lists of lists
    U_list = best_decomp[0].tolist()
    V_list = best_decomp[1].tolist()
    W_list = best_decomp[2].tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
