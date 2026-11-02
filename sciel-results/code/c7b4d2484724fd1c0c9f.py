import numpy as np
import time
import random

DIMENSIONS = [2, 4, 4]

def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i_out in range(n):
        for j in range(m):
            for k_out in range(p):
                T[i_out*m + j, j*p + k_out, k_out*n + i_out] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-10):
    R = U.shape[1]
    Tc = np.zeros_like(T)
    for r in range(R):
        Tc += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    return np.allclose(Tc, T, atol=tol)

def _strassen_terms_2x2():
    terms = []
    # M1
    u = {0: 1, 3: 1}
    v = {0: 1, 3: 1}
    w = {0: 1, 3: 1}
    terms.append((u, v, w))
    # M2
    u = {2: 1, 3: 1}
    v = {0: 1}
    w = {2: 1, 3: -1}
    terms.append((u, v, w))
    # M3
    u = {0: 1}
    v = {1: 1, 3: -1}
    w = {1: 1, 3: 1}
    terms.append((u, v, w))
    # M4
    u = {3: 1}
    v = {2: 1, 0: -1}
    w = {0: 1, 2: 1}
    terms.append((u, v, w))
    # M5
    u = {0: 1, 1: 1}
    v = {3: 1}
    w = {0: -1, 1: 1}
    terms.append((u, v, w))
    # M6
    u = {2: 1, 0: -1}
    v = {0: 1, 1: 1}
    w = {3: 1}
    terms.append((u, v, w))
    # M7
    u = {1: 1, 3: -1}
    v = {2: 1, 3: 1}
    w = {0: 1}
    terms.append((u, v, w))
    return terms

def _rank28_solution():
    n, m, p = DIMENSIONS
    R = 28
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)

    S = _strassen_terms_2x2()

    def mA(li, off):
        r, c = li // 2, li % 2
        return r * m + (off * 2 + c)

    def mB(li, ro, co):
        r, c = li // 2, li % 2
        return (ro * 2 + r) * p + (co * 2 + c)

    def mC(li, co):
        r, c = li // 2, li % 2
        return (co * 2 + c) * n + r

    blocks = [
        (0, 0, 0, 0, 0),
        (1, 1, 0, 0, 7),
        (0, 0, 1, 1, 14),
        (1, 1, 1, 1, 21),
    ]

    for a_off, b_ro, b_co, c_co, base in blocks:
        for j, (u_loc, v_loc, w_loc) in enumerate(S):
            ci = base + j
            for li, cf in u_loc.items():
                U[mA(li, a_off), ci] += cf
            for li, cf in v_loc.items():
                V[mB(li, b_ro, b_co), ci] += cf
            for li, cf in w_loc.items():
                W[mC(li, c_co), ci] += cf

    return U, V, W

def _try_rank27_optimization(U, V, W, T, deadline):
    n, m, p = DIMENSIONS
    R = 27
    # Initialize with random perturbations of the rank-28 solution (dropping one term)
    # We try dropping each of the 28 terms and then optimizing the remaining 27
    for drop_idx in range(28):
        if time.time() > deadline:
            break
        U0 = np.delete(U, drop_idx, axis=1)
        V0 = np.delete(V, drop_idx, axis=1)
        W0 = np.delete(W, drop_idx, axis=1)
        
        # Add small random noise to escape local minima
        U_rand = U0 + 0.1 * np.random.randn(n*m, R)
        V_rand = V0 + 0.1 * np.random.randn(m*p, R)
        W_rand = W0 + 0.1 * np.random.randn(p*n, R)
        
        # Use scipy.optimize.minimize to minimize the error
        # The error is the Frobenius norm of the difference between the tensor and the sum of rank-1 terms
        def error(params):
            U_p, V_p, W_p = params
            U_p = U_p.reshape(n*m, R)
            V_p = V_p.reshape(m*p, R)
            W_p = W_p.reshape(p*n, R)
            Tc = np.zeros((n*m, m*p, p*n))
            for r in range(R):
                Tc += np.einsum('i,j,k->ijk', U_p[:,r], V_p[:,r], W_p[:,r])
            return np.linalg.norm(Tc - T)
        
        def jac(params):
            # Jacobian is complex, we skip it and use numerical differentiation
            return None
            
        x0 = np.concatenate([U_rand.flatten(), V_rand.flatten(), W_rand.flatten()])
        
        try:
            from scipy.optimize import minimize
            res = minimize(error, x0, method='L-BFGS-B', jac=None, options={'maxiter': 100, 'gtol': 1e-12})
            if res.fun < 1e-10:
                U_opt = res.x[:n*m*R].reshape(n*m, R)
                V_opt = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_opt = res.x[n*m*R + m*p*R:].reshape(p*n, R)
                if _verify(U_opt, V_opt, W_opt, T):
                    return U_opt, V_opt, W_opt
        except Exception:
            pass
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    T = _build_tensor()
    
    # First, get the rank-28 solution
    U28, V28, W28 = _rank28_solution()
    
    # Verify the rank-28 solution
    if not _verify(U28, V28, W28, T):
        # Fallback to schoolbook rank 32
        R = 32
        U = np.zeros((n*m, R), dtype=float)
        V = np.zeros((m*p, R), dtype=float)
        W = np.zeros((p*n, R), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a rank-27 solution using optimization
    if time.time() < deadline:
        res27 = _try_rank27_optimization(U28, V28, W28, T, deadline)
        if res27 is not None:
            U27, V27, W27 = res27
            return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
    
    # Return the rank-28 solution if rank-27 is not found
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
