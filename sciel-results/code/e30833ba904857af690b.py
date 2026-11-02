import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 110.0
    
    n, m, p = 3, 3, 3
    nm, mp, pn = 9, 9, 9
    
    def verify(U, V, W, tol=1e-6):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for l in range(p):
                        s = np.sum(W[k*n+i, :] * U[i*m+j, :] * V[l*p+k, :])
                        expected = 1.0 if j == l else 0.0
                        if abs(s - expected) > tol:
                            return False
        return True
    
    # Schoolbook rank 27
    rank_sb = 27
    U = np.zeros((nm, rank_sb))
    V = np.zeros((mp, rank_sb))
    W = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
    best_rank = rank_sb
    
    # Try rank 23 with scipy
    if time.time() - start_time < deadline - 15:
        try:
            from scipy.optimize import fsolve
            r = 23
            np.random.seed(123)
            
            # Better initialization: start from schoolbook-like structure
            U_init = np.zeros((nm, r))
            V_init = np.zeros((mp, r))
            W_init = np.zeros((pn, r))
            
            # Fill with random small perturbations
            for idx in range(r):
                i = idx % 3
                j = (idx // 3) % 3
                k = (idx // 9) % 3
                if idx < 27:
                    U_init[i*m+j, idx] = 1.0
                    V_init[j*p+k, idx] = 1.0
                    W_init[k*n+i, idx] = 1.0
            
            # Add random noise
            U_init += np.random.randn(nm, r) * 0.1
            V_init += np.random.randn(mp, r) * 0.1
            W_init += np.random.randn(pn, r) * 0.1
            
            def equations(vars):
                U_f = vars[:nm*r].reshape(nm, r)
                V_f = vars[nm*r:2*nm*r].reshape(mp, r)
                W_f = vars[2*nm*r:3*nm*r].reshape(pn, r)
                eqs = []
                for i in range(n):
                    for k in range(p):
                        for j in range(m):
                            for l in range(p):
                                s = np.sum(W_f[k*n+i, :] * U_f[i*m+j, :] * V_f[l*p+k, :])
                                expected = 1.0 if j == l else 0.0
                                eqs.append(s - expected)
                return np.array(eqs)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            sol, info, ier, msg = fsolve(equations, x0, full_output=True, maxfev=5000)
            
            if ier == 1 and time.time() - start_time < deadline - 5:
                U_sol = sol[:nm*r].reshape(nm, r)
                V_sol = sol[nm*r:2*nm*r].reshape(mp, r)
                W_sol = sol[2*nm*r:3*nm*r].reshape(pn, r)
                
                # Round to half-integers
                U_r = np.round(U_sol * 2) / 2
                V_r = np.round(V_sol * 2) / 2
                W_r = np.round(W_sol * 2) / 2
                
                if verify(U_r, V_r, W_r, tol=1e-3):
                    best_U, best_V, best_W = U_r.tolist(), V_r.tolist(), W_r.tolist()
                    best_rank = r
        except Exception:
            pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
