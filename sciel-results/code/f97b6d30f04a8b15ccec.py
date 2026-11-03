import time
import numpy as np
import random
import scipy.optimize

DIMENSIONS = [3, 3, 4]
DEADLINE = time.time() + 110

def construct():
    n, m, p = DIMENSIONS
    total_terms = n * m * p
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n

    # Build schoolbook
    U_school = np.zeros((U_rows, total_terms), dtype=float)
    V_school = np.zeros((V_rows, total_terms), dtype=float)
    W_school = np.zeros((W_rows, total_terms), dtype=float)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    def check_decomposition(Umat, Vmat, Wmat):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = Umat[i * m + j, :] * Vmat[j * p + k, :] * Wmat[k * n + i, :]
                    s = np.sum(prod)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def residual(params, R):
        # params is flattened U, V, W
        U_flat = params[:U_rows * R]
        V_flat = params[U_rows * R : U_rows * R + V_rows * R]
        W_flat = params[U_rows * R + V_rows * R :]
        
        U = U_flat.reshape((U_rows, R))
        V = V_flat.reshape((V_rows, R))
        W = W_flat.reshape((W_rows, R))
        
        res = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
                    res[idx] = np.sum(prod) - 1.0
                    idx += 1
        return res

    def try_rank(R, max_restarts=5):
        if time.time() > DEADLINE - 10:
            return None
            
        for _ in range(max_restarts):
            if time.time() > DEADLINE - 10:
                break
                
            # Select R columns from schoolbook
            # Ensure coverage of (i,j), (j,k), (k,i)
            terms = list(range(total_terms))
            random.shuffle(terms)
            selected = terms[:R]
            
            # Check coverage
            covered_ij = set()
            covered_jk = set()
            covered_ki = set()
            for t in selected:
                i = t // (m * p)
                rem = t % (m * p)
                j = rem // p
                k = rem % p
                covered_ij.add((i, j))
                covered_jk.add((j, k))
                covered_ki.add((k, i))
                
            if len(covered_ij) < n * m or len(covered_jk) < m * p or len(covered_ki) < p * n:
                continue
                
            # Initialize
            U0 = U_school[:, selected].flatten()
            V0 = V_school[:, selected].flatten()
            W0 = W_school[:, selected].flatten()
            x0 = np.concatenate([U0, V0, W0])
            
            # Optimize
            try:
                res = scipy.optimize.minimize(
                    residual, x0, args=(R,), method='L-BFGS-B', 
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                if res.success or np.linalg.norm(res.fun) < 1e-6:
                    x_opt = res.x
                    U_opt = x_opt[:U_rows * R].reshape((U_rows, R))
                    V_opt = x_opt[U_rows * R : U_rows * R + V_rows * R].reshape((V_rows, R))
                    W_opt = x_opt[U_rows * R + V_rows * R :].reshape((W_rows, R))
                    
                    # Snap to half-integers
                    U_snap = np.round(U_opt * 2) / 2
                    V_snap = np.round(V_opt * 2) / 2
                    W_snap = np.round(W_opt * 2) / 2
                    
                    if check_decomposition(U_snap, V_snap, W_snap):
                        return U_snap, V_snap, W_snap
            except Exception:
                pass
                
        return None

    # Try rank 28
    result = try_rank(28, max_restarts=3)
    if result:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try rank 30
    if time.time() < DEADLINE - 20:
        result = try_rank(30, max_restarts=3)
        if result:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
