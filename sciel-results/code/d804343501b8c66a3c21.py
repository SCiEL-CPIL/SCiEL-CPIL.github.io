"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

import numpy as np
from scipy.optimize import least_squares

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p
    
    def residual_vectorized(params, R):
        U = params[:nm * R].reshape(nm, R)
        V = params[nm * R: nm * R + mp * R].reshape(mp, R)
        W = params[nm * R + mp * R: nm * R + mp * R + pn * R].reshape(pn, R)
        
        # For each (k,i), compute the bilinear form:
        # sum_r W_{ki,r} * (U^T A)_r * (V^T B)_r
        # and compare to the target bilinear form for C_{ki}
        
        # The target: C_{ki} = sum_j A_{ij} B_{jk}
        # As a bilinear form, this has coefficient 1 for (A_{ij}, B_{jk}) and 0 otherwise.
        
        # The computed form: sum_r W_{ki,r} * sum_{i',j'} U_{i'm+j',r} A_{i'j'} * sum_{j'',k''} V_{j''p+k'',r} B_{j''k''}
        # = sum_{i',j',j'',k''} (sum_r W_{ki,r} U_{i'm+j',r} V_{j''p+k'',r}) A_{i'j'} B_{j''k''}
        
        # So we need: sum_r W_{ki,r} U_{i'm+j',r} V_{j''p+k'',r} = delta_{i',i} delta_{j',j''} delta_{k'',k}
        
        # Let's compute the tensor T_{ki, i'j', j''k''} = sum_r W_{ki,r} U_{i'm+j',r} V_{j''p+k'',r}
        # and compare to the target.
        
        # T_{ki, i'j', j''k''} = sum_r W_{ki,r} U_{i'm+j',r} V_{j''p+k'',r}
        # This is like a tensor contraction.
        
        # Reshape: W is (pn, R) = (p*n, R), U is (nm, R) = (n*m, R), V is (mp, R) = (m*p, R)
        # T_{ki, i'j', j''k''} = sum_r W_{ki,r} U_{i'j',r} V_{j''k'',r}
        
        # We can compute this as: for each r, outer product of W[:,r], U[:,r], V[:,r]
        # T = sum_r W_r \otimes U_r \otimes V_r where W_r is (pn,), U_r is (nm,), V_r is (mp,)
        
        # The target tensor: Target_{ki, i'j', j''k''} = delta_{i',i} delta_{j',j''} delta_{k'',k}
        # This is 1 when i'=i, j'=j'', k''=k, else 0.
        
        # Let's build the target tensor as a (pn, nm, mp) array
        target = np.zeros((pn, nm, mp))
        for k in range(p):
            for i in range(n):
                for j in range(m):
                    ki = k * n + i
                    ij = i * m + j
                    jk = j * p + k
                    target[ki, ij, jk] = 1.0
        
        # Compute T: shape (pn, nm, mp)
        T = np.zeros((pn, nm, mp))
        for r in range(R):
            T += np.einsum('a,b,c->abc', W[:, r], U[:, r], V[:, r])
        
        return (T - target).flatten()
    
    best_solution = None
    best_rank = schoolbook_rank
    
    # Try ranks from 20 up to schoolbook
    for R in range(20, min(schoolbook_rank, 35) + 1):
        for trial in range(3):
            np.random.seed(R * 100 + trial)
            params0 = np.random.randn(nm * R + mp * R + pn * R) * 0.3
            
            def obj(params):
                return residual_vectorized(params, R)
            
            try:
                result = least_squares(obj, params0, method='lm', max_nfev=5000)
                res_norm = np.linalg.norm(result.fun)
                if res_norm < 1e-6:
                    U = result.x[:nm * R].reshape(nm, R)
                    V = result.x[nm * R: nm * R + mp * R].reshape(mp, R)
                    W = result.x[nm * R + mp * R: nm * R + mp * R + pn * R].reshape(pn, R)
                    
                    # Try rounding to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    res_r = residual_vectorized(np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()]), R)
                    if np.linalg.norm(res_r) < 1e-6:
                        best_solution = (U_r, V_r, W_r, R)
                    else:
                        best_solution = (U, V, W, R)
                    break
            except Exception:
                continue
        
        if best_solution is not None:
            best_rank = best_solution[3]
            if best_rank <= 25:
                break
    
    if best_solution is not None:
        U, V, W, R = best_solution
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }
    
    # Schoolbook fallback
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
