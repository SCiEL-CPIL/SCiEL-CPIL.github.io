import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target tensor T[a, b, c]
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    start_time = time.time()
    deadline = 115.0 # Safety margin

    # Schoolbook solution
    rank_schoolbook = n * m * p
    U_school = np.zeros((nm, rank_schoolbook))
    V_school = np.zeros((mp, rank_schoolbook))
    W_school = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1.0
                V_school[b, col] = 1.0
                W_school[c, col] = 1.0
                col += 1
                
    best_solution = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_schoolbook

    # Try to find lower rank
    # Start from 20, go up to 44
    ranks_to_try = list(range(20, 45))
    
    for R in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        max_restarts = 100
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            U = np.random.uniform(-1, 1, (nm, R))
            V = np.random.uniform(-1, 1, (mp, R))
            W = np.random.uniform(-1, 1, (pn, R))
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective_and_grad(params):
                U_flat = params[:nm*R]
                V_flat = params[nm*R:nm*R + mp*R]
                W_flat = params[nm*R + mp*R:]
                
                U_ = U_flat.reshape(nm, R)
                V_ = V_flat.reshape(mp, R)
                W_ = W_flat.reshape(pn, R)
                
                M = np.einsum('ar,br,cr->abc', U_, V_, W_, optimize=True)
                
                E = T - M
                data_err = np.sum(E**2)
                
                # Gradients
                # dJ/dU[a,r] = -2 * sum_{b,c} E[a,b,c] * V[b,r] * W[c,r]
                # temp_U[a, b, r] = sum_c E[a,b,c] W[c,r] -> no, sum_c E[a,b,c] W[c,r] is not right for U grad directly?
                # Let's derive carefully.
                # J = sum_{a,b,c} E_{abc}^2
                # dJ/dU_{ar} = sum_{a',b',c'} 2 E_{a'b'c'} (-delta_{aa'} delta_{bb'} delta_{cc'} V_{b'r} W_{c'r})
                # = -2 sum_{b,c} E_{a b c} V_{b r} W_{c r}
                
                # Compute term for U:
                # sum_{b,c} E_{a b c} V_{b r} W_{c r}
                # This is (E * V * W) contracted over b and c for each a, r.
                # Let's do it step by step.
                # temp1[a, b, r] = sum_c E[a, b, c] W[c, r]
                temp1 = np.einsum('abc,cr->abr', E, W_, optimize=True)
                grad_U = -2.0 * np.einsum('abr,br->ar', temp1, V_, optimize=True)
                
                # Similarly for V:
                # dJ/dV_{br} = -2 sum_{a,c} E_{a b c} U_{a r} W_{c r}
                temp2 = np.einsum('abc,ar->bcr', E, U_, optimize=True)
                grad_V = -2.0 * np.einsum('bcr,cr->br', temp2, W_, optimize=True)
                
                # Similarly for W:
                # dJ/dW_{cr} = -2 sum_{a,b} E_{a b c} U_{a r} V_{b r}
                temp3 = np.einsum('abc,ar->bcr', E, U_, optimize=True) # Reuse temp2? No, temp2 is sum_a E U.
                # Actually temp3 for W needs sum_{a,b} E U V.
                # Let's recompute or reuse.
                # temp3 = sum_a E[a,b,c] U[a,r] -> this is temp2[b,c,r]
                grad_W = -2.0 * np.einsum('bcr,br->cr', temp2, V_, optimize=True)
                
                grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
                
                return data_err, grad
            
            try:
                res = minimize(objective_and_grad, params, method='L-BFGS-B', jac=True, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            except Exception:
                continue
                
            if res.fun < 1e-6:
                U_opt = res.x[:nm*R].reshape(nm, R)
                V_opt = res.x[nm*R:nm*R + mp*R].reshape(mp, R)
                W_opt = res.x[nm*R + mp*R:].reshape(pn, R)
                
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                
                M_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded, optimize=True)
                
                if np.allclose(M_rounded, T, atol=1e-9):
                    best_rank = R
                    best_solution = {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                    found = True
                    break
                    
        if found:
            break

    return best_solution
