import numpy as np
import time
import scipy.optimize as opt

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n
    
    # Strassen steps for 2x2x2
    strassen_steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]
    
    U_strassen = np.zeros((4, 7))
    V_strassen = np.zeros((12, 7))
    W_strassen = np.zeros((12, 7))
    
    for s in range(7):
        step = strassen_steps[s]
        for l in range(4):
            U_strassen[l, s] = step['u'][l]
            b_global = [0, 1, 6, 7][l]
            V_strassen[b_global, s] = step['v'][l]
            c_global = [0, 2, 1, 3][l]
            W_strassen[c_global, s] = step['w'][l]

    # Target tensor for remaining 4 columns (indices 2,3,4,5)
    # T_rem has shape (4, 8, 8) corresponding to A (4), B cols 2-5 (8), C cols 2-5 (8)
    T_rem = np.zeros((4, 8, 8))
    for i in range(2):
        for j in range(2):
            for k_prime in range(4):
                a_idx = 2*i + j
                b_idx = 2*j + k_prime
                c_idx = 4*i + k_prime
                T_rem[a_idx, b_idx, c_idx] = 1.0

    def search_extra_terms(num_terms, u_candidates, seed):
        if time.time() > deadline - 10:
            return None
            
        # u_candidates: list of vectors of size 4
        # We fix U to be chosen from u_candidates? No, U is a matrix (4, num_terms).
        # We can optimize U, V, W.
        # To reduce search space, we can fix U to be a combination of u_candidates?
        # Or just optimize all.
        
        # Dimensions:
        # U: (4, num_terms)
        # V: (8, num_terms)
        # W: (8, num_terms)
        
        def objective(params):
            U = params[:4*num_terms].reshape(4, num_terms)
            V = params[4*num_terms:4*num_terms+8*num_terms].reshape(8, num_terms)
            W = params[4*num_terms+8*num_terms:].reshape(8, num_terms)
            
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            res = T_calc - T_rem
            return np.sum(res**2)

        bounds = [(-3, 3)] * (20 * num_terms)
        
        try:
            sol = opt.differential_evolution(
                objective, 
                bounds, 
                seed=seed, 
                maxiter=200, 
                tol=1e-8, 
                polish=True,
                update_interval=10,
                popsize=10
            )
            
            if sol.fun < 1e-8:
                U_extra = sol.x[:4*num_terms].reshape(4, num_terms)
                V_extra = sol.x[4*num_terms:4*num_terms+8*num_terms].reshape(8, num_terms)
                W_extra = sol.x[4*num_terms+8*num_terms:].reshape(8, num_terms)
                
                # Round to half-integers
                U_extra_r = np.round(U_extra * 2) / 2
                V_extra_r = np.round(V_extra * 2) / 2
                W_extra_r = np.round(W_extra * 2) / 2
                
                T_check = np.einsum('ar,br,cr->abc', U_extra_r, V_extra_r, W_extra_r)
                if np.allclose(T_check, T_rem, atol=1e-8):
                    return (U_extra_r, V_extra_r, W_extra_r)
        except Exception:
            pass
        return None

    # Try Rank 19 (12 extra terms)
    result_19 = None
    for seed in range(3):
        if time.time() > deadline - 30:
            break
        result_19 = search_extra_terms(12, None, seed)
        if result_19:
            break
            
    if result_19:
        U_extra, V_extra, W_extra = result_19
        U = np.zeros((4, 19))
        V = np.zeros((12, 19))
        W = np.zeros((12, 19))
        
        U[:, :7] = U_strassen
        V[:, :7] = V_strassen
        W[:, :7] = W_strassen
        
        # Map extra terms to global indices
        for r in range(12):
            for lb in range(8):
                val = V_extra[lb, r]
                if val != 0:
                    j = lb // 4
                    k_prime = lb % 4
                    k = 2 + k_prime
                    gb = 6*j + k
                    V[gb, 7+r] = val
                    
            for lc in range(8):
                val = W_extra[lc, r]
                if val != 0:
                    i = lc // 4
                    k_prime = lc % 4
                    k = 2 + k_prime
                    gc = 2*k + i
                    W[gc, 7+r] = val
        
        # Verify
        T_full = np.einsum('ar,br,cr->abc', U, V, W)
        T_exp = np.zeros((4, 12, 12))
        for i in range(2):
            for j in range(2):
                for k in range(6):
                    a = i * 2 + j
                    b = j * 6 + k
                    c = k * 2 + i
                    T_exp[a, b, c] = 1.0
                    
        if np.allclose(T_full, T_exp, atol=1e-8):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to Rank 21
    U = np.zeros((4, 21))
    V = np.zeros((12, 21))
    W = np.zeros((12, 21))
    
    for g in range(3):
        for s in range(7):
            r = 7 * g + s
            step = strassen_steps[s]
            for l in range(4):
                U[l, r] = step['u'][l]
                k0 = 2 * g
                k1 = 2 * g + 1
                b_map = [6*0+k0, 6*0+k1, 6*1+k0, 6*1+k1]
                V[b_map[l], r] = step['v'][l]
                c_map = [2*k0+0, 2*k1+0, 2*k0+1, 2*k1+1]
                W[c_map[l], r] = step['w'][l]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
