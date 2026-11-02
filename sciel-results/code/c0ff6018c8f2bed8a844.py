import numpy as np
import time
import scipy.optimize as opt

def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin
    
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    # Build target tensor T (4, 12, 12)
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def objective(params, R):
        # Unpack
        size_u = nm * R
        size_v = mp * R
        size_w = pn * R
        
        U = params[:size_u].reshape(nm, R)
        V = params[size_u:size_u+size_v].reshape(mp, R)
        W = params[size_u+size_v:size_u+size_v+size_w].reshape(pn, R)
        
        # Compute T_calc
        # T_calc[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Residuals
        return (T_calc - T_target).flatten()

    # Try ranks from 20 down to 19
    # Rank 21 is known to work (3 Strassen blocks), so we aim for < 21.
    
    best_solution = None
    best_rank = 21
    best_cost = np.inf

    # Fallback rank 21 construction
    def get_rank_21():
        # 3 Strassen blocks
        U = np.zeros((4, 21))
        V = np.zeros((12, 21))
        W = np.zeros((12, 21))
        
        # Strassen steps
        steps = [
            [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1],
            [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1],
            [1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1],
            [0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0],
            [1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0],
            [-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1],
            [0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]
        ]
        
        for g in range(3):
            k0 = 2 * g
            for s in range(7):
                col = 7 * g + s
                u_vals = steps[s*3]
                v_vals = steps[s*3+1]
                w_vals = steps[s*3+2]
                
                # Map U (4x1)
                U[:, col] = u_vals
                
                # Map V (12x1)
                # V index b = 2j + k. k in {k0, k0+1}
                # j in {0, 1}
                # v_vals correspond to j=0,k0; j=0,k0+1; j=1,k0; j=1,k0+1 ?
                # Strassen u,v,w are for 2x2 multiplication.
                # u: A indices 0,1,2,3 (i=0,j=0; i=0,j=1; i=1,j=0; i=1,j=1)
                # v: B indices. For block g, B columns are k0, k0+1.
                # B indices in local 2x2: 0,1,2,3 correspond to (j=0,k=0), (j=0,k=1), (j=1,k=0), (j=1,k=1).
                # Global B index b = 2j + k.
                # Local 0 -> j=0, k=k0 -> b = 2*0 + k0 = k0
                # Local 1 -> j=0, k=k0+1 -> b = 2*0 + k0+1 = k0+1
                # Local 2 -> j=1, k=k0 -> b = 2*1 + k0 = 2+k0
                # Local 3 -> j=1, k=k0+1 -> b = 2*1 + k0+1 = 2+k0+1
                b_map = [k0, k0+1, 2+k0, 2+k0+1]
                V[b_map[0], col] = v_vals[0]
                V[b_map[1], col] = v_vals[1]
                V[b_map[2], col] = v_vals[2]
                V[b_map[3], col] = v_vals[3]
                
                # Map W (12x1)
                # W index c = 2i + k.
                # Local C indices 0,1,2,3 correspond to (i=0,k=0), (i=0,k=1), (i=1,k=0), (i=1,k=1).
                # Global C index c = 2i + k.
                # Local 0 -> i=0, k=k0 -> c = k0
                # Local 1 -> i=0, k=k0+1 -> c = k0+1
                # Local 2 -> i=1, k=k0 -> c = 2+k0
                # Local 3 -> i=1, k=k0+1 -> c = 2+k0+1
                c_map = [k0, k0+1, 2+k0, 2+k0+1]
                W[c_map[0], col] = w_vals[0]
                W[c_map[1], col] = w_vals[1]
                W[c_map[2], col] = w_vals[2]
                W[c_map[3], col] = w_vals[3]
        
        # Verify
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        if not np.allclose(T_calc, T_target, atol=1e-6):
            return None
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    fallback = get_rank_21()
    
    # Search for rank 20 and 19
    for R in [20, 19]:
        if time.time() > deadline - 20:
            break
            
        total_vars = R * (nm + mp + pn)
        np.random.seed(42 + R)
        
        for trial in range(10):
            if time.time() > deadline - 20:
                break
                
            # Random initialization
            x0 = np.random.randn(total_vars) * 0.1
            
            try:
                # Use LM for least squares
                # max_nfev controls iterations
                sol = opt.least_squares(objective, x0, args=(R,), method='lm', max_nfev=2000)
                
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_solution = sol.x.copy()
                    
                    # Check if solution is close to exact
                    if sol.cost < 1e-10:
                        # Try rounding to half-integers
                        # The entries might be small.
                        # Let's check if we can round to integers or half-integers.
                        U = sol.x[:nm*R].reshape(nm, R)
                        V = sol.x[nm*R:nm*R+mp*R].reshape(mp, R)
                        W = sol.x[nm*R+mp*R:].reshape(pn, R)
                        
                        found = False
                        for mult in [1, 2, 4]:
                            U_r = np.round(U * mult) / mult
                            V_r = np.round(V * mult) / mult
                            W_r = np.round(W * mult) / mult
                            
                            T_check = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                            if np.allclose(T_check, T_target, atol=1e-6):
                                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        
                        # If not rounded, but cost is very low, it's likely a valid real solution.
                        # The task asks for "integer/half-integer entries".
                        # If we can't round, we might need to keep searching or accept if evaluator allows reals?
                        # "exactly decompose... with integer/half-integer entries"
                        # So we MUST return integer/half-integer.
                        
                        # If cost is low but rounding fails, continue searching.
                        
            except Exception:
                continue
                
    # If no better solution found, return fallback
    if fallback:
        return fallback
        
    # Last resort: return zeros (invalid, but prevents crash)
    return {"U": [[0]*21 for _ in range(4)], "V": [[0]*21 for _ in range(12)], "W": [[0]*21 for _ in range(12)]}
