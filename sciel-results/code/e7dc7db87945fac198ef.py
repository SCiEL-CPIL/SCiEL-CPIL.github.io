DIMENSIONS = [2, 3, 4]

import time
import numpy as np
from scipy.optimize import minimize

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the list of (u_idx, v_idx, w_idx) for each tensor entry
    entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                entries.append((u_idx, v_idx, w_idx))
    
    num_entries = len(entries)  # 24
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(nm)]
    V_sb = [[0] * rank_sb for _ in range(mp)]
    W_sb = [[0] * rank_sb for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Try to find lower rank using optimization + snapping to half-integers
    # Strategy: parameterize U, V, W as continuous variables, minimize sum of squares
    # of (sum_r U[u,r]*V[v,r]*W[w,r] - 1)^2, then snap to nearest half-integers
    
    def make_objective(params, target_rank):
        """params: flattened (nm + mp + pn) * target_rank vector"""
        U = params[:nm * target_rank].reshape(nm, target_rank)
        V = params[nm * target_rank : nm * target_rank + mp * target_rank].reshape(mp, target_rank)
        W = params[nm * target_rank + mp * target_rank :].reshape(pn, target_rank)
        
        err = 0.0
        for (u_idx, v_idx, w_idx) in entries:
            s = 0.0
            for r in range(target_rank):
                s += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
            err += (s - 1.0) ** 2
        return err
    
    def snap_to_half_int(x):
        """Snap to nearest half-integer"""
        return np.round(2.0 * x) / 2.0
    
    def verify_solution(U, V, W, target_rank):
        """Check if the snapped solution is exact"""
        for (u_idx, v_idx, w_idx) in entries:
            s = 0.0
            for r in range(target_rank):
                s += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
            if abs(s - 1.0) > 1e-9:
                return False
        return True
    
    np.random.seed(42)
    
    # Try ranks from 18 upward (known lower bound for 2x3x4 is around 18-19)
    for target_rank in range(18, best_rank + 1):
        if time.time() > deadline - 15:
            break
        
        found = False
        # Multiple random restarts
        num_restarts = 30
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random values
            scale = 1.0
            U_init = np.random.uniform(-scale, scale, size=(nm, target_rank))
            V_init = np.random.uniform(-scale, scale, size=(mp, target_rank))
            W_init = np.random.uniform(-scale, scale, size=(pn, target_rank))
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use Nelder-Mead for the non-smooth-ish objective (no gradient)
            # Actually, let's use L-BFGS-B with a smooth objective
            try:
                result = minimize(
                    make_objective,
                    params_init,
                    args=(target_rank,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
            except Exception:
                continue
            
            if result.fun < 1e-8:
                # Snap to half-integers
                U_opt = result.x[:nm * target_rank].reshape(nm, target_rank)
                V_opt = result.x[nm * target_rank : nm * target_rank + mp * target_rank].reshape(mp, target_rank)
                W_opt = result.x[nm * target_rank + mp * target_rank :].reshape(pn, target_rank)
                
                U_snapped = snap_to_half_int(U_opt)
                V_snapped = snap_to_half_int(V_opt)
                W_snapped = snap_to_half_int(W_opt)
                
                if verify_solution(U_snapped, V_snapped, W_snapped, target_rank):
                    # Convert to lists of lists with integer/half-integer values
                    U_list = [[float(x) for x in row] for row in U_snapped]
                    V_list = [[float(x) for x in row] for row in V_snapped]
                    W_list = [[float(x) for x in row] for row in W_snapped]
                    
                    # Clean up: convert 0.5 to 0.5, 1.0 to 1.0, etc.
                    # Represent as fractions for exactness
                    U_list = [[(int(x*2) / 2) for x in row] for row in U_snapped]
                    V_list = [[(int(x*2) / 2) for x in row] for row in V_snapped]
                    W_list = [[(int(x*2) / 2) for x in row] for row in W_snapped]
                    
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            # Found a solution at this rank, try to go lower
            # But since we're iterating upward, we break and return
            break
    
    return best_result
