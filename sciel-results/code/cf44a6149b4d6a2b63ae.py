import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Dimensions of the factor matrices
    dim_u = n * m  # 9
    dim_v = m * p  # 15
    dim_w = p * n  # 15
    
    # Precompute the index mapping for the equations
    eqs = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                eqs.append((u_idx, v_idx, w_idx))
                
    num_eqs = len(eqs)
    
    # We search for R = 44
    R_target = 44
    
    # Define objective function based on gauge fixing
    # gauge: 0 -> fix U[:, 0] = 1
    #        1 -> fix V[:, 0] = 1
    #        2 -> fix W[:, 0] = 1
    
    def make_objective(gauge):
        def objective(params, R):
            if gauge == 0:
                # U has dim_u, fix col 0. Vars: R*(dim_u-1 + dim_v + dim_w)
                U_rest = params[:R*(dim_u-1)].reshape(R, dim_u-1)
                V = params[R*(dim_u-1):R*(dim_u-1+dim_v)].reshape(R, dim_v)
                W = params[R*(dim_u-1+dim_v):].reshape(R, dim_w)
                
                U = np.zeros((R, dim_u))
                U[:, 0] = 1.0
                U[:, 1:] = U_rest
                
            elif gauge == 1:
                # V has dim_v, fix col 0. Vars: R*(dim_u + dim_v-1 + dim_w)
                U = params[:R*dim_u].reshape(R, dim_u)
                V_rest = params[R*dim_u:R*dim_u+R*(dim_v-1)].reshape(R, dim_v-1)
                W = params[R*dim_u+R*(dim_v-1):].reshape(R, dim_w)
                
                V = np.zeros((R, dim_v))
                V[:, 0] = 1.0
                V[:, 1:] = V_rest
                
            else: # gauge == 2
                # W has dim_w, fix col 0. Vars: R*(dim_u + dim_v + dim_w-1)
                U = params[:R*dim_u].reshape(R, dim_u)
                V = params[R*dim_u:R*dim_u+R*dim_v].reshape(R, dim_v)
                W_rest = params[R*dim_u+R*dim_v:R*dim_u+R*dim_v+R*(dim_w-1)].reshape(R, dim_w-1)
                
                W = np.zeros((R, dim_w))
                W[:, 0] = 1.0
                W[:, 1:] = W_rest
                
            residuals = []
            for u_idx, v_idx, w_idx in eqs:
                val = np.sum(U[:, u_idx] * V[:, v_idx] * W[:, w_idx])
                residuals.append(val - 1.0)
            return residuals
        return objective

    # Try to find a valid solution
    for gauge in range(3):
        if time.time() > deadline - 15:
            break
            
        objective = make_objective(gauge)
        
        # Calculate number of variables
        if gauge == 0:
            num_vars = R_target * (dim_u - 1 + dim_v + dim_w)
        elif gauge == 1:
            num_vars = R_target * (dim_u + dim_v - 1 + dim_w)
        else:
            num_vars = R_target * (dim_u + dim_v + dim_w - 1)
            
        # Multiple restarts
        max_restarts = 20
        for attempt in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialization
            x0 = np.random.uniform(-1, 1, num_vars)
            
            try:
                result = least_squares(objective, x0, args=(R_target,), method='lm', max_nfev=5000)
                
                if result.cost < 1e-10:
                    # Reconstruct matrices
                    if gauge == 0:
                        U_rest = result.x[:R_target*(dim_u-1)].reshape(R_target, dim_u-1)
                        V = result.x[R_target*(dim_u-1):R_target*(dim_u-1+dim_v)].reshape(R_target, dim_v)
                        W = result.x[R_target*(dim_u-1+dim_v):].reshape(R_target, dim_w)
                        
                        U = np.zeros((R_target, dim_u))
                        U[:, 0] = 1.0
                        U[:, 1:] = U_rest
                        
                    elif gauge == 1:
                        U = result.x[:R_target*dim_u].reshape(R_target, dim_u)
                        V_rest = result.x[R_target*dim_u:R_target*dim_u+R_target*(dim_v-1)].reshape(R_target, dim_v-1)
                        W = result.x[R_target*dim_u+R_target*(dim_v-1):].reshape(R_target, dim_w)
                        
                        V = np.zeros((R_target, dim_v))
                        V[:, 0] = 1.0
                        V[:, 1:] = V_rest
                        
                    else:
                        U = result.x[:R_target*dim_u].reshape(R_target, dim_u)
                        V = result.x[R_target*dim_u:R_target*dim_u+R_target*dim_v].reshape(R_target, dim_v)
                        W_rest = result.x[R_target*dim_u+R_target*dim_v:R_target*dim_u+R_target*dim_v+R_target*(dim_w-1)].reshape(R_target, dim_w-1)
                        
                        W = np.zeros((R_target, dim_w))
                        W[:, 0] = 1.0
                        W[:, 1:] = W_rest
                        
                    # Snap to half-integers
                    U_int = np.round(2 * U)
                    V_int = np.round(2 * V)
                    W_int = np.round(2 * W)
                    
                    # Verify exact integer solution for scaled tensor (target sum 8)
                    valid = True
                    for u_idx, v_idx, w_idx in eqs:
                        val = np.sum(U_int[:, u_idx] * V_int[:, v_idx] * W_int[:, w_idx])
                        if val != 8:
                            valid = False
                            break
                    
                    if valid:
                        U_final = (U_int / 2.0).tolist()
                        V_final = (V_int / 2.0).tolist()
                        W_final = (W_int / 2.0).tolist()
                        return {"U": U_final, "V": V_final, "W": W_final}
                        
            except Exception:
                continue

    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
