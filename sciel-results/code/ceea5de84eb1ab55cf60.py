"""Circle packing in perimeter-4 rectangle: analytic-gradient SLSQP + multi-start."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _penalty_and_grad(x):
    """
    Objective: Minimize -sum(r)
    Constraints (ineq >= 0):
    1. W >= 0.1
    2. W <= 1.9
    3. cx_i - r_i >= 0
    4. W - cx_i - r_i >= 0
    5. cy_i - r_i >= 0
    6. 2 - W - cy_i - r_i >= 0
    7. (cx_i - cx_j)^2 + (cy_i - cy_j)^2 - (r_i + r_j)^2 >= 0
    """
    H = 2.0 - x[0]
    if H <= 0 or x[0] <= 0:
        return 1e10, np.zeros_like(x), []
    
    cx = x[1:1+N]
    cy = x[1+N:1+2*N]
    r = x[1+2*N:1+3*N]
    
    obj = -np.sum(r)
    grad_obj = np.zeros_like(x)
    grad_obj[1+2*N:1+3*N] = -1.0
    
    cons = []
    
    # 1. W >= 0.1
    cons.append({'type': 'ineq', 'fun': lambda x: x[0] - 0.1, 
                 'jac': lambda x: np.array([1.0] + [0.0]*(3*N-1))})
    # 2. W <= 1.9
    cons.append({'type': 'ineq', 'fun': lambda x: 1.9 - x[0], 
                 'jac': lambda x: np.array([-1.0] + [0.0]*(3*N-1))})
    
    for i in range(N):
        idx_cx = 1 + i
        idx_cy = 1 + N + i
        idx_r = 1 + 2*N + i
        
        # cx - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[1+i] - x[1+2*N+i], 
                     'jac': lambda x, i=i: np.array([0.0] + [1.0 if k==1+i else 0.0 for k in range(1, 1+N)] + [0.0]*N + [-1.0 if k==1+2*N+i else 0.0 for k in range(1+2*N, 1+3*N)])})
        
        # W - cx - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[0] - x[1+i] - x[1+2*N+i], 
                     'jac': lambda x, i=i: np.array([1.0] + [-1.0 if k==1+i else 0.0 for k in range(1, 1+N)] + [0.0]*N + [-1.0 if k==1+2*N+i else 0.0 for k in range(1+2*N, 1+3*N)])})
        
        # cy - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[1+N+i] - x[1+2*N+i], 
                     'jac': lambda x, i=i: np.array([0.0] + [0.0]*N + [1.0 if k==1+N+i else 0.0 for k in range(1+N, 1+2*N)] + [-1.0 if k==1+2*N+i else 0.0 for k in range(1+2*N, 1+3*N)])})
        
        # H - cy - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 2.0 - x[0] - x[1+N+i] - x[1+2*N+i], 
                     'jac': lambda x, i=i: np.array([-1.0] + [0.0]*N + [-1.0 if k==1+N+i else 0.0 for k in range(1+N, 1+2*N)] + [-1.0 if k==1+2*N+i else 0.0 for k in range(1+2*N, 1+3*N)])})
    
    for i in range(N):
        for j in range(i+1, N):
            idx_cxi = 1 + i
            idx_cyi = 1 + N + i
            idx_ri = 1 + 2*N + i
            idx_cxj = 1 + j
            idx_cyj = 1 + N + j
            idx_rj = 1 + 2*N + j
            
            def fun_cc(x, i=i, j=j):
                dx = x[idx_cxi] - x[idx_cxj]
                dy = x[idx_cyi] - x[idx_cyj]
                dr = x[idx_ri] + x[idx_rj]
                return dx*dx + dy*dy - dr*dr
            
            def jac_cc(x, i=i, j=j):
                dx = x[idx_cxi] - x[idx_cxj]
                dy = x[idx_cyi] - x[idx_cyj]
                dr = x[idx_ri] + x[idx_rj]
                jac = np.zeros(3*N + 1)
                jac[idx_cxi] = 2*dx
                jac[idx_cxj] = -2*dx
                jac[idx_cyi] = 2*dy
                jac[idx_cyj] = -2*dy
                jac[idx_ri] = -2*dr
                jac[idx_rj] = -2*dr
                return jac
            
            cons.append({'type': 'ineq', 'fun': fun_cc, 'jac': jac_cc})
            
    return obj, grad_obj, cons

def _optimize_from(x0, deadline):
    """Run SLSQP from initial guess x0."""
    try:
        res = minimize(
            lambda x: _penalty_and_grad(x)[0],
            x0, 
            method='SLSQP', 
            jac=lambda x: _penalty_and_grad(x)[1],
            constraints=lambda x: _penalty_and_grad(x)[2],
            bounds=[(0.1, 1.9)] + [(0, 2)] * (2*N) + [(0, 1)] * N,
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        if res.success:
            return res.x, -res.fun
        else:
            val = -res.fun
            x0_val = -sum(x0[1+2*N:1+3*N])
            if val > x0_val:
                return res.x, val
            else:
                return x0, x0_val
    except Exception:
        return x0, -sum(x0[1+2*N:1+3*N])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_val = 0.0
    best_x = None
    
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 0.4, 0.9, 1.1]
    num_starts = 0
    
    for W_init in aspect_ratios:
        if time.time() > deadline:
            break
            
        H_init = 2.0 - W_init
        if H_init <= 0.1:
            continue
            
        for seed in [0, 1, 2]:
            if time.time() > deadline:
                break
                
            rng = np.random.default_rng(seed)
            
            # Grid init
            side = int(np.ceil(np.sqrt(N)))
            cx = np.array([(i % side + 0.5) / side * W_init for i in range(N)])
            cy = np.array([(i // side + 0.5) / side * H_init for i in range(N)])
            r = np.full(N, min(W_init / side, H_init / side) * 0.4)
            
            x0 = np.concatenate(([W_init], cx, cy, r))
            x_opt, val = _optimize_from(x0, deadline)
            
            if val > best_val:
                best_val = val
                best_x = x_opt
            
            # Random init
            cx_rand = rng.uniform(0.1*W_init, 0.9*W_init, N)
            cy_rand = rng.uniform(0.1*H_init, 0.9*H_init, N)
            r_rand = np.full(N, min(W_init, H_init) / 2 / np.sqrt(N) * 0.5)
            
            x0_rand = np.concatenate(([W_init], cx_rand, cy_rand, r_rand))
            x_opt_rand, val_rand = _optimize_from(x0_rand, deadline)
            
            if val_rand > best_val:
                best_val = val_rand
                best_x = x_opt_rand
                
            num_starts += 1
            if num_starts > 20:
                break
        if num_starts > 20:
            break

    if best_x is None:
        W = 1.0
        H = 1.0
        side = int(np.ceil(np.sqrt(N)))
        cx = np.array([(i % side + 0.5) / side * W for i in range(N)])
        cy = np.array([(i // side + 0.5) / side * H for i in range(N)])
        r = np.full(N, 0.5 / side * 0.9)
        return np.column_stack([cx, cy, r])
    
    W_opt = best_x[0]
    cx = best_x[1:1+N]
    cy = best_x[1+N:1+2*N]
    r = best_x[1+2*N:1+3*N]
    
    return np.column_stack([cx, cy, r])
