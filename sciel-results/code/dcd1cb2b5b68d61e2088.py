import numpy as np
from scipy.optimize import minimize, linprog
import time

N = 26
START_TIME = time.time()
TIME_LIMIT = 145.0 # Leave 5s for cleanup

def construct():
    def objective(v):
        r = v[2*N:3*N]
        return -np.sum(r)
    
    def make_constraints():
        cons = []
        # Boundary
        def c_x_min(v): return v[:N] - v[2*N:3*N]
        cons.append({'type': 'ineq', 'fun': c_x_min})
        def c_x_max(v): return 1 - v[:N] - v[2*N:3*N]
        cons.append({'type': 'ineq', 'fun': c_x_max})
        def c_y_min(v): return v[N:2*N] - v[2*N:3*N]
        cons.append({'type': 'ineq', 'fun': c_y_min})
        def c_y_max(v): return 1 - v[N:2*N] - v[2*N:3*N]
        cons.append({'type': 'ineq', 'fun': c_y_max})
        
        # Disjoint
        for i in range(N):
            for j in range(i+1, N):
                def c_disjoint(v, i=i, j=j):
                    xi, xj = v[i], v[j]
                    yi, yj = v[N+i], v[N+j]
                    ri, rj = v[2*N+i], v[2*N+j]
                    return (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
                cons.append({'type': 'ineq', 'fun': c_disjoint})
        return cons

    constraints = make_constraints()
    bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N

    best_val = -np.inf
    best_v = None
    
    # Initial guesses
    # 1. Jittered Grid
    # 2. Random
    
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    for seed in seeds:
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
            
        np.random.seed(seed)
        
        # Generate initial v0
        # Try a grid-like start first
        # 5x5 grid = 25 points.
        # Let's place 26 points in a 6x5 grid (30 spots) randomly choosing 26?
        # Or just 5x5 + 1 center?
        
        # Let's use a simple random start with small radii
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.02, 0.08, N)
        v0 = np.concatenate([x, y, r])
        
        # Try to optimize
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 100, 'ftol': 1e-6})
            
            if res.success:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_v = res.x.copy()
                    
                    # LP Radius Growth
                    x_fix = best_v[:N]
                    y_fix = best_v[N:2*N]
                    
                    c = -np.ones(N)
                    A_ub_rows = []
                    b_ub_list = []
                    
                    for i in range(N):
                        row = np.zeros(N); row[i] = 1
                        A_ub_rows.append(row); b_ub_list.append(x_fix[i])
                        A_ub_rows.append(row.copy()); b_ub_list.append(1 - x_fix[i])
                        
                        row = np.zeros(N); row[i] = 1
                        A_ub_rows.append(row); b_ub_list.append(y_fix[i])
                        A_ub_rows.append(row.copy()); b_ub_list.append(1 - y_fix[i])
                        
                    for i in range(N):
                        for j in range(i+1, N):
                            d = np.sqrt((x_fix[i] - x_fix[j])**2 + (y_fix[i] - y_fix[j])**2)
                            row = np.zeros(N); row[i] = 1; row[j] = 1
                            A_ub_rows.append(row); b_ub_list.append(d)
                            
                    A_ub = np.array(A_ub_rows)
                    b_ub = np.array(b_ub_list)
                    bounds_lp = [(0, 0.5)] * N
                    
                    res_lp = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
                    
                    if res_lp.success:
                        r_opt = res_lp.x
                        sum_r_new = np.sum(r_opt)
                        sum_r_old = np.sum(best_v[2*N:3*N])
                        
                        if sum_r_new > sum_r_old:
                            best_v = np.concatenate([x_fix, y_fix, r_opt])
                            best_val = sum_r_new
                            
                            # Polish
                            if time.time() - START_TIME < TIME_LIMIT - 5:
                                try:
                                    res_polish = minimize(objective, best_v, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 50, 'ftol': 1e-8})
                                    if res_polish.success:
                                        val_polish = -res_polish.fun
                                        if val_polish > best_val:
                                            best_val = val_polish
                                            best_v = res_polish.x.copy()
                                except:
                                    pass
                                    
        except:
            pass

    if best_v is None:
        # Fallback
        side_x, side_y = 6, 5
        cells = []
        count = 0
        for i in range(side_x):
            for j in range(side_y):
                if count < N:
                    x = (i + 0.5) / side_x
                    y = (j + 0.5) / side_y
                    r = min(0.5 / side_x, 0.5 / side_y) - 0.01
                    cells.append([x, y, r])
                    count += 1
        best_v = np.array(cells)

    return best_v
