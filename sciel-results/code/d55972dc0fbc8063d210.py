import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_val = -1
    best_res = None

    ws = np.linspace(0.6, 1.4, 10)
    
    for W in ws:
        if time.time() > deadline:
            break
        H = 2.0 - W
        
        # Init
        m = int(np.sqrt(N * W / H))
        if m < 1: m = 1
        k = int(np.ceil(N / m))
        
        xs = np.linspace(0.5/m, 1 - 0.5/m, m) * W
        ys = np.linspace(0.5/k, 1 - 0.5/k, k) * H
        XX, YY = np.meshgrid(xs, ys)
        XX = XX.flatten()
        YY = YY.flatten()
        
        x0 = XX[:N]
        y0 = YY[:N]
        r0_val = min(W/m, H/k) * 0.4
        r0 = np.full(N, r0_val)
        
        v0 = np.concatenate([x0, y0, r0])
        
        # Penalty function
        def penalty(v, lam):
            x = v[:N]
            y = v[N:2*N]
            r = v[2*N:]
            
            # Objective
            obj = -np.sum(r)
            
            # Overlaps
            pen = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    rad_sum = r[i] + r[j]
                    if dist < rad_sum:
                        pen += (rad_sum - dist)**2
            
            # Bounds
            for i in range(N):
                if x[i] < r[i]:
                    pen += (r[i] - x[i])**2
                if x[i] > W - r[i]:
                    pen += (x[i] - (W - r[i]))**2
                if y[i] < r[i]:
                    pen += (r[i] - y[i])**2
                if y[i] > H - r[i]:
                    pen += (y[i] - (H - r[i]))**2
            
            return obj + lam * pen

        # Optimization loop with adaptive lambda
        lam = 10.0
        v_best = v0
        for _ in range(10): # Max lambda increases
            if time.time() > deadline: break
            
            # Define wrapper for minimize
            def obj_func(v):
                return penalty(v, lam)
                
            res = minimize(obj_func, v_best, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            
            v_best = res.x
            
            # Check if constraints are satisfied
            x = v_best[:N]
            y = v_best[N:2*N]
            r = v_best[2*N:]
            
            max_viol = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    if dist < r[i] + r[j]:
                        max_viol = max(max_viol, r[i] + r[j] - dist)
                
                if x[i] < r[i]: max_viol = max(max_viol, r[i] - x[i])
                if x[i] > W - r[i]: max_viol = max(max_viol, x[i] - (W - r[i]))
                if y[i] < r[i]: max_viol = max(max_viol, r[i] - y[i])
                if y[i] > H - r[i]: max_viol = max(max_viol, y[i] - (H - r[i]))
                
            if max_viol < 1e-5:
                break
            else:
                lam *= 5.0
                
        # Final check
        x = v_best[:N]
        y = v_best[N:2*N]
        r = v_best[2*N:]
        
        # Verify strict validity
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dist < r[i] + r[j] - 1e-7:
                    valid = False
            if x[i] < r[i] - 1e-7 or x[i] > W - r[i] + 1e-7 or y[i] < r[i] - 1e-7 or y[i] > H - r[i] + 1e-7:
                valid = False
                
        if valid:
            val = np.sum(r)
            if val > best_val:
                best_val = val
                best_res = np.column_stack((x, y, r))
                
    if best_res is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_res = np.array(cells[:N])
        
    return best_res
