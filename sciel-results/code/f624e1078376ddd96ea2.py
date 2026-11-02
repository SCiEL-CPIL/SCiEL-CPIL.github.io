import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def check_time():
        return time.time() > deadline

    def pack_hexagonal(rows):
        """
        Generate a hexagonal packing configuration.
        Returns: w, h, r, xs, ys
        """
        if sum(rows) != N:
            return None

        pts = []
        # Use r=1 for generation, scale later
        # Distance between centers is 2.
        # Row height is sqrt(3).
        
        for k, n_circles in enumerate(rows):
            y_k = k * np.sqrt(3)
            offset = 1.0 if k % 2 == 1 else 0.0
            for i in range(n_circles):
                x_k = i * 2.0 + offset
                pts.append((x_k, y_k))
        
        pts = np.array(pts)
        if len(pts) != N:
            return None

        # Center the points
        center = pts.mean(axis=0)
        pts -= center
        
        xs = pts[:, 0]
        ys = pts[:, 1]
        w_centers = xs.max() - xs.min()
        h_centers = ys.max() - ys.min()
        
        # Bounding box of circles (r=1)
        w_total = w_centers + 2.0
        h_total = h_centers + 2.0
        
        # Scale to fit perimeter 4 (w+h=2)
        # s * (w_total + h_total) = 2
        scale = 2.0 / (w_total + h_total)
        r = scale
        w = scale * w_total
        h = scale * h_total
        
        # Final coordinates
        xs_final = pts[:, 0] * scale + w/2.0
        ys_final = pts[:, 1] * scale + h/2.0
        
        return w, h, r, xs_final, ys_final

    def pack_grid(rows, cols):
        """
        Generate a grid packing configuration.
        """
        pts = []
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                pts.append((i * 2.0, j * 2.0))
                count += 1
            if count >= N:
                break
        
        pts = np.array(pts[:N])
        
        center = pts.mean(axis=0)
        pts -= center
        
        xs = pts[:, 0]
        ys = pts[:, 1]
        w_centers = xs.max() - xs.min()
        h_centers = ys.max() - ys.min()
        
        w_total = w_centers + 2.0
        h_total = h_centers + 2.0
        
        scale = 2.0 / (w_total + h_total)
        r = scale
        w = scale * w_total
        h = scale * h_total
        
        xs_final = pts[:, 0] * scale + w/2.0
        ys_final = pts[:, 1] * scale + h/2.0
        
        return w, h, r, xs_final, ys_final

    def refine_equal(w, h, r, xs, ys, is_polish=False):
        """
        Refine equal-radius packing using SLSQP.
        Variables: w, r, xs, ys. h = 2 - w.
        """
        if check_time():
            return w, h, r, xs, ys

        v0 = np.zeros(2 + 2*N)
        v0[0] = w
        v0[1] = r
        v0[2:2+N] = xs
        v0[2+N:2+2*N] = ys

        def objective(v):
            w_val = v[0]
            r_val = v[1]
            h_val = 2.0 - w_val
            if w_val <= 0.1 or h_val <= 0.1 or r_val <= 0:
                return 1e6
            return -r_val

        def constraints(v):
            w_val = v[0]
            r_val = v[1]
            h_val = 2.0 - w_val
            xs = v[2:2+N]
            ys = v[2+N:2+2*N]
            
            cons = []
            
            # Aspect ratio sanity
            cons.append({'type': 'ineq', 'fun': lambda v: v[0] - 0.2})
            cons.append({'type': 'ineq', 'fun': lambda v: 2.0 - v[0] - 0.2})
            
            # Boundary constraints
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2+i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[0] - v[2+i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2+N+i] - v[1]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: (2.0 - v[0]) - v[2+N+i] - v[1]})
                
            # Overlap constraints (squared distance)
            for i in range(N):
                for j in range(i+1, N):
                    cons.append({
                        'type': 'ineq', 
                        'fun': lambda v, i=i, j=j: ((v[2+i]-v[2+j])**2 + (v[2+N+i]-v[2+N+j])**2) - 4*(v[1]**2)
                    })
            return cons

        bounds = [(0.2, 1.8), (0.01, 1.0)] + [(0.0, 2.0)] * (2*N)

        try:
            if is_polish:
                options = {'maxiter': 200, 'ftol': 1e-12, 'disp': False}
            else:
                options = {'maxiter': 150, 'ftol': 1e-9, 'disp': False}

            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=constraints, 
                           options=options)
            
            if res.success or res.fun < -r:
                w_res = res.x[0]
                r_res = res.x[1]
                h_res = 2.0 - w_res
                xs_res = res.x[2:2+N]
                ys_res = res.x[2+N:2+2*N]
                
                # Validate
                valid = True
                if w_res <= 0 or h_res <= 0 or r_res <= 0:
                    valid = False
                else:
                    for i in range(N):
                        if xs_res[i] < r_res - 1e-7 or xs_res[i] > w_res - r_res + 1e-7:
                            valid = False
                            break
                        if ys_res[i] < r_res - 1e-7 or ys_res[i] > h_res - r_res + 1e-7:
                            valid = False
                            break
                    if valid:
                        for i in range(N):
                            for j in range(i+1, N):
                                dist_sq = (xs_res[i]-xs_res[j])**2 + (ys_res[i]-ys_res[j])**2
                                if dist_sq < 4*(r_res**2) - 1e-7:
                                    valid = False
                                    break
                            if not valid:
                                break
                
                if valid:
                    return w_res, h_res, r_res, xs_res, ys_res
        except Exception:
            pass
            
        return w, h, r, xs, ys

    def refine_non_uniform(w, h, r, xs, ys, is_polish=False):
        """
        Refine non-uniform radius packing using SLSQP.
        Variables: w, r_1..r_N, x_1..x_N, y_1..y_N.
        Objective: maximize sum(r_i) -> minimize -sum(r_i)
        """
        if check_time():
            return w, h, r, xs, ys

        # Initialize with equal radii
        r_init = np.full(N, r)
        
        v0 = np.zeros(1 + 3*N)
        v0[0] = w
        v0[1:1+N] = r_init
        v0[1+N:1+2*N] = xs
        v0[1+2*N:1+3*N] = ys

        def objective(v):
            w_val = v[0]
            h_val = 2.0 - w_val
            r_vals = v[1:1+N]
            if w_val <= 0.1 or h_val <= 0.1 or np.any(r_vals <= 0):
                return 1e6
            return -np.sum(r_vals)

        def constraints(v):
            w_val = v[0]
            h_val = 2.0 - w_val
            r_vals = v[1:1+N]
            xs = v[1+N:1+2*N]
            ys = v[1+2*N:1+3*N]
            
            cons = []
            
            # Aspect ratio sanity
            cons.append({'type': 'ineq', 'fun': lambda v: v[0] - 0.2})
            cons.append({'type': 'ineq', 'fun': lambda v: 2.0 - v[0] - 0.2})
            
            # Boundary constraints
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[1+N+i] - v[1+i]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[0] - v[1+N+i] - v[1+i]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[1+2*N+i] - v[1+i]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: (2.0 - v[0]) - v[1+2*N+i] - v[1+i]})
                
            # Overlap constraints (squared distance)
            for i in range(N):
                for j in range(i+1, N):
                    cons.append({
                        'type': 'ineq', 
                        'fun': lambda v, i=i, j=j: ((v[1+N+i]-v[1+N+j])**2 + (v[1+2*N+i]-v[1+2*N+j])**2) - (v[1+i]+v[1+j])**2
                    })
            return cons

        bounds = [(0.2, 1.8)] + [(0.01, 1.0)] * N + [(0.0, 2.0)] * (2*N)

        try:
            if is_polish:
                options = {'maxiter': 300, 'ftol': 1e-12, 'disp': False}
            else:
                options = {'maxiter': 200, 'ftol': 1e-9, 'disp': False}

            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=constraints, 
                           options=options)
            
            if res.success:
                w_res = res.x[0]
                h_res = 2.0 - w_res
                r_res = res.x[1:1+N]
                xs_res = res.x[1+N:1+2*N]
                ys_res = res.x[1+2*N:1+3*N]
                
                # Validate
                valid = True
                if w_res <= 0 or h_res <= 0 or np.any(r_res <= 0):
                    valid = False
                else:
                    for i in range(N):
                        if xs_res[i] < r_res[i] - 1e-7 or xs_res[i] > w_res - r_res[i] + 1e-7:
                            valid = False
                            break
                        if ys_res[i] < r_res[i] - 1e-7 or ys_res[i] > h_res - r_res[i] + 1e-7:
                            valid = False
                            break
                    if valid:
                        for i in range(N):
                            for j in range(i+1, N):
                                dist_sq = (xs_res[i]-xs_res[j])**2 + (ys_res[i]-ys_res[j])**2
                                if dist_sq < (r_res[i]+r_res[j])**2 - 1e-7:
                                    valid = False
                                    break
                            if not valid:
                                break
                
                if valid:
                    # Return in a format compatible with the main loop
                    # We need to return w, h, r_avg, xs, ys? 
                    # The main loop expects a single 'r' for equal radius logic, 
                    # but we want to keep non-uniform results.
                    # Let's modify the return to indicate non-uniform if needed, 
                    # or just store the full vector.
                    # For simplicity in the main loop, let's return the best sum of radii
                    # and store the full configuration.
                    return w_res, h_res, np.mean(r_res), xs_res, ys_res, r_res
        except Exception:
            pass
            
        return w, h, r, xs, ys, None

    best_sum_r = 0
    best_config = None # (w, h, xs, ys, r_array)

    def update_best(w, h, xs, ys, r_array):
        nonlocal best_sum_r, best_config
        s = np.sum(r_array)
        if s > best_sum_r:
            best_sum_r = s
            best_config = (w, h, xs, ys, r_array)

    # List of initial configurations
    configs = []
    
    # Hexagonal variations
    configs.append(pack_hexagonal([7, 7, 7]))
    configs.append(pack_hexagonal([4, 5, 4, 4, 4]))
    configs.append(pack_hexagonal([3, 4, 4, 4, 4, 2]))
    configs.append(pack_hexagonal([5, 5, 5, 3, 3]))
    configs.append(pack_hexagonal([6, 6, 5, 4]))
    
    # Grid variations
    configs.append(pack_grid(5, 5))
    configs.append(pack_grid(3, 7))
    configs.append(pack_grid(4, 6))
    configs.append(pack_grid(7, 3))
    configs.append(pack_grid(6, 4))
    
    # Random restarts
    for _ in range(10):
        if check_time():
            break
        w = 1.0
        h = 1.0
        r = 0.08
        xs = np.random.uniform(r, w-r, N)
        ys = np.random.uniform(r, h-r, N)
        configs.append((w, h, r, xs, ys))

    # Filter out invalid/None configs
    valid_configs = [c for c in configs if c is not None]

    # First pass: Coarse optimization (Equal Radius)
    for cfg in valid_configs:
        if check_time():
            break
        w, h, r, xs, ys = cfg
        w_new, h_new, r_new, xs_new, ys_new = refine_equal(w, h, r, xs, ys, is_polish=False)
        
        if r_new > 0:
            # Update best with equal radius array
            r_array = np.full(N, r_new)
            update_best(w_new, h_new, xs_new, ys_new, r_array)

    # Second pass: Non-uniform refinement on top configs
    # We need to keep track of the top few equal-radius configs to refine them non-uniformly
    # For simplicity, let's just take the best equal-radius config and the first few
    top_configs = []
    # Re-run or store? Storing is better. Let's modify the loop above to store top 5.
    # To save code complexity, I'll just re-identify the best one and a couple of others
    # Or better, store the results from the first pass.
    
    # Let's refactor slightly to store top results from pass 1
    pass1_results = []
    for cfg in valid_configs[:10]: # Only refine top 10 distinct starts
        if check_time():
            break
        w, h, r, xs, ys = cfg
        w_new, h_new, r_new, xs_new, ys_new = refine_equal(w, h, r, xs, ys, is_polish=False)
        if r_new > 0:
            r_array = np.full(N, r_new)
            s = np.sum(r_array)
            pass1_results.append((s, w_new, h_new, xs_new, ys_new, r_array))
            update_best(w_new, h_new, xs_new, ys_new, r_array)
            
    # Sort pass1 results by sum of radii
    pass1_results.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 5 for non-uniform refinement
    for _, w, h, xs, ys, r_array in pass1_results[:5]:
        if check_time():
            break
        # Non-uniform refinement
        w_new, h_new, r_avg, xs_new, ys_new, r_new_array = refine_non_uniform(w, h, np.mean(r_array), xs, ys, is_polish=False)
        if r_new_array is not None:
            update_best(w_new, h_new, xs_new, ys_new, r_new_array)

    # Third pass: Polish the very best result
    if best_config is not None:
        w, h, xs, ys, r_array = best_config
        if check_time():
            return np.column_stack((xs, ys, r_array))
        
        # If the best result has equal radii, try polishing it as non-uniform
        # If it's already non-uniform, just polish it again?
        # SLSQP is local, so a second polish might not help much unless we perturb.
        # Let's try a non-uniform polish on the best result found so far.
        w_new, h_new, r_avg, xs_new, ys_new, r_new_array = refine_non_uniform(w, h, np.mean(r_array), xs, ys, is_polish=True)
        if r_new_array is not None:
            update_best(w_new, h_new, xs_new, ys_new, r_new_array)

    if best_config is not None:
        w, h, xs, ys, r_array = best_config
        return np.column_stack((xs, ys, r_array))
    else:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

