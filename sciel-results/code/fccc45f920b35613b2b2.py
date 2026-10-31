    import numpy as np
    from scipy.optimize import minimize

    def construct():
        N = 21
        
        def overlap_cost(pos, r, w, h):
            cost = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < 2 * r:
                        cost += (2 * r - d) ** 2
            for i in range(N):
                if pos[i][0] < r:
                    cost += (r - pos[i][0]) ** 2
                if pos[i][0] > w - r:
                    cost += (pos[i][0] - (w - r)) ** 2
                if pos[i][1] < r:
                    cost += (r - pos[i][1]) ** 2
                if pos[i][1] > h - r:
                    cost += (pos[i][1] - (h - r)) ** 2
            return cost

        def optimize_positions(r, w, h, init_pos):
            x0 = init_pos.flatten()
            bounds = [(r, w - r) for _ in range(N)] + [(r, h - r) for _ in range(N)]
            
            def cost_func(x):
                pos = x.reshape(N, 2)
                return overlap_cost(pos, r, w, h)
            
            try:
                res = minimize(cost_func, x0, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
                return res.x.reshape(N, 2), res.fun
            except Exception:
                return init_pos, 1e10

        best_sum = 0
        best_result = None
        
        aspect_ratios = [0.5, 0.45, 0.55, 0.4, 0.6]
        
        for w_frac in aspect_ratios:
            w = 2 * w_frac
            h = 2 - w
            
            # Initial grid
            nx = int(np.ceil(np.sqrt(N * w / h)))
            ny = int(np.ceil(N / nx))
            xs = np.linspace(w / (2*nx), w - w / (2*nx), nx)
            ys = np.linspace(h / (2*ny), h - h / (2*ny), ny)
            
            init_pos = np.array([[x, y] for x in xs for y in ys])[:N]
            
            # Start with a small radius
            r = 0.05
            current_pos = init_pos
            
            # Increase radius
            for step in range(50):
                # Try to increase r
                r_new = r * 1.02 # 2% increase
                if 2 * r_new > w or 2 * r_new > h:
                    break
                
                new_pos, cost = optimize_positions(r_new, w, h, current_pos)
                
                if cost < 1e-4: # Feasible
                    r = r_new
                    current_pos = new_pos
                else:
                    # Try smaller step
                    r_new = r * 1.005
                    if 2 * r_new > w or 2 * r_new > h:
                        break
                    new_pos, cost = optimize_positions(r_new, w, h, current_pos)
                    if cost < 1e-4:
                        r = r_new
                        current_pos = new_pos
                    else:
                        break
                        
            # Final polish
            # Try to squeeze a bit more
            for _ in range(10):
                r_new = r * 1.001
                if 2 * r_new > w or 2 * r_new > h:
                    break
                new_pos, cost = optimize_positions(r_new, w, h, current_pos)
                if cost < 1e-6:
                    r = r_new
                    current_pos = new_pos
                else:
                    break
            
            total_sum = N * r
            if total_sum > best_sum:
                best_sum = total_sum
                result = np.zeros((N, 3))
                result[:, 0] = current_pos[:, 0]
                result[:, 1] = current_pos[:, 1]
                result[:, 2] = r
                best_result = result

        if best_result is None:
            side = 5
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                     for i in range(side) for j in range(side)]
            return np.array(cells[:N])
            
        return best_result
    