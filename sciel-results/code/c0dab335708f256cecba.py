def _nelder_mead(func, x0, bounds=None, max_iter=100, tol=1e-8, time_limit=None, start_time=None):
    import numpy as np
    n = len(x0)
    if bounds is None:
        bounds = (-np.inf, np.inf)
    b_min, b_max = bounds
    
    # Initialize simplex
    simplex = [x0.copy()]
    for i in range(n):
        x = x0.copy()
        if x[i] == 0:
            x[i] = 1e-4
        else:
            x[i] *= 1.1
        simplex.append(x)
        
    for i in range(n + 1):
        simplex[i] = np.clip(simplex[i], b_min, b_max)
        
    values = [func(x) for x in simplex]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time > time_limit):
            break
            
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        if np.abs(values[-1] - values[0]) < tol:
            break
            
        centroid = np.mean(simplex[:-1], axis=0)
        
        xr = centroid + alpha * (centroid - simplex[-1])
        xr = np.clip(xr, b_min, b_max)
        fr = func(xr)
        
        if fr < values[0]:
            xe = centroid + gamma * (centroid - simplex[-1])
            xe = np.clip(xe, b_min, b_max)
            fe = func(xe)
            if fe < fr:
                simplex[-1] = xe
                values[-1] = fe
            else:
                simplex[-1] = xr
                values[-1] = fr
        elif fr < values[-2]:
            simplex[-1] = xr
            values[-1] = fr
        else:
            xc = centroid + rho * (simplex[-1] - centroid)
            xc = np.clip(xc, b_min, b_max)
            fc = func(xc)
            if fc < values[-1]:
                simplex[-1] = xc
                values[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], b_min, b_max)
                    values[i] = func(simplex[i])
                    
    best_idx = np.argmin(values)
    return simplex[best_idx], values[best_idx]
