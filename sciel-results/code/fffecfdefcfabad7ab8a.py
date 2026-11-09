<<<<<<< SEARCH
def _nelder_mead(f, x0, bounds, max_iter=2000, tol=1e-10, deadline=None):
    """
    Nelder-Mead optimization with bounds handling.
    f: objective function
    x0: initial point
    bounds: list of (low, high) for each dimension
    """
    n = len(x0)
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    # Create initial simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        xi = np.array(x0, dtype=float)
        if xi[i] != 0:
            xi[i] *= 1.05
        else:
            xi[i] = 1e-4
        simplex.append(xi)
    
    # Evaluate initial simplex
    fvals = [f(x) for x in simplex]
    
    # Sort simplex by function value
    order = np.argsort(fvals)
    simplex = [simplex[i] for i in order]
    fvals = [fvals[i] for i in order]
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        # Enforce bounds
        for i in range(n):
            xr[i] = np.clip(xr[i], bounds[i][0], bounds[i][1])
        fr = f(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            for i in range(n):
                xe[i] = np.clip(xe[i], bounds[i][0], bounds[i][1])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            # Accept reflection
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            if fr < fvals[-1]:
                # Outside contraction
                xc = centroid + rho * (xr - centroid)
            else:
                # Inside contraction
                xc = centroid - rho * (simplex[-1] - centroid)
            for i in range(n):
                xc[i] = np.clip(xc[i], bounds[i][0], bounds[i][1])
            fc = f(xc)
            
            if fc < min(fr, fvals[-1]):
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    for j in range(n):
                        simplex[i][j] = np.clip(simplex[i][j], bounds[j][0], bounds[j][1])
                    fvals[i] = f(simplex[i])
        
        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Convergence check
        if abs(fvals[0] - fvals[-1]) < tol:
            break
    
    return simplex[0], fvals[0]
=======
def _nelder_mead(f, x0, bounds, max_iter=2000, tol=1e-10, deadline=None):
    """
    Nelder-Mead optimization with bounds handling.
    f: objective function
    x0: initial point
    bounds: list of (low, high) for each dimension
    """
    n = len(x0)
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    # Create initial simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        xi = np.array(x0, dtype=float)
        if xi[i] != 0:
            xi[i] *= 1.05
        else:
            xi[i] = 1e-4
        simplex.append(xi)
    
    # Evaluate initial simplex
    fvals = [f(x) for x in simplex]
    
    # Sort simplex by function value
    order = np.argsort(fvals)
    simplex = [simplex[i] for i in order]
    fvals = [fvals[i] for i in order]
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        # Enforce bounds
        for i in range(n):
            xr[i] = np.clip(xr[i], bounds[i][0], bounds[i][1])
        fr = f(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            for i in range(n):
                xe[i] = np.clip(xe[i], bounds[i][0], bounds[i][1])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            # Accept reflection
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            if fr < fvals[-1]:
                # Outside contraction
                xc = centroid + rho * (xr - centroid)
            else:
                # Inside contraction
                xc = centroid - rho * (simplex[-1] - centroid)
            for i in range(n):
                xc[i] = np.clip(xc[i], bounds[i][0], bounds[i][1])
            fc = f(xc)
            
            if fc < min(fr, fvals[-1]):
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    for j in range(n):
                        simplex[i][j] = np.clip(simplex[i][j], bounds[j][0], bounds[j][1])
                    fvals[i] = f(simplex[i])
        
        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Convergence check
        if abs(fvals[0] - fvals[-1]) < tol:
            break
    
    return simplex[0], fvals[0]
>>>>>>> REPLACE

