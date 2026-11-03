# EVOLVE-BLOCK-START
def construct():
    """
    Pack N circles in the unit square maximizing sum of radii.
    Uses SLSQP optimization with multiple restarts.
    """
    N = 26
    
    def objective(vars):
        # vars: [x0, y0, r0, x1, y1, r1, ...]
        radii = vars[2::3]
        return -np.sum(radii)
    
    def constraints_fun(vars):
        n = N
        cons = []
        # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r, r > 0
        for i in range(n):
            x = vars[i*3]
            y = vars[i*3+1]
            r = vars[i*3+2]
            cons.append(x - r)
            cons.append(1 - x - r)
            cons.append(y - r)
            cons.append(1 - y - r)
            cons.append(r - 1e-8)
        
        # Pairwise constraints: dist >= ri + rj
        for i in range(n):
            xi = vars[i*3]
            yi = vars[i*3+1]
            ri = vars[i*3+2]
            for j in range(i+1, n):
                xj = vars[j*3]
                yj = vars[j*3+1]
                rj = vars[j*3+2]
                dist_sq = (xi-xj)**2 + (yi-yj)**2
                min_dist_sq = (ri+rj)**2
                cons.append(dist_sq - min_dist_sq)
        
        return np.array(cons)
    
    def get_initial_guess(method):
        n = N
        if method == 'grid':
            side = int(np.ceil(np.sqrt(n)))
            cells = []
            for i in range(side):
                for j in range(side):
                    if len(cells) >= n:
                        break
                    cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01])
            return np.array(cells).flatten()
        elif method == 'random':
            np.random.seed(42 + method)
            x = np.random.uniform(0.1, 0.9, n)
            y = np.random.uniform(0.1, 0.9, n)
            r = np.full(n, 0.02)
            return np.column_stack([x, y, r]).flatten()
        else:
            return get_initial_guess('grid')

    best_vars = None
    best_score = -np.inf

    # Try a few different initializations
    inits = ['grid', 'random', 'random2']
    
    # Add a second random init with different seed
    def get_initial_guess2():
        n = N
        np.random.seed(123)
        x = np.random.uniform(0.1, 0.9, n)
        y = np.random.uniform(0.1, 0.9, n)
        r = np.full(n, 0.02)
        return np.column_stack([x, y, r]).flatten()
        
    inits_funcs = [get_initial_guess, get_initial_guess, get_initial_guess2]
    init_names = ['grid', 'random', 'random2']
    
    for name, func in zip(init_names, inits_funcs):
        x0 = func()
        
        cons = {'type': 'ineq', 'fun': constraints_fun}
        
        bounds = []
        for i in range(N):
            bounds.append((1e-6, 1)) # x
            bounds.append((1e-6, 1)) # y
            bounds.append((1e-6, 0.5)) # r
            
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 500, 'ftol': 1e-10})
            
            # Check validity and score
            x = res.x[0::3]
            y = res.x[1::3]
            r = res.x[2::3]
            
            valid = True
            # Boundary check
            if np.any(x < r - 1e-9) or np.any(x > 1 - r + 1e-9) or np.any(y < r - 1e-9) or np.any(y > 1 - r + 1e-9):
                valid = False
            if valid:
                # Pairwise check
                for i in range(N):
                    for j in range(i+1, N):
                        dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                        if dist_sq < (r[i]+r[j])**2 - 1e-9:
                            valid = False
                            break
                    if not valid: break
            
            if valid:
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_vars = res.x
        except Exception as e:
            pass

    if best_vars is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01])
        return np.array(cells)
        
    return best_vars.reshape(N, 3)
# EVOLVE-BLOCK-END
