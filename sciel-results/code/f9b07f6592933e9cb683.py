def _differential_evolution(f, x0, bounds, max_iter=200, pop_size=50, F=0.8, CR=0.9, timeout=None, start_time=None):
    """Differential evolution optimization."""
    n = len(x0)
    pop = [x0 + np.random.uniform(-0.1, 0.1, n) for _ in range(pop_size)]
    # Ensure bounds
    for i in range(pop_size):
        for j in range(n):
            pop[i][j] = max(bounds[j][0], min(bounds[j][1], pop[i][j]))
    
    fvals = [f(x) for x in pop]
    best_idx = np.argmin(fvals)
    best_x = pop[best_idx].copy()
    best_f = fvals[best_idx]
    
    for it in range(max_iter):
        if timeout is not None and start_time is not None:
            if time.time() - start_time > timeout:
                break
        
        new_pop = [x.copy() for x in pop]
        new_fvals = fvals[:]
        
        for i in range(pop_size):
            if time.time() - start_time > timeout:
                break
            
            # Select three distinct random individuals
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            v = pop[r1] + F * (pop[r2] - pop[r3])
            
            # Clip to bounds
            for j in range(n):
                v[j] = max(bounds[j][0], min(bounds[j][1], v[j]))
            
            # Crossover
            trial = pop[i].copy()
            for j in range(n):
                if np.random.random() < CR or j == np.random.randint(n):
                    trial[j] = v[j]
            
            f_trial = f(trial)
            if f_trial < fvals[i]:
                new_pop[i] = trial
                new_fvals[i] = f_trial
        
        pop = new_pop
        fvals = new_fvals
        
        best_idx = np.argmin(fvals)
        if fvals[best_idx] < best_f:
            best_f = fvals[best_idx]
            best_x = pop[best_idx].copy()
        
        # Check convergence
        if max(fvals) - min(fvals) < 1e-10:
            break
    
    return best_x, best_f
