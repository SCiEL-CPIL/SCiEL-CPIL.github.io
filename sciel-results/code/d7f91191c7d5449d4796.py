import time
import numpy as np
import random

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance (percent) over the band using characteristic matrices."""
    if not materials:
        return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[800-400], 401) # 401 points
    # Fixing a potential bug in seed: band[0] to band[1] is 400 to 800.
    # linspace(400, 800, 401) is correct.
    # The seed had: wavs = np.linspace(band[0], band[1], 401)
    
    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = [n_mat[m] for m in materials]
    ds = np.asarray(thicknesses, dtype=float)

    # Build per-layer optical path lengths
    n_ds = np.array([n * d for n, d in zip(ns, ds)])  # (N,)
    delta = 2.0 * np.pi * n_ds[:, None] / wavs[None, :]  # (N, K)
    c = np.cos(delta)
    s = np.sin(delta)

    # Initialize characteristic matrix product as identity for all wavelengths
    A = np.ones_like(wavs)
    B = np.zeros_like(wavs)
    C = np.zeros_like(wavs)
    D = np.ones_like(wavs)

    i = 1.0j
    for j in range(len(materials)):
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        # Update A,B,C,D
        A_new = A * cj + 1j * B * n_j * sj
        B_new = 1j * A * sj / n_j + B * cj
        C_new = C * cj + 1j * D * n_j * sj
        D_new = 1j * C * sj / n_j + D * cj
        A, B, C, D = A_new, B_new, C_new, D_new

    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    # Reflectance: r = (n_0 * (A + B * n_s) - (C + D * n_s)) / (n_0 * (A + B * n_s) + (C + D * n_s))
    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    r = num / den
    R = np.abs(r) ** 2
    return 100.0 * np.mean(R)

def design(problem):
    start_time = time.time()
    # 150s wall clock limit. Use 145s to be safe.
    deadline = start_time + 145.0
    
    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]
    n_L = materials_dict['L']
    n_H = materials_dict['H']
    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    
    # Precompute quarter-wave thicknesses at center wavelength
    qw_L = wl_center / (4.0 * n_L)
    qw_H = wl_center / (4.0 * n_H)
    
    # Objective function wrapper for optimization
    # We fix the material sequence (alternating L/H) to reduce dimensionality
    # and improve convergence reliability compared to discrete+continuous mixed search.
    # The physics of broadband AR coatings strongly favors graded index profiles,
    # which are best approximated by alternating L/H layers.
    
    def objective(thicknesses, n_layers, start_with_L):
        # thicknesses is a numpy array of length n_layers
        if np.any(thicknesses < 0) or np.any(thicknesses > max_layer):
            return 1000.0
        if np.sum(thicknesses) > max_total:
            return 1000.0
            
        mats = []
        for i in range(n_layers):
            if (i + (1 if start_with_L else 0)) % 2 == 0:
                mats.append('L')
            else:
                mats.append('H')
        
        return _mean_reflectance(mats, thicknesses, problem)

    best_score = float('inf')
    best_result = None
    
    # We will test different layer counts and starting materials
    # N=5, 7, 9, 11, 13, 15, 17, 19
    # Starting with L or H
    
    # To maximize performance within time, we prioritize higher layer counts
    # as they allow better broadband performance, but we must ensure we can
    # optimize them.
    
    # Strategy:
    # 1. Try Nelder-Mead with multiple random restarts for a fixed N.
    # 2. Use Differential Evolution (DE) for global search if time permits.
    
    # We implement a simple DE and Nelder-Mead to avoid scipy dependency issues 
    # if scipy is not available or slow, but scipy is available. 
    # However, custom implementation gives tighter control over time budget.
    
    # Let's use a simple Nelder-Mead implementation
    def nelder_mead(f, x0, bounds, max_iter=500, alpha=1, gamma=2, rho=0.5, sigma=0.5, tol=1e-8):
        n = len(x0)
        # Initialize simplex
        simplex = [x0.copy()]
        for i in range(n):
            x_new = x0.copy()
            x_new[i] += 0.1 if x0[i] == 0 else 0.1 * x0[i]
            # Ensure bounds
            x_new = np.clip(x_new, bounds[0], bounds[1])
            simplex.append(x_new)
        
        fx = [f(x) for x in simplex]
        
        for _ in range(max_iter):
            # Sort
            order = np.argsort(fx)
            simplex = [simplex[i] for i in order]
            fx = [fx[i] for i in order]
            
            # Check convergence
            if fx[0] < tol:
                break
            if max(fx) - min(fx) < tol:
                break
                
            # Centroid
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, bounds[0], bounds[1])
            f_r = f(x_r)
            
            if f_r < fx[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, bounds[0], bounds[1])
                f_e = f(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    fx[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fx[-1] = f_r
            elif f_r < fx[-2]:
                simplex[-1] = x_r
                fx[-1] = f_r
            else:
                # Contraction
                x_c = centroid + rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, bounds[0], bounds[1])
                f_c = f(x_c)
                if f_c < fx[-1]:
                    simplex[-1] = x_c
                    fx[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], bounds[0], bounds[1])
                        fx[i] = f(simplex[i])
            
            # Update best
            order = np.argsort(fx)
            simplex = [simplex[i] for i in order]
            fx = [fx[i] for i in order]
            
        return simplex[0], fx[0]

    def differential_evolution(f, bounds, pop_size=20, max_iter=50, F=0.8, CR=0.9):
        n = len(bounds)
        lb, ub = bounds
        # Initialize population
        pop = [np.random.uniform(lb, ub) for _ in range(pop_size)]
        fx = [f(x) for x in pop]
        
        for _ in range(max_iter):
            new_pop = pop[:]
            new_fx = fx[:]
            
            for i in range(pop_size):
                # Select 3 distinct random individuals
                a, b, c = random.sample(range(pop_size), 3)
                
                # Mutation
                v = pop[a] + F * (pop[b] - pop[c])
                v = np.clip(v, lb, ub)
                
                # Crossover
                u = np.random.random(n) < CR
                if not np.any(u):
                    j0 = random.randint(0, n-1)
                    u[j0] = True
                trial = np.where(u, v, pop[i])
                trial = np.clip(trial, lb, ub)
                
                # Selection
                f_trial = f(trial)
                if f_trial < fx[i]:
                    new_pop[i] = trial
                    new_fx[i] = f_trial
            
            pop = new_pop
            fx = new_fx
            
            # Early stopping if converged
            if max(fx) - min(fx) < 1e-6:
                break
                
        best_idx = np.argmin(fx)
        return pop[best_idx], fx[best_idx]

    # Main loop: Try different configurations
    # We iterate through likely good configurations
    # N=9 is a sweet spot often. Let's try 5, 7, 9, 11.
    # We allocate time based on remaining budget.
    
    configs = [
        (5, True), (5, False),
        (7, True), (7, False),
        (9, True), (9, False),
        (11, True), (11, False),
        (13, True), (13, False),
        (15, True), (15, False)
    ]
    
    # Shuffle to randomize order if time runs out
    random.shuffle(configs)
    
    # Time allocation: 10s per config for NM restarts
    time_per_config = 10.0
    
    for n_layers, start_with_L in configs:
        if time.time() > deadline - 10: # 10s reserve
            break
            
        # Define bounds
        # Total thickness constraint: sum(d) <= max_total
        # Individual constraint: 0 <= d <= max_layer
        # For NM, we can just use box constraints and penalize violations in objective
        
        lb = np.zeros(n_layers)
        ub = np.full(n_layers, max_layer)
        
        # Objective function for this specific config
        def obj_func(d):
            if np.any(d < 0) or np.any(d > max_layer):
                return 1000.0
            if np.sum(d) > max_total:
                return 1000.0
            mats = []
            for i in range(n_layers):
                if (i + (1 if start_with_L else 0)) % 2 == 0:
                    mats.append('L')
                else:
                    mats.append('H')
            return _mean_reflectance(mats, d, problem)
        
        # Initial guess: quarter wave
        x0 = []
        for i in range(n_layers):
            if (i + (1 if start_with_L else 0)) % 2 == 0:
                x0.append(qw_L)
            else:
                x0.append(qw_H)
        x0 = np.array(x0)
        
        # Run Nelder-Mead with a few restarts
        # Time limit for this config
        config_deadline = time.time() + time_per_config
        
        best_x_config = None
        best_f_config = float('inf')
        
        # 1 restart from x0
        if time.time() < config_deadline:
            x_opt, f_opt = nelder_mead(obj_func, x0, (lb, ub), max_iter=200)
            if f_opt < best_f_config:
                best_f_config = f_opt
                best_x_config = x_opt
        
        # 3 random restarts
        for _ in range(3):
            if time.time() > config_deadline:
                break
            x_rand = np.random.uniform(lb, ub)
            # Scale to fit total thickness roughly
            if np.sum(x_rand) > max_total:
                x_rand *= max_total / np.sum(x_rand)
            x_opt, f_opt = nelder_mead(obj_func, x_rand, (lb, ub), max_iter=100)
            if f_opt < best_f_config:
                best_f_config = f_opt
                best_x_config = x_opt
        
        if best_f_config < best_score and best_x_config is not None:
            best_score = best_f_config
            mats = []
            for i in range(n_layers):
                if (i + (1 if start_with_L else 0)) % 2 == 0:
                    mats.append('L')
                else:
                    mats.append('H')
            best_result = (mats, list(best_x_config))
            
    # If time remains, run DE on the best configuration found or a new one
    if time.time() < deadline - 20 and best_result is not None:
        # Use DE to polish or explore
        # Let's try DE on N=9 (often robust)
        n_layers = 9
        start_with_L = True # Or check which was best
        
        lb = np.zeros(n_layers)
        ub = np.full(n_layers, max_layer)
        
        def obj_func_de(d):
            if np.any(d < 0) or np.any(d > max_layer):
                return 1000.0
            if np.sum(d) > max_total:
                return 1000.0
            mats = []
            for i in range(n_layers):
                if (i + (1 if start_with_L else 0)) % 2 == 0:
                    mats.append('L')
                else:
                    mats.append('H')
            return _mean_reflectance(mats, d, problem)
        
        # DE is expensive. Run for limited time.
        de_deadline = deadline - 5
        pop_size = 20
        max_iter = 100
        
        # Custom DE loop with time check
        pop = [np.random.uniform(lb, ub) for _ in range(pop_size)]
        # Scale initial pop to valid range
        for i in range(pop_size):
            if np.sum(pop[i]) > max_total:
                pop[i] *= max_total / np.sum(pop[i])
        fx = [obj_func_de(x) for x in pop]
        
        for _ in range(max_iter):
            if time.time() > de_deadline:
                break
            new_pop = pop[:]
            new_fx = fx[:]
            
            for i in range(pop_size):
                a, b, c = random.sample(range(pop_size), 3)
                v = pop[a] + 0.8 * (pop[b] - pop[c])
                v = np.clip(v, lb, ub)
                
                u = np.random.random(n_layers) < 0.9
                if not np.any(u):
                    j0 = random.randint(0, n_layers-1)
                    u[j0] = True
                trial = np.where(u, v, pop[i])
                trial = np.clip(trial, lb, ub)
                
                f_trial = obj_func_de(trial)
                if f_trial < fx[i]:
                    new_pop[i] = trial
                    new_fx[i] = f_trial
            
            pop = new_pop
            fx = new_fx
            if max(fx) - min(fx) < 1e-5:
                break
        
        best_idx = np.argmin(fx)
        if fx[best_idx] < best_score:
            best_score = fx[best_idx]
            mats = []
            for i in range(n_layers):
                if (i + (1 if start_with_L else 0)) % 2 == 0:
                    mats.append('L')
                else:
                    mats.append('H')
            best_result = (mats, list(pop[best_idx]))

    # Fallback if nothing found
    if best_result is None:
        return ["L"], [qw_L]
        
    mats, ths = best_result
    # Final validation and clipping
    total = sum(ths)
    if total > max_total:
        scale = max_total / total
        ths = [t * scale for t in ths]
    ths = [max(0.0, min(max_layer, t)) for t in ths]
    
    return list(mats), list(ths)

