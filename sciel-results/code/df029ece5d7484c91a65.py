"""Broadband antireflection coating design via Differential Evolution + Nelder-Mead Polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _mean_reflectance_pct(thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls):
    """
    Compute mean reflectance in percent for a single design across all wavelengths.
    """
    n_layers = len(thicknesses)
    if n_layers == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        R = (r * r).real
        return float(R * 100.0)
    
    # Initialize characteristic matrix M = I
    M11 = np.ones_like(wls, dtype=complex)
    M12 = np.zeros_like(wls, dtype=complex)
    M21 = np.zeros_like(wls, dtype=complex)
    M22 = np.ones_like(wls, dtype=complex)
    
    for i in range(n_layers):
        mat = materials[i]
        if mat == 'H':
            n = n_H
        else:
            n = n_L
            
        d = thicknesses[i]
        # beta = 2*pi*n*d / wl
        beta = 2.0 * np.pi * n * d / wls
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        # Layer matrix components
        L11 = cos_b
        L12 = -1j * sin_b / n
        L21 = 1j * n * sin_b
        L22 = cos_b
        
        # M = M @ L
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
    
    # Admittance Y
    Y = (M11 * n_substrate + M12) / (M21 * n_substrate + M22)
    
    # Reflection coefficient r
    r = (n_ambient - Y) / (n_ambient + Y)
    
    # Reflectance R
    R = (r * r.conjugate()).real
    
    # Mean reflectance in percent
    return float(np.mean(R) * 100.0)

def _nm_optimize(thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls, max_total_nm, max_layer_nm, deadline):
    """
    Nelder-Mead optimization on thicknesses for a fixed material sequence.
    """
    if len(thicknesses) == 0:
        return thicknesses, _mean_reflectance_pct(thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls)
    
    def objective(x):
        x = np.clip(x, 0, max_layer_nm)
        if np.sum(x) > max_total_nm:
            # Penalty for exceeding total thickness
            return 100.0 + (np.sum(x) - max_total_nm)
        return _mean_reflectance_pct(x, materials, n_ambient, n_substrate, n_L, n_H, wls)
    
    x0 = np.array(thicknesses, dtype=float)
    x0 = np.clip(x0, 0, max_layer_nm)
    if np.sum(x0) > max_total_nm:
        x0 *= max_total_nm / np.sum(x0)
        
    # Simple Nelder-Mead implementation
    n = len(x0)
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        xi[i] += 0.1 * (max_layer_nm if xi[i] == 0 else 1.0) # Heuristic initial step
        simplex.append(xi)
        
    f_simplex = [objective(v) for v in simplex]
    
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    max_iter = 300
    
    for _ in range(max_iter):
        if time.time() > deadline - 5.0:
            break
            
        # Sort simplex
        order = np.argsort(f_simplex)
        simplex = [simplex[i] for i in order]
        f_simplex = [f_simplex[i] for i in order]
        
        # Check convergence
        if f_simplex[-1] - f_simplex[0] < 1e-6:
            break
            
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)
        
        if fr < f_simplex[0]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                f_simplex[-1] = fe
            else:
                simplex[-1] = xr
                f_simplex[-1] = fr
        elif fr < f_simplex[-2]:
            simplex[-1] = xr
            f_simplex[-1] = fr
        else:
            # Contraction
            if fr < f_simplex[-1]:
                # Outside contraction
                xc = centroid - 0.5 * (centroid - simplex[-1])
            else:
                # Inside contraction
                xc = centroid + 0.5 * (centroid - simplex[-1])
            fc = objective(xc)
            if fc < f_simplex[-1]:
                simplex[-1] = xc
                f_simplex[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    f_simplex[i] = objective(simplex[i])
                    
    # Return best
    best_idx = np.argmin(f_simplex)
    return simplex[best_idx], f_simplex[best_idx]

def design(problem):
    """Design a broadband antireflection coating."""
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    wl_min, wl_max = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 401 equally spaced wavelengths
    wls = np.linspace(wl_min, wl_max, 401)
    wl_center = 0.5 * (wl_min + wl_max)
    
    qw_L = wl_center / (4.0 * n_L)
    qw_H = wl_center / (4.0 * n_H)
    
    best_score = float('inf')
    best_th = None
    best_mat = None
    
    # Differential Evolution Parameters
    pop_size = 40
    F = 0.8
    CR = 0.9
    
    # Try a few layer counts, focusing on odd numbers which often perform well for broadband AR
    # We will allocate time to different layer counts.
    layer_counts_to_try = [5, 7, 9, 11, 13]
    
    # Filter valid layer counts
    layer_counts_to_try = [c for c in layer_counts_to_try if c <= max_layers]
    
    # If max_layers is small, adjust
    if not layer_counts_to_try:
        layer_counts_to_try = [1, 3]
        
    # Ensure we try at least one
    if max_layers == 1:
        layer_counts_to_try = [1]
    elif max_layers == 2:
        layer_counts_to_try = [2]
        
    # Time allocation
    time_per_config = (deadline - start_time - 10.0) / len(layer_counts_to_try)
    
    for n_layers in layer_counts_to_try:
        if time.time() > deadline - 15.0:
            break
            
        t_start = time.time()
        t_limit = t_start + time_per_config
        
        # Encoding: 
        # First n_layers values: thicknesses (0 to max_layer_nm)
        # Next n_layers values: material (0 to 1, mapped to L/H)
        # Total dimensions = 2 * n_layers
        
        # Bounds
        bounds = []
        for _ in range(n_layers):
            bounds.append((0.0, max_layer_nm)) # Thickness
        for _ in range(n_layers):
            bounds.append((0.0, 1.0)) # Material (binary-like)
            
        # Initialize Population
        pop = []
        for i in range(pop_size):
            x = []
            # Thicknesses: Start near quarter-wave
            for j in range(n_layers):
                # Random initialization around quarter-wave values
                # We need to guess material first to know which qw to use?
                # Let's just randomize thicknesses in [0, max_layer_nm]
                t = np.random.uniform(0, max_layer_nm)
                x.append(t)
            # Materials: Random 0/1
            for j in range(n_layers):
                m = np.random.choice([0.0, 1.0])
                x.append(m)
            pop.append(np.array(x))
            
        # Evaluate initial population
        scores = []
        for x in pop:
            th = x[:n_layers]
            mat_vals = x[n_layers:]
            mat = ['H' if v > 0.5 else 'L' for v in mat_vals]
            
            # Check total thickness
            if np.sum(th) > max_total_nm:
                scores.append(float('inf'))
                continue
                
            scores.append(_mean_reflectance_pct(th, mat, n_ambient, n_substrate, n_L, n_H, wls))
            
        # DE Loop
        gen = 0
        while time.time() < t_limit - 5.0:
            gen += 1
            new_pop = []
            new_scores = []
            
            # Sort current pop
            idx = np.argsort(scores)
            sorted_pop = [pop[i] for i in idx]
            sorted_scores = [scores[i] for i in idx]
            
            for i in range(pop_size):
                # Select 3 distinct random individuals
                a_idx, b_idx, c_idx = np.random.choice(pop_size, 3, replace=False)
                a, b, c = sorted_pop[a_idx], sorted_pop[b_idx], sorted_pop[c_idx]
                
                # Mutation
                v = a + F * (b - c)
                
                # Crossover
                trial = np.zeros_like(v)
                j_rand = np.random.randint(len(v))
                for j in range(len(v)):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = sorted_pop[i][j] # Use target (sorted_pop[i] is the i-th best, which acts as target in this loop structure? 
                        # Wait, standard DE usually iterates over targets. 
                        # Let's stick to standard: target is pop[i].
                        # But I sorted pop. So target is sorted_pop[i].
                        pass 
                # Correct Crossover logic:
                for j in range(len(v)):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = sorted_pop[i][j]
                        
                # Clip to bounds
                for j in range(len(trial)):
                    lb, ub = bounds[j]
                    if trial[j] < lb: trial[j] = lb
                    if trial[j] > ub: trial[j] = ub
                    
                # Evaluate trial
                th = trial[:n_layers]
                mat_vals = trial[n_layers:]
                mat = ['H' if v > 0.5 else 'L' for v in mat_vals]
                
                if np.sum(th) > max_total_nm:
                    trial_score = float('inf')
                else:
                    trial_score = _mean_reflectance_pct(th, mat, n_ambient, n_substrate, n_L, n_H, wls)
                    
                # Selection
                if trial_score < sorted_scores[i]:
                    new_pop.append(trial)
                    new_scores.append(trial_score)
                else:
                    new_pop.append(sorted_pop[i])
                    new_scores.append(sorted_scores[i])
                    
            pop = new_pop
            scores = new_scores
            
            # Update global best
            min_idx = np.argmin(scores)
            if scores[min_idx] < best_score:
                best_score = scores[min_idx]
                best_th = pop[min_idx][:n_layers].copy()
                best_mat = ['H' if v > 0.5 else 'L' for v in pop[min_idx][n_layers:]]
                
            # Convergence check
            if scores[0] < 0.5: # Very good
                break
            if gen % 10 == 0 and scores[0] == scores[-1]:
                # Stagnation, perturb
                for k in range(pop_size):
                    pop[k] += np.random.normal(0, 5, len(pop[k]))
                    # Re-clip
                    for j in range(len(pop[k])):
                        lb, ub = bounds[j]
                        if pop[k][j] < lb: pop[k][j] = lb
                        if pop[k][j] > ub: pop[k][j] = ub
                # Re-evaluate
                for k in range(pop_size):
                    th = pop[k][:n_layers]
                    mat_vals = pop[k][n_layers:]
                    mat = ['H' if v > 0.5 else 'L' for v in mat_vals]
                    if np.sum(th) > max_total_nm:
                        scores[k] = float('inf')
                    else:
                        scores[k] = _mean_reflectance_pct(th, mat, n_ambient, n_substrate, n_L, n_H, wls)

    # Final Polish with Nelder-Mead on the best candidate
    if best_th is not None and time.time() < deadline - 10.0:
        polished_th, polished_score = _nm_optimize(
            best_th, best_mat, n_ambient, n_substrate, n_L, n_H, wls, 
            max_total_nm, max_layer_nm, deadline
        )
        if polished_score < best_score:
            best_score = polished_score
            best_th = polished_th
            # best_mat remains the same
            
    # Fallback
    if best_th is None:
        return ["L"], [qw_L]
        
    # Final validity check
    final_th_list = [float(t) for t in best_th]
    final_mat_list = [str(m) for m in best_mat]
    
    if len(final_th_list) != len(final_mat_list):
        return ["L"], [qw_L]
        
    for t in final_th_list:
        if t < 0 or t > max_layer_nm:
            return ["L"], [qw_L]
            
    if sum(final_th_list) > max_total_nm:
        # Scale down
        s = max_total_nm / sum(final_th_list)
        final_th_list = [t * s for t in final_th_list]
        
    return final_mat_list, final_th_list

# EVOLVE-BLOCK-END
