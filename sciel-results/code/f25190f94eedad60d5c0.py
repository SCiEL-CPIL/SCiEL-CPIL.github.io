# EVOLVE-BLOCK-START
import time
import numpy as np

def _reflectance_spectrum(n_vals, d_vals, lambdas, n_ambient, n_substrate):
    """
    Vectorized reflectance computation for a stack of layers.
    n_vals: array of refractive indices (length N)
    d_vals: array of thicknesses in nm (length N)
    lambdas: array of wavelengths in nm (length W)
    Returns: array of reflectance (0 to 1) for each wavelength.
    """
    N = len(n_vals)
    W = len(lambdas)
    
    if N == 0:
        r0 = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.full(W, np.abs(r0) ** 2)
    
    # Initialize total matrix as identity for each wavelength
    M11 = np.ones(W, dtype=complex)
    M12 = np.zeros(W, dtype=complex)
    M21 = np.zeros(W, dtype=complex)
    M22 = np.ones(W, dtype=complex)
    
    for j in range(N):
        n = n_vals[j]
        d = d_vals[j]
        if d < 1e-10:
            continue
        beta = 2.0 * np.pi * n * d / lambdas
        c = np.cos(beta)
        s = np.sin(beta)
        
        L11 = c
        L12 = 1j * s / n
        L21 = 1j * n * s
        L22 = c
        
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22
    
    Y = (M11 + M12 * n_substrate) / (M21 + M22 * n_substrate)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r) ** 2
    
    return R

def _evaluate(n_vals, d_vals, lambdas, n_ambient, n_substrate):
    """Compute mean reflectance in percent."""
    R = _reflectance_spectrum(n_vals, d_vals, lambdas, n_ambient, n_substrate)
    return 100.0 * np.mean(R)

def _is_valid(d_vals, max_total_nm, max_layer_nm):
    """Check thickness constraints."""
    if len(d_vals) == 0:
        return True
    total = sum(d_vals)
    if total > max_total_nm + 1e-6:
        return False
    for t in d_vals:
        if t < -1e-6 or t > max_layer_nm + 1e-6:
            return False
    return True

def _n_from_materials(materials, n_L, n_H):
    """Convert material labels to refractive index array."""
    return np.array([n_L if m == 'L' else n_H for m in materials])

def _project_feasible(x, max_total_nm, max_layer_nm):
    """Project to feasible region."""
    x = np.clip(x, 0.0, max_layer_nm)
    total = np.sum(x)
    if total > max_total_nm:
        x = x * (max_total_nm / total)
    return x

def _nelder_mead_optimize(materials, initial_thicks, problem, time_limit):
    """Nelder-Mead optimization on layer thicknesses."""
    n_params = len(initial_thicks)
    if n_params == 0:
        return initial_thicks, _evaluate(_n_from_materials(materials, problem["materials"]["L"], problem["materials"]["H"]),
                                         np.array(initial_thicks, dtype=float),
                                         np.linspace(problem["band_nm"][0], problem["band_nm"][1], 401),
                                         problem["ambient_index"], problem["substrate_index"])
        
    n_vals = _n_from_materials(materials, problem["materials"]["L"], problem["materials"]["H"])
    lambdas = np.linspace(problem["band_nm"][0], problem["band_nm"][1], 401)
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    
    def eval_vec(x):
        if not _is_valid(x, problem["max_total_nm"], problem["max_layer_nm"]):
            return float('inf')
        return _evaluate(n_vals, x, lambdas, n_ambient, n_substrate)

    x0 = _project_feasible(np.array(initial_thicks, dtype=float), problem["max_total_nm"], problem["max_layer_nm"])
    
    # Build simplex
    simplex = [x0.copy()]
    step = 5.0
    for i in range(n_params):
        xi = x0.copy()
        xi[i] += step
        xi = _project_feasible(xi, problem["max_total_nm"], problem["max_layer_nm"])
        simplex.append(xi)
    
    simplex = np.array(simplex)
    f_values = np.array([eval_vec(simplex[i]) for i in range(n_params + 1)])
    
    best_score = np.min(f_values)
    best_thicks = simplex[np.argmin(f_values)].tolist()
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    max_iter = 200
    
    for _ in range(max_iter):
        if time.time() > time_limit:
            break
            
        idx = np.argsort(f_values)
        simplex = simplex[idx]
        f_values = f_values[idx]
        
        if np.abs(f_values[0] - f_values[-1]) < 1e-10:
            break
        if np.max(np.std(simplex, axis=0)) < 1e-8:
            break

        x_worst = simplex[-1]
        x_best = simplex[0]
        x_c = np.mean(simplex[:-1], axis=0)
        
        x_r = x_c + alpha * (x_c - x_worst)
        x_r = _project_feasible(x_r, problem["max_total_nm"], problem["max_layer_nm"])
        f_r = eval_vec(x_r)
        
        if f_r < f_values[0]:
            x_e = x_c + gamma * (x_r - x_c)
            x_e = _project_feasible(x_e, problem["max_total_nm"], problem["max_layer_nm"])
            f_e = eval_vec(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_values[-1] = f_e
            else:
                simplex[-1] = x_r
                f_values[-1] = f_r
        elif f_r < f_values[-2]:
            simplex[-1] = x_r
            f_values[-1] = f_r
        else:
            if f_r < f_values[-1]:
                x_c_new = x_c + rho * (x_r - x_c)
            else:
                x_c_new = x_c - rho * (x_c - x_r)
            x_c_new = _project_feasible(x_c_new, problem["max_total_nm"], problem["max_layer_nm"])
            f_c = eval_vec(x_c_new)
            
            if f_c < min(f_r, f_values[-1]):
                simplex[-1] = x_c_new
                f_values[-1] = f_c
            else:
                for i in range(1, n_params + 1):
                    simplex[i] = x_best + sigma * (simplex[i] - x_best)
                    simplex[i] = _project_feasible(simplex[i], problem["max_total_nm"], problem["max_layer_nm"])
                    f_values[i] = eval_vec(simplex[i])
                    
        current_best_idx = np.argmin(f_values)
        if f_values[current_best_idx] < best_score:
            best_score = f_values[current_best_idx]
            best_thicks = simplex[current_best_idx].tolist()
            
    return best_thicks, best_score

def _differential_evolution(materials, bounds, problem, pop_size, max_iter, time_limit):
    """Differential evolution for global optimization."""
    n_params = len(bounds)
    n_vals = _n_from_materials(materials, problem["materials"]["L"], problem["materials"]["H"])
    lambdas = np.linspace(problem["band_nm"][0], problem["band_nm"][1], 401)
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    def eval_vec(x):
        if not _is_valid(x, max_total, max_layer):
            return float('inf')
        return _evaluate(n_vals, x, lambdas, n_ambient, n_substrate)

    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    # Initialize population with quarter-wave starting points and random
    population = []
    # Add quarter-wave stack
    center_lam = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    qw = []
    for i in range(n_params):
        mat = materials[i]
        n = n_vals[i]
        qw.append(center_lam / (4.0 * n))
    qw = _project_feasible(np.array(qw, dtype=float), max_total, max_layer)
    population.append(qw)
    
    # Add random points
    for _ in range(pop_size - 1):
        pt = np.random.uniform(lb, ub)
        pt = _project_feasible(pt, max_total, max_layer)
        population.append(pt)
    
    population = np.array(population)
    fitness = np.array([eval_vec(population[i]) for i in range(pop_size)])
    
    best_idx = np.argmin(fitness)
    best_individual = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    F = 0.7
    CR = 0.85
    
    for gen in range(max_iter):
        if time.time() > time_limit:
            break
        
        for i in range(pop_size):
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            
            v = population[r1] + F * (population[r2] - population[r3])
            v = _project_feasible(v, max_total, max_layer)
            
            trial = population[i].copy()
            j_rand = np.random.randint(n_params)
            for j in range(n_params):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = v[j]
            trial = _project_feasible(trial, max_total, max_layer)
            
            ft = eval_vec(trial)
            if ft < fitness[i]:
                population[i] = trial
                fitness[i] = ft
                if ft < best_fitness:
                    best_fitness = ft
                    best_individual = trial.copy()
    
    return best_individual.tolist(), best_fitness

def _simulated_annealing(materials, initial_thicks, problem, time_limit):
    """Simulated annealing for local refinement."""
    n_params = len(initial_thicks)
    n_vals = _n_from_materials(materials, problem["materials"]["L"], problem["materials"]["H"])
    lambdas = np.linspace(problem["band_nm"][0], problem["band_nm"][1], 401)
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    def eval_vec(x):
        if not _is_valid(x, max_total, max_layer):
            return float('inf')
        return _evaluate(n_vals, x, lambdas, n_ambient, n_substrate)

    x = _project_feasible(np.array(initial_thicks, dtype=float), max_total, max_layer)
    f_current = eval_vec(x)
    x_best = x.copy()
    f_best = f_current
    
    T = 10.0
    T_min = 0.001
    max_iter = 15000
    cooling = (T_min / T) ** (1.0 / max_iter)
    
    for _ in range(max_iter):
        if time.time() > time_limit:
            break
        
        # Random perturbation - larger steps early, smaller later
        step_scale = T / 10.0
        x_new = x.copy()
        for i in range(n_params):
            x_new[i] += np.random.randn() * step_scale
        x_new = _project_feasible(x_new, max_total, max_layer)
        
        f_new = eval_vec(x_new)
        
        if f_new < f_current:
            x = x_new
            f_current = f_new
            if f_new < f_best:
                f_best = f_new
                x_best = x_new.copy()
        else:
            if T > T_min and np.random.rand() < np.exp(-(f_new - f_current) / T):
                x = x_new
                f_current = f_new
        
        T *= cooling
        if T < T_min:
            break
    
    return x_best.tolist(), f_best

def design(problem):
    """Design an antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 145.0
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    lambdas = np.linspace(band[0], band[1], 401)
    center_lam = 0.5 * (band[0] + band[1])
    
    def time_ok(margin=5.0):
        return time.time() < deadline - margin
    
    def eval_design(materials, thicknesses):
        if len(materials) > max_layers:
            return float('inf')
        d = np.array(thicknesses, dtype=float)
        if not _is_valid(d, max_total_nm, max_layer_nm):
            return float('inf')
        n_vals = _n_from_materials(materials, n_L, n_H)
        return _evaluate(n_vals, d, lambdas, n_ambient, n_substrate)
    
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    # Try multiple layer configurations
    # Focus on odd number of layers (standard for AR coatings on glass)
    # Use a mix of configurations to explore the design space
    layer_configs = [5, 7, 9, 11, 13, 15]
    
    for n_layers in layer_configs:
        if not time_ok(20.0):
            break
        if n_layers > max_layers:
            continue
        
        # Generate multiple initial guesses for this layer count
        for init_type in ['qw', 'random1', 'random2']:
            if not time_ok(15.0):
                break
            
            materials = []
            thicknesses = []
            
            for i in range(n_layers):
                mat = 'L' if (i % 2 == 0) else 'H'
                materials.append(mat)
                n = n_L if mat == 'L' else n_H
                
                if init_type == 'qw':
                    thicknesses.append(center_lam / (4.0 * n))
                elif init_type == 'random1':
                    thicknesses.append(np.random.uniform(20, 200))
                else:
                    # Vary the quarter-wave thickness
                    thicknesses.append((center_lam / (4.0 * n)) * np.random.uniform(0.5, 1.5))
            
            # Project to feasible region
            d_arr = np.array(thicknesses, dtype=float)
            d_arr = _project_feasible(d_arr, max_total_nm, max_layer_nm)
            thicknesses = d_arr.tolist()
            
            initial_score = eval_design(materials, thicknesses)
            
            # Strategy 1: Differential Evolution (global search)
            if time_ok(12.0):
                bounds = [(0.0, max_layer_nm)] * n_layers
                de_thicks, de_score = _differential_evolution(materials, bounds, problem, 
                                                               pop_size=25, max_iter=150, 
                                                               time_limit=time.time() + 10.0)
                if de_score < best_score:
                    best_score = de_score
                    best_materials = materials[:]
                    best_thicknesses = de_thicks[:]
                current_best_thicks = de_thicks
            else:
                current_best_thicks = thicknesses
            
            # Strategy 2: Simulated Annealing
            if time_ok(10.0):
                sa_thicks, sa_score = _simulated_annealing(materials, current_best_thicks, problem, 
                                                            time_limit=time.time() + 8.0)
                if sa_score < best_score:
                    best_score = sa_score
                    best_materials = materials[:]
                    best_thicknesses = sa_thicks[:]
                current_best_thicks = sa_thicks
            
            # Strategy 3: Nelder-Mead polishing on best result
            if time_ok(6.0):
                nm_thicks, nm_score = _nelder_mead_optimize(materials, current_best_thicks, problem, 
                                                              time_limit=time.time() + 5.0)
                if nm_score < best_score:
                    best_score = nm_score
                    best_materials = materials[:]
                    best_thicknesses = nm_thicks[:]
    
    # Final refinement on the best result
    if time_ok(8.0) and best_materials is not None:
        final_thicks, final_score = _nelder_mead_optimize(best_materials, best_thicknesses, problem,
                                                           time_limit=time.time() + 6.0)
        if final_score < best_score:
            best_score = final_score
            best_thicknesses = final_thicks
    
    # Ensure valid output
    if best_materials is None:
        best_materials = ['L']
        best_thicknesses = [center_lam / (4.0 * n_L)]
    
    # Final validation
    d = np.array(best_thicknesses, dtype=float)
    d = _project_feasible(d, max_total_nm, max_layer_nm)
    best_thicknesses = d.tolist()
    
    return (best_materials, best_thicknesses)
# EVOLVE-BLOCK-END
