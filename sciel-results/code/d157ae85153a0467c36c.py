"""Broadband antireflection coating design with admittance-trajectory parameterization + DE/SA/NM."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _reflectance_vectorized(wavelengths, thicknesses, indices, ambient_n, substrate_n):
    """Compute reflectance spectrum using vectorized characteristic matrix method."""
    n_layers = len(thicknesses)
    if n_layers == 0:
        r0 = (ambient_n - substrate_n) / (ambient_n + substrate_n)
        return np.full(len(wavelengths), np.abs(r0) ** 2)
    
    M00 = np.ones(len(wavelengths), dtype=complex)
    M01 = np.zeros(len(wavelengths), dtype=complex)
    M10 = np.zeros(wavelengths, dtype=complex)
    M11 = np.ones(wavelengths, dtype=complex)
    
    for j in range(n_layers):
        n_j = indices[j]
        d_j = thicknesses[j]
        
        if d_j < 1e-10:
            continue
        
        delta_j = 2.0 * np.pi * n_j * d_j / wavelengths
        cos_d = np.cos(delta_j)
        sin_d = np.sin(delta_j)
        
        m00 = cos_d
        m01 = 1j * sin_d / n_j
        m10 = 1j * n_j * sin_d
        m11 = cos_d
        
        new_M00 = M00 * m00 + M01 * m10
        new_M01 = M00 * m01 + M01 * m11
        new_M10 = M10 * m00 + M11 * m10
        new_M11 = M10 * m01 + M11 * m11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    B = M00 + M01 * substrate_n
    C = M10 * ambient_n + M11 * substrate_n
    
    denom = C + B
    denom_safe = np.where(np.abs(denom) < 1e-15, 1e-15, denom)
    
    r = (C - B) / denom_safe
    reflectance = np.abs(r) ** 2
    
    return reflectance

def _evaluate(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    wavelengths = np.linspace(problem["band_nm"][0], problem["band_nm"][1], 401)
    indices = np.array([problem["materials"][m] for m in materials], dtype=float)
    thicknesses_arr = np.array(thicknesses, dtype=float)
    refl = _reflectance_vectorized(wavelengths, thicknesses_arr, indices,
                                   problem["ambient_index"], problem["substrate_index"])
    return 100.0 * np.mean(refl)

def _is_valid(materials, thicknesses, problem):
    """Check if the design is valid."""
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > problem["max_layers"]:
        return False
    if len(materials) < 0:
        return False
    if any(t < -1e-9 or t > problem["max_layer_nm"] + 1e-9 for t in thicknesses):
        return False
    if sum(thicknesses) > problem["max_total_nm"] + 1e-9:
        return False
    if any(m not in problem["materials"] for m in materials):
        return False
    return True

def _project_to_feasible(x, max_total_nm, max_layer_nm):
    """Project a candidate onto the feasible region."""
    x = np.clip(x, 0.0, max_layer_nm)
    total = x.sum()
    if total > max_total_nm:
        x = x * (max_total_nm / total)
    return x

def _nelder_mead(f, x0, bounds, maxiter=200, tol=1e-8, time_limit=None, start_time=None):
    """Nelder-Mead optimization with bounds and time limit."""
    if time_limit is None:
        time_limit = 10.0
    if start_time is None:
        start_time = time.time()
    
    n = len(x0)
    x0 = np.array(x0, dtype=float)
    
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        if abs(xi[i]) > 1e-10:
            xi[i] *= 1.05
        else:
            xi[i] += 0.05 * (bounds[i][1] - bounds[i][0])
        simplex.append(xi)
    
    f_values = [f(x) for x in simplex]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for _ in range(maxiter):
        if time.time() - start_time > time_limit:
            break
        
        order = np.argsort(f_values)
        simplex = [simplex[i] for i in order]
        f_values = [f_values[i] for i in order]
        
        if abs(f_values[-1] - f_values[0]) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        xr = centroid + alpha * (centroid - simplex[-1])
        xr = np.clip(xr, [b[0] for b in bounds], [b[1] for b in bounds])
        fr = f(xr)
        
        if fr < f_values[0]:
            xe = centroid + gamma * (centroid - simplex[-1])
            xe = np.clip(xe, [b[0] for b in bounds], [b[1] for b in bounds])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                f_values[-1] = fe
            else:
                simplex[-1] = xr
                f_values[-1] = fr
        elif fr < f_values[-2]:
            simplex[-1] = xr
            f_values[-1] = fr
        else:
            if fr < f_values[-1]:
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = f(xc)
            else:
                xc = centroid - rho * (centroid - simplex[-1])
                fc = f(xc)
            xc = np.clip(xc, [b[0] for b in bounds], [b[1] for b in bounds])
            fc = f(xc)
            
            if fc <= fr:
                simplex[-1] = xc
                f_values[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    f_values[i] = f(simplex[i])
    
    best_idx = np.argmin(f_values)
    return simplex[best_idx], f_values[best_idx]

def _differential_evolution(objective, bounds, pop_size=40, max_iter=300, F=0.8, CR=0.9,
                            time_limit=30.0, start_time=None):
    """Differential Evolution optimization."""
    if start_time is None:
        start_time = time.time()
    
    n = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    population = np.random.uniform(lb, ub, size=(pop_size, n))
    fitness = np.array([objective(ind) for ind in population])
    
    best_idx = np.argmin(fitness)
    best_individual = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    for gen in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        for i in range(pop_size):
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            
            v = population[r1] + F * (population[r2] - population[r3])
            v = np.clip(v, lb, ub)
            
            trial = population[i].copy()
            j_rand = np.random.randint(n)
            for j in range(n):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = v[j]
            
            ft = objective(trial)
            if ft < fitness[i]:
                population[i] = trial
                fitness[i] = ft
                if ft < best_fitness:
                    best_fitness = ft
                    best_individual = trial.copy()
    
    return best_individual, best_fitness

def _simulated_annealing(objective, x0, bounds, max_iter=50000, T0=1.0, T1=0.001,
                         time_limit=20.0, start_time=None):
    """Simulated Annealing with diverse move operators."""
    if start_time is None:
        start_time = time.time()
    
    n = len(x0)
    x = np.array(x0, dtype=float)
    f_current = objective(x)
    x_best = x.copy()
    f_best = f_current
    
    T = T0
    cooling_factor = (T1 / T0) ** (1.0 / max_iter)
    
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        move_type = np.random.random()
        x_new = x.copy()
        
        if move_type < 0.45:
            idx = np.random.randint(n)
            range_size = ub[idx] - lb[idx]
            x_new[idx] += range_size * 0.15 * np.random.randn()
        elif move_type < 0.60:
            idx1, idx2 = np.random.choice(n, 2, replace=False)
            for idx in [idx1, idx2]:
                range_size = ub[idx] - lb[idx]
                x_new[idx] += range_size * 0.15 * np.random.randn()
        elif move_type < 0.70:
            idx = np.random.randint(n)
            range_size = ub[idx] - lb[idx]
            x_new[idx] += range_size * 0.5 * np.random.randn()
        elif move_type < 0.80:
            idx = np.random.randint(n)
            x_new[idx] = np.random.uniform(lb[idx], ub[idx])
        elif move_type < 0.90:
            idx1, idx2 = np.random.choice(n, 2, replace=False)
            x_new[idx1], x_new[idx2] = x_new[idx2], x_new[idx1]
        else:
            n_restart = max(1, n // 4)
            idxs = np.random.choice(n, n_restart, replace=False)
            for idx in idxs:
                x_new[idx] = np.random.uniform(lb[idx], ub[idx])
        
        x_new = np.clip(x_new, lb, ub)
        
        f_new = objective(x_new)
        
        if f_new < f_current:
            x = x_new
            f_current = f_new
            if f_new < f_best:
                x_best = x_new.copy()
                f_best = f_new
        else:
            delta_f = f_new - f_current
            if delta_f < 0:
                accept_prob = 1.0
            else:
                accept_prob = np.exp(-delta_f / max(T, 1e-10))
            
            if np.random.rand() < accept_prob:
                x = x_new
                f_current = f_new
        
        T *= cooling_factor
    
    return x_best, f_best

def _admittance_surrogate(u, materials, problem, wavelengths, n_order=3):
    """
    Fast surrogate for mean reflectance using truncated Taylor expansion of
    the characteristic matrix in phase angles.
    
    u: array of continuous variables in [0,1], one per layer.
       thickness_j = u_j * max_layer_nm
    """
    n_layers = len(u)
    if n_layers == 0:
        r0 = (problem["ambient_index"] - problem["substrate_index"]) / \
             (problem["ambient_index"] + problem["substrate_index"])
        return 100.0 * np.abs(r0) ** 2
    
    thicknesses = u * problem["max_layer_nm"]
    indices = np.array([problem["materials"][m] for m in materials], dtype=float)
    
    # Compute phase deltas for all wavelengths
    # delta_j = 2*pi*n_j*d_j / lambda
    # For the surrogate, we use the mean wavelength and a small bandwidth approximation
    mean_wl = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    half_bw = 0.5 * (problem["band_nm"][1] - problem["band_nm"][0])
    
    # Use a few sample wavelengths for the surrogate (much fewer than 401)
    n_samples = 21
    wl_samples = np.linspace(problem["band_nm"][0], problem["band_nm"][1], n_samples)
    
    M00 = np.ones(n_samples, dtype=complex)
    M01 = np.zeros(n_samples, dtype=complex)
    M10 = np.zeros(n_samples, dtype=complex)
    M11 = np.ones(n_samples, dtype=complex)
    
    for j in range(n_layers):
        n_j = indices[j]
        d_j = thicknesses[j]
        
        if d_j < 1e-10:
            continue
        
        delta_j = 2.0 * np.pi * n_j * d_j / wl_samples
        cos_d = np.cos(delta_j)
        sin_d = np.sin(delta_j)
        
        m00 = cos_d
        m01 = 1j * sin_d / n_j
        m10 = 1j * n_j * sin_d
        m11 = cos_d
        
        new_M00 = M00 * m00 + M01 * m10
        new_M01 = M00 * m01 + M01 * m11
        new_M10 = M10 * m00 + M11 * m10
        new_M11 = M10 * m01 + M11 * m11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    B = M00 + M01 * problem["substrate_index"]
    C = M10 * problem["ambient_index"] + M11 * problem["substrate_index"]
    
    denom = C + B
    denom_safe = np.where(np.abs(denom) < 1e-15, 1e-15, denom)
    
    r = (C - B) / denom_safe
    reflectance = np.abs(r) ** 2
    
    return 100.0 * np.mean(reflectance)

def _gradient_descent_admittance(f, x0, bounds, lr=0.01, max_iter=5000,
                                  time_limit=20.0, start_time=None):
    """Gradient descent with numerical gradients in admittance space."""
    if start_time is None:
        start_time = time.time()
    
    n = len(x0)
    x = np.array(x0, dtype=float)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    f_current = f(x)
    x_best = x.copy()
    f_best = f_current
    
    eps = 1e-4
    momentum = np.zeros(n)
    beta1 = 0.9
    beta2 = 0.999
    m = np.zeros(n)
    v = np.zeros(n)
    t = 0
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute numerical gradient
        grad = np.zeros(n)
        for j in range(n):
            x_plus = x.copy()
            x_plus[j] = min(x_plus[j] + eps, ub[j])
            x_minus = x.copy()
            x_minus[j] = max(x_minus[j] - eps, lb[j])
            
            if x_plus[j] == x_minus[j]:
                continue
            
            grad[j] = (f(x_plus) - f(x_minus)) / (x_plus[j] - x_minus[j])
        
        # Adam update
        t += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        x_new = x - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        x_new = np.clip(x_new, lb, ub)
        
        f_new = f(x_new)
        
        if f_new < f_best:
            x_best = x_new.copy()
            f_best = f_new
        
        x = x_new
        f_current = f_new
        
        # Early stopping if converged
        if np.max(np.abs(grad)) < 1e-6:
            break
    
    return x_best, f_best

def design(problem):
    """Design antireflection coating using admittance-trajectory optimization."""
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s wall-clock
    
    L_index = problem["materials"]["L"]
    H_index = problem["materials"]["H"]
    center_wl = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    best_overall_score = float('inf')
    best_overall_materials = ["L"]
    best_overall_thicknesses = [center_wl / (4.0 * L_index)]
    
    # Layer configurations to try
    layer_configs = []
    
    for n_layers in [4, 6, 8, 10, 12, 14]:
        if n_layers > max_layers:
            continue
        # Pattern 1: LHLH...
        pat1 = ["L" if i % 2 == 0 else "H" for i in range(n_layers)]
        # Pattern 2: HLHL...
        pat2 = ["H" if i % 2 == 0 else "L" for i in range(n_layers)]
        # Pattern 3: LLHH...
        pat3 = ["L" if i % 4 < 2 else "H" for i in range(n_layers)]
        # Pattern 4: HHLH...
        pat4 = ["H" if i % 4 < 2 else "L" for i in range(n_layers)]
        
        for pat in [pat1, pat2, pat3, pat4]:
            layer_configs.append(pat)
    
    # Sort by expected quality
    def config_priority(config):
        n = len(config)
        score = 0
        if n % 2 == 0:
            score += 1
        if config[0] == "L":
            score += 1
        if 8 <= n <= 12:
            score += 2
        return -score
    
    layer_configs.sort(key=config_priority)
    
    # Phase 1: Fast surrogate-based optimization to find good starting points
    # Phase 2: Exact evaluation + refinement
    
    for materials in layer_configs:
        if time.time() - start_time > time_limit - 20:
            break
        
        n_layers = len(materials)
        if n_layers > max_layers:
            continue
        
        # Check if total quarter-wave thickness is feasible
        qw_thicknesses = [center_wl / (4.0 * problem["materials"][m]) for m in materials]
        if sum(qw_thicknesses) > max_total_nm:
            continue
        
        # Admittance space: u in [0,1], thickness = u * max_layer_nm
        bounds_u = [(0.0, 1.0)] * n_layers
        max_total_u = max_total_nm / max_layer_nm  # max sum of u values
        
        def project_u(u):
            """Project u to feasible region (sum constraint)."""
            u = np.clip(u, 0.0, 1.0)
            total = u.sum()
            if total > max_total_u:
                u = u * (max_total_u / total)
            return u
        
        # Surrogate objective (fast, 21 wavelengths)
        def surrogate_obj(u):
            u = project_u(u)
            return _admittance_surrogate(u, materials, problem, None)
        
        # Exact objective (401 wavelengths)
        def exact_obj(u):
            u = project_u(u)
            thicknesses = list(u * max_layer_nm)
            return _evaluate(materials, thicknesses, problem)
        
        # Multiple starts in admittance space
        n_starts = 4
        best_u_for_config = None
        best_f_for_config = float('inf')
        
        for start_i in range(n_starts):
            if time.time() - start_time > time_limit - 15:
                break
            
            # Generate initial u
            if start_i == 0:
                # Quarter-wave normalized
                u0 = np.array([t / max_layer_nm for t in qw_thicknesses])
            elif start_i == 1:
                # Quarter-wave with noise
                u0 = np.array([t / max_layer_nm for t in qw_thicknesses])
                u0 += 0.1 * np.random.randn(n_layers)
            elif start_i == 2:
                # Random
                u0 = np.random.uniform(0, 1, n_layers)
            else:
                # Half-wave based
                hw = [center_wl / (2.0 * problem["materials"][m]) for m in materials]
                u0 = np.array([t / max_layer_nm for t in hw])
            
            u0 = project_u(u0)
            
            # Step 1: Gradient descent on surrogate (fast)
            gd_time = min(8.0, (time_limit - (time.time() - start_time)) * 0.3)
            if gd_time < 2.0:
                break
            
            u_gd, f_gd = _gradient_descent_admittance(
                surrogate_obj, u0, bounds_u, lr=0.05, max_iter=3000,
                time_limit=gd_time, start_time=start_time
            )
            
            # Step 2: DE on surrogate for broader exploration
            de_time = min(6.0, (time_limit - (time.time() - start_time)) * 0.3)
            if de_time < 2.0:
                break
            
            u_de, f_de = _differential_evolution(
                surrogate_obj, bounds_u, pop_size=25, max_iter=150,
                F=0.8, CR=0.9, time_limit=de_time, start_time=start_time
            )
            
            # Step 3: SA on surrogate
            sa_time = min(5.0, (time_limit - (time.time() - start_time)) * 0.3)
            if sa_time < 1.5:
                break
            
            # Use best of GD and DE as starting point for SA
            if f_gd < f_de:
                u_start_sa = u_gd
            else:
                u_start_sa = u_de
            
            u_sa, f_sa = _simulated_annealing(
                surrogate_obj, u_start_sa, bounds_u, max_iter=20000,
                T0=0.1, T1=0.001, time_limit=sa_time, start_time=start_time
            )
            
            # Track best surrogate result
            best_u_candidate = None
            best_f_candidate = float('inf')
            for u_cand, f_cand in [(u_gd, f_gd), (u_de, f_de), (u_sa, f_sa)]:
                if f_cand < best_f_candidate:
                    best_f_candidate = f_cand
                    best_u_candidate = u_cand
            
            # Step 4: NM polish on exact objective
            nm_time = min(5.0, (time_limit - (time.time() - start_time)) * 0.4)
            if nm_time < 1.0:
                # Use surrogate best directly
                if best_f_candidate < best_f_for_config:
                    best_f_for_config = best_f_candidate
                    best_u_for_config = best_u_candidate
                continue
            
            # Convert surrogate best to exact and polish
            u_exact_start = best_u_candidate.copy()
            
            x_nm, f_nm = _nelder_mead(
                exact_obj, u_exact_start, bounds_u, maxiter=150, tol=1e-8,
                time_limit=nm_time, start_time=start_time
            )
            
            if f_nm < best_f_for_config:
                best_f_for_config = f_nm
                best_u_for_config = x_nm
        
        # Update global best with exact evaluation
        if best_u_for_config is not None:
            u_final = project_u(best_u_for_config)
            thicknesses_opt = [float(u * max_layer_nm) for u in u_final]
            
            if _is_valid(materials, thicknesses_opt, problem):
                score = _evaluate(materials, thicknesses_opt, problem)
                if score < best_overall_score:
                    best_overall_score = score
                    best_overall_materials = list(materials)
                    best_overall_thicknesses = thicknesses_opt
    
    # Final refinement on the best configuration found
    if time.time() - start_time < time_limit - 8:
        n_layers = len(best_overall_materials)
        bounds_u = [(0.0, 1.0)] * n_layers
        max_total_u = max_total_nm / max_layer_nm
        
        def project_u(u):
            u = np.clip(u, 0.0, 1.0)
            total = u.sum()
            if total > max_total_u:
                u = u * (max_total_u / total)
            return u
        
        def exact_obj(u):
            u = project_u(u)
            thicknesses = list(u * max_layer_nm)
            return _evaluate(best_overall_materials, thicknesses, problem)
        
        u0 = np.array([t / max_layer_nm for t in best_overall_thicknesses])
        
        # SA refinement
        sa_time = min(15.0, time_limit - (time.time() - start_time) - 5)
        if sa_time > 3.0:
            u_sa, f_sa = _simulated_annealing(
                exact_obj, u0, bounds_u, max_iter=40000,
                T0=0.05, T1=0.0001, time_limit=sa_time, start_time=start_time
            )
            u0 = u_sa
        
        # NM polish
        nm_time = min(8.0, time_limit - (time.time() - start_time) - 2)
        if nm_time > 2.0:
            u_nm, f_nm = _nelder_mead(
                exact_obj, u0, bounds_u, maxiter=200, tol=1e-10,
                time_limit=nm_time, start_time=start_time
            )
            u_final = project_u(u_nm)
            thicknesses_opt = [float(u * max_layer_nm) for u in u_final]
            
            if _is_valid(best_overall_materials, thicknesses_opt, problem):
                score = _evaluate(best_overall_materials, thicknesses_opt, problem)
                if score < best_overall_score:
                    best_overall_score = score
                    best_overall_thicknesses = thicknesses_opt
    
    # Fallback
    if not _is_valid(best_overall_materials, best_overall_thicknesses, problem):
        best_overall_materials = ["L"]
        best_overall_thicknesses = [center_wl / (4.0 * L_index)]
    
    return best_overall_materials, best_overall_thicknesses
# EVOLVE-BLOCK-END
