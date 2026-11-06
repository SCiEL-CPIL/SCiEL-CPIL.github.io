"""
Broadband antireflection coating design via CMA-ES optimization.

Strategy:
- Use CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for global optimization.
- CMA-ES is superior to Nelder-Mead for this problem due to better handling of correlated variables and rugged landscapes.
- Search over several layer configurations (7, 9, 11 layers) with alternating L/H indices.
- Polish the best result with coordinate descent.
"""
import numpy as np
import time

def _compute_reflectance(wavelengths_nm, indices, thicknesses_nm, ambient_idx, substrate_idx):
    """
    Compute reflectance (fraction, 0-1) at each wavelength using the characteristic matrix method.
    """
    n_layers = len(indices)
    if n_layers == 0:
        r0 = (ambient_idx - substrate_idx) / (ambient_idx + substrate_idx)
        return np.full(len(wavelengths_nm), r0 * r0)
    
    k = 2.0 * np.pi / wavelengths_nm
    
    M00 = np.ones(len(wavelengths_nm), dtype=complex)
    M01 = np.zeros(len(wavelengths_nm), dtype=complex)
    M10 = np.zeros(len(wavelengths_nm), dtype=complex)
    M11 = np.ones(len(wavelengths_nm), dtype=complex)
    
    for i in range(n_layers):
        n_i = indices[i]
        d_i = thicknesses_nm[i]
        delta = k * n_i * d_i
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        B00 = cos_d
        B01 = 1j * sin_d / n_i
        B10 = 1j * n_i * sin_d
        B11 = cos_d
        
        new_M00 = M00 * B00 + M01 * B10
        new_M01 = M00 * B01 + M01 * B11
        new_M10 = M10 * B00 + M11 * B10
        new_M11 = M10 * B01 + M11 * B11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    Y_out = substrate_idx
    Y_in = (M10 + M11 * Y_out) / (M00 + M01 * Y_out)
    
    r = (ambient_idx - Y_in) / (ambient_idx + Y_in)
    R = np.abs(r) ** 2
    
    return R

def _objective(thicknesses, indices, wavelengths_nm, ambient_idx, substrate_idx, 
               max_layer_nm, max_total_nm):
    """
    Objective: mean reflectance in percent. Returns large penalty if constraints violated.
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    if np.any(thicknesses < -1e-6) or np.any(thicknesses > max_layer_nm + 1e-6):
        return 100.0
    if np.sum(thicknesses) > max_total_nm + 1e-6:
        return 100.0
    
    thicknesses_clipped = np.clip(thicknesses, 0.0, max_layer_nm)
    if np.sum(thicknesses_clipped) > max_total_nm:
        return 100.0
    
    R = _compute_reflectance(wavelengths_nm, indices, thicknesses_clipped, 
                             ambient_idx, substrate_idx)
    return np.mean(R) * 100.0

def _cmaes(f, x0, bounds, max_iter=100, pop_size=None, time_deadline=None, seed=42):
    """
    CMA-ES implementation with box constraints.
    """
    n = len(x0)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    
    x0 = np.clip(np.asarray(x0, dtype=float), lo, hi)
    if pop_size is None:
        pop_size = max(4 + int(3 * np.log(n)), 10)
    pop_size = max(pop_size, n + 1)
    
    # Initial parameters
    sigma = 0.1 * np.max(hi - lo)
    x_mean = x0.copy()
    fval = f(x_mean)
    
    # Covariance matrix
    C = np.diag((hi - lo) ** 2 / 4.0)
    
    # Learning rates
    mu = pop_size // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mu_eff = 1.0 / np.sum(weights ** 2)
    
    c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
    c_c = (4 - mu_eff + c_sigma) / (n + 4 - mu_eff)
    c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
    c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
    
    chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
    
    p_sigma = np.zeros(n)
    p_c = np.zeros(n)
    
    # Cholesky factor
    try:
        B = np.linalg.cholesky(C)
    except:
        B = np.sqrt(np.diag(C)).reshape(n, 1)
    
    rng = np.random.RandomState(seed)
    
    for it in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break
        
        # Generate population
        population = np.zeros((pop_size, n))
        for i in range(pop_size):
            z = rng.randn(n)
            y = B @ z
            population[i] = x_mean + sigma * y
            population[i] = np.clip(population[i], lo, hi)
        
        # Evaluate
        fvals = np.array([f(population[i]) for i in range(pop_size)])
        
        # Sort
        idx = np.argsort(fvals)
        population = population[idx]
        fvals = fvals[idx]
        
        # Update mean
        new_x_mean = np.sum(weights[:, None] * population[:mu], axis=0)
        
        # Update step size
        p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ (new_x_mean - x_mean) / sigma)
        if np.linalg.norm(p_sigma) / chi_n < (1.4 + 1 / mu_eff) * np.sqrt((1 - c_sigma) / (1 - c_sigma ** (2 * (it + 1)))):
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
        
        # Update covariance
        p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_x_mean - x_mean) / sigma
        C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (c_c * (2 - c_c) - 1) * C)
        
        delta_h = 1 if np.linalg.norm(p_c) / chi_n < (1.4 + 1 / mu_eff) * np.sqrt((1 - c_c) / (1 - c_c ** (2 * (it + 1)))) else 0
        C = (1 - c_1 - c_mu + c_mu * delta_h) * C + c_mu * np.sum(weights[:, None, None] * np.outer(population[:mu] - x_mean, population[:mu] - x_mean) / sigma ** 2, axis=0)
        
        # Ensure symmetry and positive semi-definite
        C = (C + C.T) / 2
        try:
            B = np.linalg.cholesky(C + 1e-12 * np.eye(n))
        except:
            B = np.sqrt(np.diag(np.maximum(np.diag(C), 1e-12))).reshape(n, 1)
        
        x_mean = new_x_mean
        x_mean = np.clip(x_mean, lo, hi)
        fval = f(x_mean)
        
        # Convergence check
        if np.all(np.abs(x_mean - lo) < 1e-6) or np.all(np.abs(x_mean - hi) < 1e-6):
            break
    
    return x_mean, fval

def _coordinate_descent(f, x0, bounds, max_iter=100, time_deadline=None):
    """
    Cyclic coordinate descent with golden-section line search per coordinate.
    """
    n = len(x0)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    x = np.clip(np.asarray(x0, dtype=float), lo, hi)
    fval = f(x)
    
    for it in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break
        
        improved = False
        for i in range(n):
            # Golden section search on dimension i
            a = lo[i]
            b = hi[i]
            x_i = x[i]
            
            # Define function of single variable
            def g(t):
                x_temp = x.copy()
                x_temp[i] = t
                return f(x_temp)
            
            # Simple line search: try to improve around current x_i
            # We'll just do a few steps of descent
            step = 0.1 * (hi[i] - lo[i])
            for _ in range(10):
                x_plus = x.copy()
                x_plus[i] = min(x_i + step, hi[i])
                x_minus = x.copy()
                x_minus[i] = max(x_i - step, lo[i])
                
                f_plus = f(x_plus)
                f_minus = f(x_minus)
                
                if f_plus < fval:
                    x = x_plus
                    fval = f_plus
                    x_i = x[i]
                    improved = True
                elif f_minus < fval:
                    x = x_minus
                    fval = f_minus
                    x_i = x[i]
                    improved = True
                else:
                    step *= 0.5
                    if step < 1e-4:
                        break
        
        if not improved:
            break
            
    return x, fval

def design(problem):
    """
    Design a broadband antireflection coating.
    Returns: (labels, thicknesses)
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    best_labels = None
    best_thicknesses = None
    best_score = np.inf
    
    # Define candidate configurations
    # Alternating L/H is standard. We try a few lengths.
    configs = []
    for n_layers in [7, 9, 11, 13]:
        if n_layers > max_layers:
            continue
        # Start with L (air side)
        labels_l = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        configs.append(labels_l)
        # Start with H
        labels_h = ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
        configs.append(labels_h)
    
    # We will allocate time to each config.
    # Let's run CMA-ES for each config.
    
    for labels in configs:
        if time.time() > deadline - 10:
            break
        
        indices = [materials[label] for label in labels]
        n_layers = len(indices)
        bounds = [(0.0, float(max_layer_nm))] * n_layers
        
        # Initial guess: quarter-wave stack
        x0 = np.array([center_wl / (4.0 * n_i) for n_i in indices])
        if np.sum(x0) > max_total_nm:
            x0 = x0 * (max_total_nm / np.sum(x0))
        x0 = np.clip(x0, 0, max_layer_nm)
        
        def make_f(indices):
            def f(x):
                return _objective(x, indices, wavelengths_nm, ambient_idx, substrate_idx,
                                  max_layer_nm, max_total_nm)
            return f
        
        f = make_f(indices)
        
        # Run CMA-ES
        # Time slice: use up to 20 seconds per config, or until deadline
        cma_deadline = min(time.time() + 20.0, deadline - 5)
        
        try:
            x_cma, f_cma = _cmaes(f, x0, bounds, max_iter=200, pop_size=30, 
                                   time_deadline=cma_deadline, seed=42)
            
            if f_cma < best_score:
                best_score = f_cma
                best_labels = labels.copy()
                best_thicknesses = x_cma.copy()
                
        except Exception:
            pass
        
        # Polish with coordinate descent if we have time and a good result
        if best_labels == labels and time.time() < deadline - 5:
            gd_deadline = min(time.time() + 5.0, deadline - 2)
            try:
                x_gd, f_gd = _coordinate_descent(f, best_thicknesses, bounds, 
                                                 max_iter=100, time_deadline=gd_deadline)
                if f_gd < best_score:
                    best_score = f_gd
                    best_thicknesses = x_gd.copy()
            except Exception:
                pass
    
    # Final constraint check
    if best_labels is None:
        # Fallback
        best_labels = ["L"]
        best_thicknesses = np.array([center_wl / (4.0 * materials["L"])])
    
    thicknesses = np.array(best_thicknesses)
    thicknesses = np.clip(thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return best_labels, thicknesses.tolist()
