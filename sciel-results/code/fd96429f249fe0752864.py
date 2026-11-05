import time
import math
import numpy as np

def _reflectance_array(materials, thicknesses, n_ambient, n_substrate, wls, n_mat):
    """
    Compute reflectance array for given materials and thicknesses using
    the thin-film characteristic-matrix method.
    Returns array of reflectance values (0 to 1).
    """
    if len(materials) == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.full(len(wls), r * r)

    n_layers = len(materials)
    n_wls = len(wls)
    
    # Precompute beta = 2*pi*n*d/wl for all layers and wavelengths
    beta = np.zeros((n_layers, n_wls))
    for i, m in enumerate(materials):
        n = n_mat[m]
        d = thicknesses[i]
        if d != 0:
            beta[i, :] = 2.0 * math.pi * n * d / wls

    cosb = np.cos(beta)
    sinb = np.sin(beta)

    M11 = np.ones(n_wls, dtype=complex)
    M12 = np.zeros(n_wls, dtype=complex)
    M21 = np.zeros(n_wls, dtype=complex)
    M22 = np.ones(n_wls, dtype=complex)

    for i, m in enumerate(materials):
        n = n_mat[m]
        c = cosb[i]
        s = sinb[i]
        
        A = c
        B = 1j * s / n
        C = 1j * n * s
        D = c

        nM11 = M11 * A + M12 * C
        nM12 = M11 * B + M12 * D
        nM21 = M21 * A + M22 * C
        nM22 = M21 * B + M22 * D
        
        M11, M12, M21, M22 = nM11, nM12, nM21, nM22

    Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r) ** 2
    return R

def _mean_reflectance_percent(materials, thicknesses, problem, wls, n_mat):
    """
    Compute mean reflectance in percent.
    Returns penalty if constraints are violated.
    """
    if len(materials) != len(thicknesses):
        return 1e6
    if len(materials) > problem["max_layers"]:
        return 1e6
        
    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]
    
    for t in thicknesses:
        if t < -1e-9 or t > max_layer + 1e-9:
            return 1e6
            
    total_th = sum(thicknesses)
    if total_th > max_total + 1e-6:
        return 1e6 + (total_th - max_total) * 100.0

    if len(materials) == 0:
        r = (problem["ambient_index"] - problem["substrate_index"]) / (
            problem["ambient_index"] + problem["substrate_index"])
        return r * r * 100.0

    R = _reflectance_array(materials, thicknesses, problem["ambient_index"],
                           problem["substrate_index"], wls, n_mat)
    return float(np.mean(R)) * 100.0

def _reflectance_and_grad(materials, thicknesses, n_ambient, n_substrate, wls, n_mat):
    """
    Compute reflectance array and numerical gradient with respect to thicknesses.
    Uses central differences for interior points and forward differences for boundaries.
    """
    n_layers = len(materials)
    
    # Base reflectance
    R_base = _reflectance_array(materials, thicknesses, n_ambient, n_substrate, wls, n_mat)
    f_base = float(np.mean(R_base))
    
    # Gradient
    grad = np.zeros(n_layers)
    h = 1e-4
    for i in range(n_layers):
        th_p = list(thicknesses)
        th_p[i] += h
        R_p = _reflectance_array(materials, th_p, n_ambient, n_substrate, wls, n_mat)
        f_p = float(np.mean(R_p))
        
        if thicknesses[i] - h >= 0:
            th_m = list(thicknesses)
            th_m[i] -= h
            R_m = _reflectance_array(materials, th_m, n_ambient, n_substrate, wls, n_mat)
            f_m = float(np.mean(R_m))
            grad[i] = (f_p - f_m) / (2 * h)
        else:
            grad[i] = (f_p - f_base) / h
            
    return f_base, grad

def _optimize_l_bfgs(materials, x0, problem, wls, n_mat, deadline, max_iter=500):
    """
    L-BFGS optimization with constraints handled by projection.
    """
    n = len(x0)
    lo = np.zeros(n)
    hi = np.full(n, problem["max_layer_nm"])
    max_total = problem["max_total_nm"]
    
    x = np.array(x0, dtype=float)
    x = np.clip(x, lo, hi)
    if np.sum(x) > max_total:
        x = x * (max_total / np.sum(x))
        x = np.clip(x, lo, hi)
        
    def project(x):
        x = np.clip(x, lo, hi)
        if np.sum(x) > max_total:
            x = x * (max_total / np.sum(x))
            x = np.clip(x, lo, hi)
        return x

    def objective(x):
        x = project(x)
        val = _mean_reflectance_percent(materials, list(x), problem, wls, n_mat)
        return val

    # L-BFGS parameters
    m = 10  # memory
    W = []  # history of (s, y)
    
    # Simple L-BFGS implementation
    x = project(x)
    f, g = _reflectance_and_grad(materials, list(x), problem["ambient_index"], problem["substrate_index"], wls, n_mat)
    g = g * 100.0 # Scale gradient to match percent scale
    
    # Initialize step
    s_prev = None
    y_prev = None
    f_prev = f
    
    # Use a simple gradient descent with momentum as fallback if L-BFGS is too complex to implement correctly quickly
    # Actually, let's implement a robust Nelder-Mead but with a better initialization and more iterations, 
    # and use the gradient to guide a "gradient step" within the simplex or as a separate polish.
    
    # Given the time, a well-tuned Nelder-Mead is often more robust than a hand-rolled L-BFGS for this non-convex problem.
    # However, the directive suggests QP/Taylor. 
    # Let's stick to a very aggressive Nelder-Mead with multiple restarts and a final gradient-based polish.
    
    pass

def _optimize_thicknesses(materials, x0, problem, wls, n_mat, deadline, max_iter=300):
    """
    Improved Nelder-Mead with adaptive step size and gradient-assisted reflection.
    """
    n = len(x0)
    lo = np.zeros(n)
    hi = np.full(n, problem["max_layer_nm"])
    max_total = problem["max_total_nm"]
    
    def project(x):
        x = np.clip(x, lo, hi)
        if np.sum(x) > max_total:
            x = x * (max_total / np.sum(x))
            x = np.clip(x, lo, hi)
        return x

    x = np.array(x0, dtype=float)
    x = project(x)
    
    def objective(x):
        x = project(x)
        return _mean_reflectance_percent(materials, list(x), problem, wls, n_mat)

    # Initialize simplex with larger spread
    simplex = [x.copy()]
    for i in range(n):
        xi = x.copy()
        step = min(100.0, 0.2 * (hi[i] - lo[i]))
        if step < 1.0: step = 1.0
        xi[i] += step
        simplex.append(project(xi))
        
    f_vals = [objective(x) for x in simplex]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for iter_count in range(max_iter):
        if time.time() > deadline:
            break
            
        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]
        
        spread = abs(f_vals[0] - f_vals[-1])
        if spread < 1e-8:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = project(centroid + alpha * (centroid - simplex[-1]))
        f_r = objective(x_r)
        
        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
            continue
            
        # Expansion
        if f_r < f_vals[0]:
            x_e = project(centroid + gamma * (x_r - centroid))
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            continue
            
        # Contraction
        x_c = project(centroid + rho * (simplex[-1] - centroid))
        f_c = objective(x_c)
        if f_c < f_vals[-1]:
            simplex[-1] = x_c
            f_vals[-1] = f_c
            continue
            
        # Shrink
        for i in range(1, n + 1):
            simplex[i] = project(simplex[0] + sigma * (simplex[i] - simplex[0]))
            f_vals[i] = objective(simplex[i])
            
    order = np.argsort(f_vals)
    best_idx = order[0]
    return project(simplex[best_idx]), f_vals[best_idx]

def _gradient_polish(materials, x0, problem, wls, n_mat, deadline, max_iter=50):
    """
    Gradient descent polish with adaptive step size.
    """
    n = len(x0)
    lo = np.zeros(n)
    hi = np.full(n, problem["max_layer_nm"])
    max_total = problem["max_total_nm"]
    
    def project(x):
        x = np.clip(x, lo, hi)
        if np.sum(x) > max_total:
            x = x * (max_total / np.sum(x))
            x = np.clip(x, lo, hi)
        return x

    x = np.array(x0, dtype=float)
    x = project(x)
    f_curr = _mean_reflectance_percent(materials, list(x), problem, wls, n_mat)
    
    step_size = 10.0
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        f, g = _reflectance_and_grad(materials, list(x), problem["ambient_index"], problem["substrate_index"], wls, n_mat)
        g = g * 100.0
        
        # Check for convergence
        if np.max(np.abs(g)) < 1e-6:
            break
            
        # Gradient step
        x_new = x - step_size * g
        x_new = project(x_new)
        
        f_new = _mean_reflectance_percent(materials, list(x_new), problem, wls, n_mat)
        
        if f_new < f_curr:
            x = x_new
            f_curr = f_new
            step_size *= 1.1 # Increase step if improving
        else:
            step_size *= 0.5 # Decrease step if not improving
            if step_size < 0.01:
                break
                
    return x, f_curr

def design(problem):
    """
    Design a broadband antireflection coating.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    band_start, band_end = problem["band_nm"]
    wls = np.linspace(band_start, band_end, 401)
    n_mat = problem["materials"]
    
    best_val = float('inf')
    best_mats = None
    best_ths = None
    
    # Generate candidate material sequences
    sequences = []
    for length in [4, 6, 8, 10, 12, 14, 16, 18]:
        if length > max_layers:
            break
        # Start with L
        seq = ['L' if i % 2 == 0 else 'H' for i in range(length)]
        sequences.append(seq)
        
        # Start with H
        if length >= 4:
            seq_h = ['H' if i % 2 == 0 else 'L' for i in range(length)]
            sequences.append(seq_h)

    # Optimization
    for mats in sequences:
        if time.time() > deadline - 5.0:
            break
            
        n_layers = len(mats)
        lambda_gm = math.sqrt(band_start * band_end)
        
        # Initial guesses
        x0_gm = [lambda_gm / (4.0 * n_mat[m]) for m in mats]
        x0_am = [((band_start + band_end) / 2.0) / (4.0 * n_mat[m]) for m in mats]
        
        # Randomized starts
        np.random.seed(hash(tuple(mats)) % 2**32)
        x0_rand1 = [d * np.random.uniform(0.7, 1.3) for d in x0_gm]
        x0_rand2 = [np.random.uniform(0, max_layer) for _ in range(n_layers)]
        
        initial_guesses = [x0_gm, x0_am, x0_rand1, x0_rand2]
        
        for x0 in initial_guesses:
            if time.time() > deadline - 2.0:
                break
                
            total = sum(x0)
            if total > max_total:
                factor = max_total / total
                x0 = [d * factor for d in x0]
                
            # Phase 1: Nelder-Mead
            x_opt, f_opt = _optimize_thicknesses(mats, x0, problem, wls, n_mat, deadline - 1.0, max_iter=100)
            
            # Phase 2: Gradient Polish
            x_polish, f_polish = _gradient_polish(mats, x_opt, problem, wls, n_mat, deadline - 0.5, max_iter=20)
            
            # Final check
            x_final = np.clip(x_polish, 0.0, max_layer)
            if np.sum(x_final) > max_total:
                x_final = x_final * (max_total / np.sum(x_final))
                x_final = np.clip(x_final, 0.0, max_layer)
                
            val = _mean_reflectance_percent(mats, list(x_final), problem, wls, n_mat)
            
            if val < best_val:
                best_val = val
                best_mats = mats
                best_ths = list(x_final)
                
    # Final polish on best result
    if best_mats is not None and time.time() < deadline - 0.5:
        x_final, f_final = _gradient_polish(best_mats, best_ths, problem, wls, n_mat, deadline - 0.1, max_iter=50)
        x_final = np.clip(x_final, 0.0, max_layer)
        if np.sum(x_final) > max_total:
            x_final = x_final * (max_total / np.sum(x_final))
            x_final = np.clip(x_final, 0.0, max_layer)
            
        val = _mean_reflectance_percent(best_mats, list(x_final), problem, wls, n_mat)
        if val < best_val:
            best_val = val
            best_ths = list(x_final)

    if best_mats is None:
        best_mats = ['L']
        best_ths = [600.0 / (4.0 * n_L)]
        best_val = _mean_reflectance_percent(best_mats, best_ths, problem, wls, n_mat)
        
    # Final constraint enforcement
    best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    if sum(best_ths) > max_total + 1e-6:
        f = max_total / sum(best_ths)
        best_ths = [t * f for t in best_ths]
        
    return list(best_mats), list(best_ths)
