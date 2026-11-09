"""Broadband antireflection coating design: Differential Evolution on continuous 20-layer space."""
import time
import numpy as np
from scipy.optimize import differential_evolution


def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance (percent) over the band using characteristic matrices."""
    if not materials or len(materials) == 0:
        return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = np.array([n_mat[m] for m in materials], dtype=float)
    ds = np.asarray(thicknesses, dtype=float)

    # delta_j = 2*pi*n_j*d_j/lambda  -> (N, K)
    delta = 2.0 * np.pi * (ns[:, None] * ds[:, None]) / wavs[None, :]
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
        A_new = A * cj + 1j * B * n_j * sj
        B_new = 1j * A * sj / n_j + B * cj
        C_new = C * cj + 1j * D * n_j * sj
        D_new = 1j * C * sj / n_j + D * cj
        A, B, C, D = A_new, B_new, C_new, D_new

    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    r = num / den
    R = np.abs(r) ** 2
    return 100.0 * np.mean(R)


def design(problem):
    start_time = time.time()
    deadline = start_time + 140  # 150s kill, 10s safety

    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]

    # Precompute quarter-wave thicknesses at center wavelength
    qw = {}
    for m, n in materials_dict.items():
        qw[m] = wl_center / (4.0 * n)

    n_layers = max_layers  # Use all 20 layers as continuous variables

    def objective(x):
        """Objective function: mean reflectance for 20-layer design.
        x is a 20-element array of thicknesses.
        Material for layer j is determined by which quarter-wave reference is closer.
        """
        if time.time() > deadline:
            return 100.0  # Timeout penalty
        
        materials = []
        thicknesses = []
        total_th = 0.0
        
        for i in range(n_layers):
            d = x[i]
            if d < 1.0:  # Effectively zero thickness
                continue
            total_th += d
            # Determine material: closer to L or H quarter-wave
            if abs(d - qw['L']) < abs(d - qw['H']):
                materials.append('L')
            else:
                materials.append('H')
            thicknesses.append(d)
        
        # If no layers, return bare substrate reflectance
        if not materials:
            return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                            (problem["substrate_index"] + problem["ambient_index"])) ** 2
        
        # Check total thickness constraint
        if total_th > max_total:
            # Scale down proportionally
            scale = max_total / total_th
            thicknesses = [t * scale for t in thicknesses]
        
        # Check individual layer constraint
        for i in range(len(thicknesses)):
            if thicknesses[i] > max_layer:
                thicknesses[i] = max_layer
        
        return _mean_reflectance(materials, thicknesses, problem)

    # Define bounds for each thickness variable
    bounds = [(0.0, max_layer) for _ in range(n_layers)]

    # Run Differential Evolution
    try:
        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            mutation=(0.3, 1.0),
            recombination=0.9,
            seed=42,
            polish=False,  # We'll do our own polish
            init='latinhypercube',
            callback=None
        )
    except Exception:
        # Fallback to simple heuristic if DE fails
        materials = ['L']
        thicknesses = [qw['L']]
        return materials, thicknesses

    if time.time() > deadline:
        # Return best found so far if timeout
        best_x = result.x
        materials = []
        thicknesses = []
        total_th = 0.0
        for i in range(n_layers):
            d = best_x[i]
            if d < 1.0:
                continue
            total_th += d
            if abs(d - qw['L']) < abs(d - qw['H']):
                materials.append('L')
            else:
                materials.append('H')
            thicknesses.append(d)
        if not materials:
            return ['L'], [qw['L']]
        if total_th > max_total:
            scale = max_total / total_th
            thicknesses = [t * scale for t in thicknesses]
        for i in range(len(thicknesses)):
            if thicknesses[i] > max_layer:
                thicknesses[i] = max_layer
        return materials, thicknesses

    best_x = result.x
    
    # Extract materials and thicknesses from best solution
    materials = []
    thicknesses = []
    total_th = 0.0
    for i in range(n_layers):
        d = best_x[i]
        if d < 1.0:
            continue
        total_th += d
        if abs(d - qw['L']) < abs(d - qw['H']):
            materials.append('L')
        else:
            materials.append('H')
        thicknesses.append(d)
    
    if not materials:
        return ['L'], [qw['L']]
    
    # Enforce constraints
    if total_th > max_total:
        scale = max_total / total_th
        thicknesses = [t * scale for t in thicknesses]
    for i in range(len(thicknesses)):
        if thicknesses[i] > max_layer:
            thicknesses[i] = max_layer
    
    # Final polish with Nelder-Mead on the reduced problem
    if time.time() < deadline - 15 and len(materials) > 0:
        from scipy.optimize import minimize
        
        def obj_polish(x):
            if time.time() > deadline:
                return 100.0
            ths = list(x)
            # Constrain
            total = sum(ths)
            if total > max_total:
                scale = max_total / total
                ths = [t * scale for t in ths]
            for i in range(len(ths)):
                ths[i] = max(0.0, min(max_layer, ths[i]))
            return _mean_reflectance(materials, ths, problem)
        
        try:
            res = minimize(
                obj_polish,
                x0=np.array(thicknesses, dtype=float),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 0.1, 'fatol': 1e-8}
            )
            if res.success and res.fun < _mean_reflectance(materials, thicknesses, problem):
                final_ths = list(res.x)
                # Re-constrain
                total = sum(final_ths)
                if total > max_total:
                    scale = max_total / total
                    final_ths = [t * scale for t in final_ths]
                for i in range(len(final_ths)):
                    final_ths[i] = max(0.0, min(max_layer, final_ths[i]))
                thicknesses = final_ths
        except Exception:
            pass
    
    return materials, thicknesses
