"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _reflectance_spectrum(materials, thicknesses, n_layers, ambient_index, substrate_index, wavelengths):
    """
    Compute reflectance spectrum using characteristic-matrix method.
    Returns array of reflectance values (0 to 1) for each wavelength.
    """
    n_w = len(wavelengths)
    R = np.zeros(n_w)
    
    for i in range(n_layers):
        n_i = materials[i]
        d_i = thicknesses[i]
        # Optical thickness in radians: 2*pi*n*d/lambda
        # For vectorized: we need to compute for each wavelength
        # Actually, let's vectorize over wavelengths
        pass
    
    # Vectorized version
    # wavelengths: (n_w,)
    # thicknesses: (n_layers,)
    # materials: (n_layers,)
    
    # Build the product of matrices for each wavelength
    # M_i = [[cos(delta_i), i*sin(delta_i)/n_i], [i*n_i*sin(delta_i), cos(delta_i)]]
    # where delta_i = 2*pi*n_i*d_i/lambda
    
    # For each wavelength, compute the total matrix M = M_1 @ M_2 @ ... @ M_n
    # Then Y = M[1,0]/M[0,0] (normalized by n_0? No, the admittance)
    # Actually, the characteristic matrix gives us:
    # [B, C]^T = M @ [1, n_s]^T where n_s is substrate index
    # Then R = |(n_0*B - C)/(n_0*B + C)|^2
    
    # Let's vectorize: for each wavelength, compute all delta values
    # delta = 2*pi*n*d/lambda, shape (n_w, n_layers)
    delta = 2.0 * np.pi * np.outer(thicknesses, 1.0 / wavelengths) * np.outer(1.0, materials)
    # Wait, that's not right. Let me be more careful.
    # delta[i, j] = 2*pi*materials[j]*thicknesses[j]/wavelengths[i]
    delta = 2.0 * np.pi * np.outer(1.0 / wavelengths, materials * thicknesses)
    # delta shape: (n_w, n_layers)
    
    cos_d = np.cos(delta)  # (n_w, n_layers)
    sin_d = np.sin(delta)  # (n_w, n_layers)
    
    # For each layer, the matrix is:
    # [[cos(d), 1j*sin(d)/n], [1j*n*sin(d), cos(d)]]
    # We need to multiply these matrices in order (from air side to substrate)
    
    # Initialize the running product as identity for each wavelength
    # M_total = I (2x2)
    # We'll store M_total[0,0], M_total[0,1], M_total[1,0], M_total[1,1]
    M00 = np.ones(n_w, dtype=complex)
    M01 = np.zeros(n_w, dtype=complex)
    M10 = np.zeros(n_w, dtype=complex)
    M11 = np.ones(n_w, dtype=complex)
    
    for j in range(n_layers):
        c = cos_d[:, j]
        s = sin_d[:, j]
        n_j = materials[j]
        
        # New M = M_old @ M_j
        # M_j = [[c, 1j*s/n_j], [1j*n_j*s, c]]
        new_M00 = M00 * c + M01 * (1j * s / n_j)
        new_M01 = M00 * (1j * s / n_j) + M01 * c
        new_M10 = M10 * c + M11 * (1j * n_j * s)
        new_M11 = M10 * (1j * n_j * s) + M11 * c
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    # Now compute B and C:
    # [B, C]^T = M @ [1, n_s]^T
    B = M00 * 1.0 + M01 * substrate_index
    C = M10 * 1.0 + M11 * substrate_index
    
    # Reflectance:
    # r = (n_0 * B - C) / (n_0 * B + C)
    # R = |r|^2
    r = (ambient_index * B - C) / (ambient_index * B + C)
    R = np.abs(r) ** 2
    
    return R

def _compute_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    band = problem["band_nm"]
    wavelengths = np.linspace(band[0], band[1], 401)
    n_layers = len(materials)
    if n_layers == 0:
        # Bare substrate
        R = ((problem["ambient_index"] - problem["substrate_index"]) / 
             (problem["ambient_index"] + problem["substrate_index"])) ** 2
        return R * 100.0
    
    R = _reflectance_spectrum(materials, thicknesses, n_layers, 
                               problem["ambient_index"], problem["substrate_index"], wavelengths)
    return np.mean(R) * 100.0

def _nelder_mead(objective, x0, bounds, max_iter=500, tol=1e-8, deadline=None):
    """
    Nelder-Mead optimization with bounds.
    Returns (best_x, best_f).
    """
    n = len(x0)
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    # Initialize simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        perturbation = np.zeros(n)
        perturbation[i] = 0.1 * (bounds[i][1] - bounds[i][0]) if bounds[i][1] > bounds[i][0] else 0.1
        simplex.append(np.array(x0 + perturbation, dtype=float))
    
    # Evaluate initial simplex
    f_values = []
    for i in range(n + 1):
        simplex[i] = np.clip(simplex[i], bounds[:, 0], bounds[:, 1])
        f_values.append(objective(simplex[i]))
    
    best_x = simplex[0].copy()
    best_f = f_values[0]
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Sort simplex by function value
        order = np.argsort(f_values)
        simplex = [simplex[i].copy() for i in order]
        f_values = [f_values[i] for i in order]
        
        if f_values[0] < best_f:
            best_f = f_values[0]
            best_x = simplex[0].copy()
        
        # Check convergence
        if np.max(f_values) - np.min(f_values) < tol:
            break
        
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = np.clip(x_r, bounds[:, 0], bounds[:, 1])
        f_r = objective(x_r)
        
        if f_r < f_values[0]:
            # Expansion
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, bounds[:, 0], bounds[:, 1])
            f_e = objective(x_e)
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
            # Contraction
            x_c = centroid + rho * (simplex[-1] - centroid)
            x_c = np.clip(x_c, bounds[:, 0], bounds[:, 1])
            f_c = objective(x_c)
            if f_c < f_values[-1]:
                simplex[-1] = x_c
                f_values[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], bounds[:, 0], bounds[:, 1])
                    f_values[i] = objective(simplex[i])
    
    return best_x, best_f

def design(problem):
    """
    Design a broadband antireflection coating using Nelder-Mead optimization.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    materials_dict = problem["materials"]
    n_L = materials_dict["L"]
    n_H = materials_dict["H"]
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    band = problem["band_nm"]
    
    # Precompute wavelength grid for fast evaluation
    wavelengths = np.linspace(band[0], band[1], 401)
    
    def fast_reflectance(materials, thicknesses):
        """Fast reflectance computation using precomputed wavelengths."""
        n_layers = len(materials)
        if n_layers == 0:
            R = ((ambient_index - substrate_index) / 
                 (ambient_index + substrate_index)) ** 2
            return R * 100.0
        R = _reflectance_spectrum(materials, thicknesses, n_layers, 
                                   ambient_index, substrate_index, wavelengths)
        return np.mean(R) * 100.0
    
    # Try different numbers of layers
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    
    # Start with a reasonable guess: alternating L and H layers
    # Quarter-wave stack at center wavelength
    centre = 0.5 * (band[0] + band[1])
    
    for n_layers in range(1, min(max_layers + 1, 15)):
        if time.time() > deadline:
            break
        
        # Create initial guess: alternating L, H, L, H...
        # Starting with L (low index) at air side
        init_materials = []
        init_thicknesses = []
        for i in range(n_layers):
            if i % 2 == 0:
                init_materials.append(n_L)
                init_thicknesses.append(centre / (4.0 * n_L))
            else:
                init_materials.append(n_H)
                init_thicknesses.append(centre / (4.0 * n_H))
        
        # Scale thicknesses to respect total constraint
        total = sum(init_thicknesses)
        if total > max_total_nm:
            scale = max_total_nm / total
            init_thicknesses = [t * scale for t in init_thicknesses]
        
        # Ensure each thickness is within [0, max_layer_nm]
        init_thicknesses = [min(max(t, 0.0), max_layer_nm) for t in init_thicknesses]
        
        # Define objective function
        def objective(x):
            # x contains thicknesses only; materials are fixed for this trial
            th = np.clip(x, 0.0, max_layer_nm)
            # Check total thickness constraint
            if np.sum(th) > max_total_nm:
                return 100.0  # Penalty
            return fast_reflectance(init_materials, th)
        
        bounds = np.array([[0.0, max_layer_nm]] * n_layers)
        
        # Run Nelder-Mead
        try:
            x_opt, f_opt = _nelder_mead(objective, np.array(init_thicknesses), bounds, 
                                          max_iter=300, deadline=deadline)
        except Exception:
            continue
        
        if f_opt < best_score:
            best_score = f_opt
            best_materials = init_materials[:]
            best_thicknesses = list(np.clip(x_opt, 0.0, max_layer_nm))
        
        # Also try starting with H
        if n_layers > 1:
            init_materials2 = []
            init_thicknesses2 = []
            for i in range(n_layers):
                if i % 2 == 0:
                    init_materials2.append(n_H)
                    init_thicknesses2.append(centre / (4.0 * n_H))
                else:
                    init_materials2.append(n_L)
                    init_thicknesses2.append(centre / (4.0 * n_L))
            
            total2 = sum(init_thicknesses2)
            if total2 > max_total_nm:
                scale2 = max_total_nm / total2
                init_thicknesses2 = [t * scale2 for t in init_thicknesses2]
            init_thicknesses2 = [min(max(t, 0.0), max_layer_nm) for t in init_thicknesses2]
            
            def objective2(x):
                th = np.clip(x, 0.0, max_layer_nm)
                if np.sum(th) > max_total_nm:
                    return 100.0
                return fast_reflectance(init_materials2, th)
            
            try:
                x_opt2, f_opt2 = _nelder_mead(objective2, np.array(init_thicknesses2), bounds, 
                                                max_iter=300, deadline=deadline)
            except Exception:
                continue
            
            if f_opt2 < best_score:
                best_score = f_opt2
                best_materials = init_materials2[:]
                best_thicknesses = list(np.clip(x_opt2, 0.0, max_layer_nm))
    
    # Convert materials from indices back to labels
    if best_materials is None:
        # Fallback: single quarter-wave L layer
        return ["L"], [centre / (4.0 * n_L)]
    
    material_labels = []
    for n in best_materials:
        if abs(n - n_L) < 1e-6:
            material_labels.append("L")
        else:
            material_labels.append("H")
    
    # Final constraint check
    total_thickness = sum(best_thicknesses)
    if total_thickness > max_total_nm:
        scale = max_total_nm / total_thickness
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    best_thicknesses = [min(max(t, 0.0), max_layer_nm) for t in best_thicknesses]
    
    return material_labels, best_thicknesses
# EVOLVE-BLOCK-END
