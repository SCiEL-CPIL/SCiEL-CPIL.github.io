"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _char_matrix_method(wavelengths, thicknesses, indices, ambient_n, substrate_n):
    """Compute reflectance using characteristic matrix method.
    
    Parameters:
    - wavelengths: array of wavelengths in nm
    - thicknesses: array of physical thicknesses in nm
    - indices: array of refractive indices for each layer
    - ambient_n: ambient refractive index
    - substrate_n: substrate refractive index
    
    Returns:
    - reflectance: array of reflectance values (0 to 1)
    """
    n_layers = len(thicknesses)
    reflectance = np.zeros(len(wavelengths))
    
    for i, wl in enumerate(wavelengths):
        # Build the characteristic matrix for each layer
        # M_j = [[cos(delta_j), (i/sin(delta_j))/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
        # where delta_j = 2*pi*n_j*d_j/(lambda)
        
        # Start with identity matrix
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        
        for j in range(n_layers):
            n_j = indices[j]
            d_j = thicknesses[j]
            delta_j = 2.0 * np.pi * n_j * d_j / wl
            
            cos_d = np.cos(delta_j)
            sin_d = np.sin(delta_j)
            
            # Handle sin_d near zero to avoid division by zero
            if abs(sin_d) < 1e-15:
                sin_d = 1e-15 if sin_d >= 0 else -1e-15
            
            M_j = np.array([
                [cos_d, 1j * sin_d / n_j],
                [1j * n_j * sin_d, cos_d]
            ], dtype=complex)
            
            M = M @ M_j
        
        # The admittance Y = M[1,0] / M[0,0]
        # Reflectance R = |(Y - n_ambient) / (Y + n_ambient)|^2
        Y = M[1, 0] / M[0, 0]
        
        # For normal incidence, we need to account for the substrate
        # Actually, let me reconsider: the characteristic matrix gives the field
        # at the substrate interface. We need to compute the reflection coefficient.
        
        # Let me use the standard approach:
        # The total characteristic matrix M = M_1 * M_2 * ... * M_N
        # B = M[0,0] + M[0,1] * n_s
        # C = M[1,0] * n_0 + M[1,1] * n_s
        # R = |(C - B) / (C + B)|^2
        
        n_s = substrate_n
        n_0 = ambient_n
        
        B = M[0, 0] + M[0, 1] * n_s
        C = M[1, 0] * n_0 + M[1, 1] * n_s
        
        if abs(C + B) < 1e-15:
            reflectance[i] = 1.0
        else:
            r = (C - B) / (C + B)
            reflectance[i] = np.abs(r) ** 2
    
    return reflectance

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    wavelengths = np.linspace(problem["band_nm"][0], problem["band_nm"][1], 401)
    indices = [problem["materials"][m] for m in materials]
    refl = _char_matrix_method(wavelengths, np.array(thicknesses), np.array(indices),
                               problem["ambient_index"], problem["substrate_index"])
    return 100.0 * np.mean(refl)

def _is_valid(materials, thicknesses, problem):
    """Check if the design is valid."""
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > problem["max_layers"]:
        return False
    if any(t < 0 or t > problem["max_layer_nm"] for t in thicknesses):
        return False
    if sum(thicknesses) > problem["max_total_nm"]:
        return False
    if any(m not in problem["materials"] for m in materials):
        return False
    return True

def _coordinate_descent(materials, thicknesses, problem, time_limit):
    """Optimize thicknesses using coordinate descent."""
    start_time = time.time()
    current_score = _mean_reflectance(materials, thicknesses, problem)
    best_materials = list(materials)
    best_thicknesses = list(thicknesses)
    best_score = current_score
    
    n_layers = len(thicknesses)
    if n_layers == 0:
        return best_materials, best_thicknesses, best_score
    
    # Initial guess: quarter-wave at center wavelength
    center_wl = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    
    # Try different step sizes
    step_sizes = [50.0, 25.0, 10.0, 5.0, 2.0, 1.0, 0.5]
    
    for step in step_sizes:
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 5:
                break
            improved = False
            for i in range(n_layers):
                if time.time() - start_time > time_limit - 5:
                    break
                # Try increasing and decreasing
                for delta in [step, -step]:
                    new_thickness = thicknesses[i] + delta
                    if new_thickness < 0 or new_thickness > problem["max_layer_nm"]:
                        continue
                    new_thicknesses = list(thicknesses)
                    new_thicknesses[i] = new_thickness
                    if sum(new_thicknesses) > problem["max_total_nm"]:
                        continue
                    
                    new_score = _mean_reflectance(materials, new_thicknesses, problem)
                    if new_score < current_score - 1e-10:
                        current_score = new_score
                        thicknesses = new_thicknesses
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_materials = list(materials)
                            best_thicknesses = list(thicknesses)
    
    return best_materials, best_thicknesses, best_score

def design(problem):
    """Design antireflection coating using greedy layer addition and coordinate descent."""
    start_time = time.time()
    time_limit = 140.0  # Leave 10s safety margin
    
    L_index = problem["materials"]["L"]
    H_index = problem["materials"]["H"]
    center_wl = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    
    # Start with a good initial guess: alternating L-H quarter-wave stack
    # For broadband AR, start with a few layers
    
    best_materials = ["L"]
    best_thicknesses = [center_wl / (4.0 * L_index)]
    best_score = _mean_reflectance(best_materials, best_thicknesses, problem)
    
    # Try adding layers greedily
    current_materials = list(best_materials)
    current_thicknesses = list(best_thicknesses)
    current_score = best_score
    
    for _ in range(problem["max_layers"] - 1):
        if time.time() - start_time > time_limit - 20:
            break
        
        best_new_score = current_score
        best_new_materials = list(current_materials)
        best_new_thicknesses = list(current_thicknesses)
        
        # Try adding L or H at the end
        for new_mat in ["L", "H"]:
            new_materials = current_materials + [new_mat]
            n_idx = problem["materials"][new_mat]
            
            # Try several thicknesses
            for frac in [0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]:
                new_thickness = center_wl * frac / n_idx
                if new_thickness > problem["max_layer_nm"]:
                    continue
                if sum(current_thicknesses) + new_thickness > problem["max_total_nm"]:
                    continue
                
                new_thicknesses = current_thicknesses + [new_thickness]
                new_score = _mean_reflectance(new_materials, new_thicknesses, problem)
                
                if new_score < best_new_score - 1e-10:
                    best_new_score = new_score
                    best_new_materials = new_materials
                    best_new_thicknesses = new_thicknesses
        
        if best_new_score < current_score - 1e-10:
            current_materials = best_new_materials
            current_thicknesses = best_new_thicknesses
            current_score = best_new_score
            if current_score < best_score:
                best_score = current_score
                best_materials = list(current_materials)
                best_thicknesses = list(current_thicknesses)
        else:
            # Try inserting in the middle as well
            best_insert_score = current_score
            best_insert_materials = list(current_materials)
            best_insert_thicknesses = list(current_thicknesses)
            
            for insert_pos in range(1, len(current_materials)):
                for new_mat in ["L", "H"]:
                    new_materials = current_materials[:insert_pos] + [new_mat] + current_materials[insert_pos:]
                    n_idx = problem["materials"][new_mat]
                    
                    for frac in [0.25, 0.35, 0.5, 0.65, 0.75]:
                        new_thickness = center_wl * frac / n_idx
                        if new_thickness > problem["max_layer_nm"]:
                            continue
                        if sum(current_thicknesses) + new_thickness > problem["max_total_nm"]:
                            continue
                        
                        new_thicknesses = current_thicknesses[:insert_pos] + [new_thickness] + current_thicknesses[insert_pos:]
                        new_score = _mean_reflectance(new_materials, new_thicknesses, problem)
                        
                        if new_score < best_insert_score - 1e-10:
                            best_insert_score = new_score
                            best_insert_materials = new_materials
                            best_insert_thicknesses = new_thicknesses
            
            if best_insert_score < current_score - 1e-10:
                current_materials = best_insert_materials
                current_thicknesses = best_insert_thicknesses
                current_score = best_insert_score
                if current_score < best_score:
                    best_score = current_score
                    best_materials = list(current_materials)
                    best_thicknesses = list(current_thicknesses)
            else:
                break
    
    # Final coordinate descent optimization
    if time.time() - start_time < time_limit - 10:
        best_materials, best_thicknesses, best_score = _coordinate_descent(
            best_materials, best_thicknesses, problem, time_limit - 10
        )
    
    # Ensure validity
    if not _is_valid(best_materials, best_thicknesses, problem):
        best_materials = ["L"]
        best_thicknesses = [center_wl / (4.0 * L_index)]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
