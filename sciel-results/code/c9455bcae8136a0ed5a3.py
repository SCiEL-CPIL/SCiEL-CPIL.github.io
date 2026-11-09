import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _reflectance(materials, thicknesses, n_ambient, n_substrate, wavelengths):
    """Compute reflectance (%) using the characteristic-matrix method.
    materials: list of 'L'/'H'
    thicknesses: list of physical thicknesses in nm
    wavelengths: array of wavelengths in nm
    Returns array of reflectance in percent.
    """
    n_L = 1.38
    n_H = 2.35
    n_amb = n_ambient
    n_sub = n_substrate
    
    # Build the stack matrix M = M_1 * M_2 * ... * M_N
    # Each layer matrix: [[cos(delta), i*sin(delta)/n_j], [i*n_j*sin(delta), cos(delta)]]
    # where delta = 2*pi*n_j*d_j / lambda
    
    N = len(materials)
    if N == 0:
        # Bare substrate
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return 100.0 * np.abs(r)**2
    
    # Start with identity matrix
    M = np.eye(2, dtype=complex)
    
    for i in range(N):
        n_j = n_L if materials[i] == 'L' else n_H
        d_j = thicknesses[i]
        delta = 2.0 * np.pi * n_j * d_j / wavelengths  # array shape (401,)
        
        c = np.cos(delta)
        s = np.sin(delta)
        
        # Layer matrix for each wavelength
        # M_j = [[c, 1j*s/n_j], [1j*n_j*s, c]]
        M_j = np.zeros((2, 2, len(wavelengths)), dtype=complex)
        M_j[0, 0] = c
        M_j[0, 1] = 1j * s / n_j
        M_j[1, 0] = 1j * n_j * s
        M_j[1, 1] = c
        
        # M = M * M_j (matrix multiply over last two axes)
        M = np.einsum('abw,bcw->acw', M, M_j)
    
    # Reflection coefficient: r = (n_amb*M[0,0] + n_amb*M[0,1]*n_sub - M[1,0] - M[1,1]*n_sub) / 
    #                      (n_amb*M[0,0] + n_amb*M[0,1]*n_sub + M[1,0] + M[1,1]*n_sub)
    # Wait, let me use the standard formula:
    # Y = (M[0,0] + M[0,1]*n_sub) / (M[1,0] + M[1,1]*n_sub)  -- admittance
    # r = (n_amb - Y) / (n_amb + Y)
    
    num = M[0, 0] + M[0, 1] * n_sub
    den = M[1, 0] + M[1, 1] * n_sub
    Y = num / den
    r = (n_amb - Y) / (n_amb + Y)
    
    return 100.0 * np.abs(r)**2


def _mean_reflectance(materials, thicknesses, problem, wavelengths):
    """Compute mean reflectance in percent over the band."""
    R = _reflectance(materials, thicknesses, problem["ambient_index"], 
                     problem["substrate_index"], wavelengths)
    return np.mean(R)


def _valid(materials, thicknesses, problem):
    """Check if the design is valid."""
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > problem["max_layers"]:
        return False
    if any(t < 0 or t > problem["max_layer_nm"] for t in thicknesses):
        return False
    if sum(thicknesses) > problem["max_total_nm"] + 1e-6:
        return False
    return True


def design(problem):
    """Design a broadband antireflection coating."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    # 401 equally spaced wavelengths
    wavelengths = np.linspace(band[0], band[1], 401)
    
    # Baseline: bare glass
    r0 = (n_amb - n_sub) / (n_amb + n_sub)
    baseline_R = 100.0 * r0**2  # ~4.26%
    
    # Best solution so far
    best_materials = ["L"]
    best_thicknesses = [0.5 * (band[0] + band[1]) / (4.0 * n_L)]
    best_score = _mean_reflectance(best_materials, best_thicknesses, problem, wavelengths)
    
    # --- Stage 1: Simulated Annealing ---
    # We'll search over layer count (1 to max_layers), material pattern, and thicknesses.
    
    np.random.seed(42)
    
    T = 0.5  # Initial temperature (reflectance percent)
    T_min = 0.001
    cooling = 0.995
    
    # Start with a reasonable initial guess
    N_init = 5
    materials = ['L' if i % 2 == 0 else 'H' for i in range(N_init)]
    # Quarter-wave thicknesses centered in band
    center = 0.5 * (band[0] + band[1])
    thicknesses = []
    for mat in materials:
        n = n_L if mat == 'L' else n_H
        thicknesses.append(center / (4.0 * n))
    
    current_score = _mean_reflectance(materials, thicknesses, problem, wavelengths)
    
    # Track best
    if current_score < best_score:
        best_score = current_score
        best_materials = list(materials)
        best_thicknesses = list(thicknesses)
    
    # SA loop
    while time.time() < deadline - 20.0:  # Leave 20s for polish
        # Propose a move
        move_type = np.random.choice(3, p=[0.3, 0.3, 0.4])
        
        if move_type == 0:
            # Change thickness of a random layer
            if len(materials) == 0:
                continue
            idx = np.random.randint(len(materials))
            delta_t = np.random.normal(0, 30.0)
            new_t = thicknesses[idx] + delta_t
            if 0 <= new_t <= max_layer:
                old_t = thicknesses[idx]
                thicknesses[idx] = new_t
                # Check total thickness
                if sum(thicknesses) <= max_total + 1e-6:
                    new_score = _mean_reflectance(materials, thicknesses, problem, wavelengths)
                    delta = new_score - current_score
                    if delta < 0 or np.random.random() < np.exp(-delta / T):
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_materials = list(materials)
                            best_thicknesses = list(thicknesses)
                    else:
                        thicknesses[idx] = old_t
                else:
                    thicknesses[idx] = old_t
            else:
                thicknesses[idx] = old_t if 'old_t' in dir() else thicknesses[idx]
                
        elif move_type == 1:
            # Change material of a random layer
            if len(materials) == 0:
                continue
            idx = np.random.randint(len(materials))
            old_mat = materials[idx]
            new_mat = 'H' if old_mat == 'L' else 'L'
            materials[idx] = new_mat
            new_score = _mean_reflectance(materials, thicknesses, problem, wavelengths)
            delta = new_score - current_score
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_materials = list(materials)
                    best_thicknesses = list(thicknesses)
            else:
                materials[idx] = old_mat
                
        else:
            # Change number of layers (add or remove)
            if len(materials) < max_layers:
                # Add a layer
                new_mat = 'L' if np.random.random() < 0.5 else 'H'
                n_new = n_L if new_mat == 'L' else n_H
                new_t = center / (4.0 * n_new)
                # Check if adding keeps total under max
                if sum(thicknesses) + new_t <= max_total + 1e-6:
                    materials.append(new_mat)
                    thicknesses.append(new_t)
                    new_score = _mean_reflectance(materials, thicknesses, problem, wavelengths)
                    delta = new_score - current_score
                    if delta < 0 or np.random.random() < np.exp(-delta / T):
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_materials = list(materials)
                            best_thicknesses = list(thicknesses)
                    else:
                        materials.pop()
                        thicknesses.pop()
            elif len(materials) > 1:
                # Remove a layer
                idx = np.random.randint(len(materials))
                old_mat = materials[idx]
                old_t = thicknesses[idx]
                materials.pop(idx)
                thicknesses.pop(idx)
                new_score = _mean_reflectance(materials, thicknesses, problem, wavelengths)
                delta = new_score - current_score
                if delta < 0 or np.random.random() < np.exp(-delta / T):
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_materials = list(materials)
                        best_thicknesses = list(thicknesses)
                else:
                    materials.insert(idx, old_mat)
                    thicknesses.insert(idx, old_t)
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # --- Stage 2: Nelder-Mead Polish ---
    # Polish the best configuration found
    if time.time() < deadline - 5.0:
        N_best = len(best_materials)
        if N_best > 0:
            x0 = np.array(best_thicknesses, dtype=float)
            
            def objective(x):
                # Clamp to valid range
                x_clamped = np.clip(x, 0, max_layer)
                # Check total thickness
                if np.sum(x_clamped) > max_total + 1e-6:
                    # Scale down proportionally
                    x_clamped = x_clamped * (max_total / np.sum(x_clamped))
                return _mean_reflectance(best_materials, list(x_clamped), problem, wavelengths)
            
            # Use Nelder-Mead
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 0.1, 'fatol': 1e-4,
                                      'adaptive': True})
            
            if result.success or result.fun < best_score:
                x_opt = np.clip(result.x, 0, max_layer)
                if np.sum(x_opt) > max_total + 1e-6:
                    x_opt = x_opt * (max_total / np.sum(x_opt))
                new_score = objective(x_opt)
                if new_score < best_score:
                    best_score = new_score
                    best_thicknesses = list(x_opt)
    
    # Final validation
    best_thicknesses = [max(0.0, min(max_layer, t)) for t in best_thicknesses]
    if sum(best_thicknesses) > max_total:
        scale = max_total / sum(best_thicknesses)
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    # Ensure at least 1 layer
    if len(best_materials) == 0:
        best_materials = ["L"]
        best_thicknesses = [0.5 * (band[0] + band[1]) / (4.0 * n_L)]
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
