import numpy as np
import time
import random

def design(problem):
    """
    Design a broadband antireflection coating using multi-start local search
    with coordinate descent and random restarts.
    """
    n_0 = problem["ambient_index"]
    n_s = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths for evaluation (401 points)
    wavelengths = np.linspace(band[0], band[1], 401)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def compute_reflectance(materials, thicknesses):
        """
        Compute reflectance spectrum using characteristic matrix method.
        Returns array of reflectance values (0 to 1) at each wavelength.
        """
        n_layers = len(materials)
        if n_layers == 0:
            # Bare substrate
            r = (n_0 - n_s) / (n_0 + n_s)
            return np.full_like(wavelengths, r * r)
        
        # Precompute refractive indices for each layer
        n_layers_arr = np.array([n_L if m == 'L' else n_H for m in materials])
        d_arr = np.array(thicknesses, dtype=float)
        
        # Optical thicknesses: n*d
        # Phase: delta = 2*pi*n*d / lambda
        # We'll compute the characteristic matrix for each wavelength
        
        # Vectorized computation
        # M_i = [[cos(delta_i), i*sin(delta_i)/n_i], [i*n_i*sin(delta_i), cos(delta_i)]]
        # For normal incidence, s and p polarizations are the same
        
        # We'll compute the total characteristic matrix M = M_1 * M_2 * ... * M_N
        # Then Y = M_11 / M_21 (admittance)
        # R = |Y - n_0| / (Y + n_0) squared... wait, let me be more careful.
        
        # Actually, the characteristic matrix approach:
        # For each layer i:
        #   M_i = [[cos(delta_i), (i/n_i)sin(delta_i)], [i*n_i*sin(delta_i), cos(delta_i)]]
        # Total M = M_1 * M_2 * ... * M_N
        # B = M_11 * n_s + i * M_21 * n_s^2 ... no wait
        # Actually: B = M_11 * n_s + i * M_21 * n_s^2 is not right either.
        
        # Let me use the standard formulation:
        # The characteristic matrix for layer i is:
        # M_i = [[cos(delta_i), i*sin(delta_i)/n_i], [i*n_i*sin(delta_i), cos(delta_i)]]
        # Total matrix M = M_1 * M_2 * ... * M_N
        # Then:
        # B = M_11 * n_s + i * M_21 * n_s^2  -- this is for admittance
        # C = M_12 * n_s + i * M_22 * n_s^2
        # Y = B/C
        # R = |(Y - n_0)/(Y + n_0)|^2
        
        # Wait, I need to be more careful. Let me use a cleaner formulation.
        
        # Actually, for normal incidence:
        # Each layer has characteristic matrix:
        # M_i = [[cos(delta_i), i*sin(delta_i)/n_i], [i*n_i*sin(delta_i), cos(delta_i)]]
        # where delta_i = 2*pi*n_i*d_i/lambda
        
        # The total characteristic matrix is M = M_1 * M_2 * ... * M_N
        # Then the admittance at the input side is:
        # Y = (M_11 * n_s + i * M_21 * n_s^2) / (M_12 * n_s + i * M_22 * n_s^2)
        
        # Hmm, actually I think the correct formula is:
        # B = M_11 * n_s + i * M_21 * n_s^2
        # C = M_12 * n_s + i * M_22 * n_s^2
        # Y = B / C
        # R = |Y - n_0|^2 / |Y + n_0|^2
        
        # Let me verify with a single layer:
        # M = [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
        # B = cos(delta)*n_s + i*(i*n*sin(delta))*n_s^2 = cos(delta)*n_s - n*sin(delta)*n_s^2
        # C = (i*sin(delta)/n)*n_s + i*cos(delta)*n_s^2 = i*sin(delta)*n_s/n + i*cos(delta)*n_s^2
        # Y = B/C
        # For quarter wave: delta = pi/2, cos=0, sin=1
        # B = -n*n_s^2
        # C = i*n_s/n + i*n_s^2 = i*n_s(1/n + n_s)
        # Y = -n*n_s^2 / (i*n_s*(1/n + n_s)) = -n*n_s / (i*(1/n + n_s)) = i*n*n_s / (1/n + n_s)
        # Y = i*n*n_s / ((n + n_s*n)/n) = i*n^2*n_s / (n + n_s*n) ... this is getting messy
        
        # Let me just use a simpler approach: compute it numerically for each wavelength.
        
        reflectances = np.zeros(len(wavelengths))
        
        for idx, lam in enumerate(wavelengths):
            # Compute phase for each layer
            delta = 2.0 * np.pi * n_layers_arr * d_arr / lam
            
            # Build characteristic matrix
            # Start with identity
            M11 = 1.0 + 0j
            M12 = 0.0 + 0j
            M21 = 0.0 + 0j
            M22 = 1.0 + 0j
            
            for i in range(n_layers):
                cos_d = np.cos(delta[i])
                sin_d = np.sin(delta[i])
                n_i = n_layers_arr[i]
                
                # Layer characteristic matrix
                m11 = cos_d
                m12 = 1j * sin_d / n_i
                m21 = 1j * n_i * sin_d
                m22 = cos_d
                
                # Multiply M = M * m
                new_M11 = M11 * m11 + M12 * m21
                new_M12 = M11 * m12 + M12 * m22
                new_M21 = M21 * m11 + M22 * m21
                new_M22 = M21 * m12 + M22 * m22
                
                M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
            
            # Compute admittance
            B = M11 * n_s + 1j * M21 * n_s ** 2
            C = M12 * n_s + 1j * M22 * n_s ** 2
            Y = B / C
            
            # Reflectance
            r = (Y - n_0) / (Y + n_0)
            reflectances[idx] = np.abs(r) ** 2
        
        return reflectances
    
    def objective(materials, thicknesses):
        """Mean reflectance in percent."""
        if len(materials) == 0:
            r_bare = (n_0 - n_s) / (n_0 + n_s)
            return 100.0 * r_bare ** 2
        refl = compute_reflectance(materials, thicknesses)
        return 100.0 * np.mean(refl)
    
    def is_valid(materials, thicknesses):
        """Check constraints."""
        if len(materials) > max_layers:
            return False
        if len(materials) != len(thicknesses):
            return False
        for t in thicknesses:
            if t < -1e-9 or t > max_layer_nm + 1e-9:
                return False
        if sum(thicknesses) > max_total_nm + 1e-9:
            return False
        for m in materials:
            if m not in ('L', 'H'):
                return False
        return True
    
    def clamp_thicknesses(thicknesses):
        """Ensure thicknesses are within bounds."""
        return [max(0.0, min(max_layer_nm, t)) for t in thicknesses]
    
    # Initial best solution
    best_materials = ["L"]
    best_thicknesses = [600.0 / (4.0 * n_L)]  # Quarter wave at 600nm
    best_score = objective(best_materials, best_thicknesses)
    
    # Try some initial configurations
    initial_configs = [
        # Single quarter-wave layer
        (["L"], [600.0 / (4.0 * n_L)]),
        (["H"], [600.0 / (4.0 * n_H)]),
        # Two-layer quarter-wave stack
        (["L", "H"], [600.0 / (4.0 * n_L), 600.0 / (4.0 * n_H)]),
        (["H", "L"], [600.0 / (4.0 * n_H), 600.0 / (4.0 * n_L)]),
        # Three-layer
        (["L", "H", "L"], [600.0 / (4.0 * n_L), 600.0 / (4.0 * n_H), 600.0 / (4.0 * n_L)]),
        (["H", "L", "H"], [600.0 / (4.0 * n_H), 600.0 / (4.0 * n_L), 600.0 / (4.0 * n_H)]),
    ]
    
    for mats, thicks in initial_configs:
        if is_valid(mats, thicks):
            score = objective(mats, thicks)
            if score < best_score:
                best_score = score
                best_materials = list(mats)
                best_thicknesses = list(thicks)
    
    # Coordinate descent with multiple step sizes
    def coordinate_descent(materials, thicknesses, max_iterations=200, step_sizes=None):
        """
        Perform coordinate descent on the thicknesses.
        """
        if step_sizes is None:
            step_sizes = [50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.1]
        
        materials = list(materials)
        thicknesses = [float(t) for t in thicknesses]
        n_layers = len(materials)
        
        if n_layers == 0:
            return materials, thicknesses, objective(materials, thicknesses)
        
        current_score = objective(materials, thicknesses)
        
        for step in step_sizes:
            improved = True
            while improved and max_iterations > 0:
                improved = False
                for i in range(n_layers):
                    # Try adjusting thickness[i] by +step and -step
                    for delta in [step, -step]:
                        new_thicks = list(thicknesses)
                        new_thicks[i] = max(0.0, min(max_layer_nm, thicknesses[i] + delta))
                        
                        if not is_valid(materials, new_thicks):
                            continue
                        
                        new_score = objective(materials, new_thicks)
                        if new_score < current_score - 1e-12:
                            thicknesses = new_thicks
                            current_score = new_score
                            improved = True
                            max_iterations -= 1
                            if max_iterations <= 0:
                                break
                    if max_iterations <= 0:
                        break
                if max_iterations <= 0:
                    break
        
        return materials, thicknesses, current_score
    
    # Multi-start local search
    random.seed(42)
    
    # Define layer count ranges to explore
    layer_counts = list(range(1, min(max_layers + 1, 15)))
    
    # Shuffle for diversity
    random.shuffle(layer_counts)
    
    for n_layers in layer_counts:
        if time.time() > deadline:
            break
        
        # Generate several random initial configurations for this layer count
        n_starts = max(1, min(5, int((deadline - time.time()) / 2.0)))
        
        for start_idx in range(n_starts):
            if time.time() > deadline:
                break
            
            # Random initial configuration
            materials = []
            thicknesses = []
            
            # Ensure total thickness doesn't exceed max_total_nm
            avg_thickness = max_total_nm / n_layers
            max_per_layer = min(max_layer_nm, avg_thickness * 1.5)
            
            for i in range(n_layers):
                materials.append(random.choice(['L', 'H']))
                # Start with random thickness, biased towards quarter-wave values
                if random.random() < 0.5:
                    n_mat = n_L if materials[-1] == 'L' else n_H
                    t = 600.0 / (4.0 * n_mat) * random.uniform(0.5, 1.5)
                else:
                    t = random.uniform(0.0, max_per_layer)
                thicknesses.append(min(max_layer_nm, t))
            
            # Check validity
            if not is_valid(materials, thicknesses):
                # Try to fix by scaling down
                total = sum(thicknesses)
                if total > max_total_nm:
                    scale = max_total_nm / total
                    thicknesses = [t * scale for t in thicknesses]
                if not is_valid(materials, thicknesses):
                    continue
            
            # Run coordinate descent
            mats, thicks, score = coordinate_descent(materials, thicknesses)
            
            if score < best_score:
                best_score = score
                best_materials = mats
                best_thicknesses = thicks
    
    # Final refinement with smaller steps
    if time.time() < deadline - 5.0:
        mats, thicks, score = coordinate_descent(
            best_materials, best_thicknesses,
            max_iterations=100,
            step_sizes=[1.0, 0.5, 0.1, 0.05, 0.01]
        )
        if score < best_score:
            best_score = score
            best_materials = mats
            best_thicknesses = thicks
    
    # Ensure final result is valid
    if not is_valid(best_materials, best_thicknesses):
        # Fallback to simple solution
        best_materials = ["L"]
        best_thicknesses = [600.0 / (4.0 * n_L)]
    
    return best_materials, best_thicknesses
