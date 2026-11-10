import numpy as np
import time
import random

def design(problem):
    """
    Design a broadband antireflection coating using local search.
    """
    n_0 = problem["ambient_index"]
    n_s = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths for evaluation
    wavelengths = np.linspace(band[0], band[1], 401)
    
    def compute_reflectance(materials, thicknesses):
        """
        Compute mean reflectance for a given stack.
        """
        if not materials:
            # Bare substrate
            r = (n_0 - n_s) / (n_0 + n_s)
            return (abs(r)**2) * 100.0
        
        total_ref = 0.0
        N = len(materials)
        
        for lam in wavelengths:
            # Calculate characteristic matrix
            # M = prod of individual layer matrices
            # Layer matrix: [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
            # Note: n here is the refractive index of the layer.
            
            # Initialize identity matrix
            M = np.array([[1.0+0j, 0.0+0j], [0.0+0j, 1.0+0j]])
            
            for i in range(N):
                n_layer = n_L if materials[i] == 'L' else n_H
                d = thicknesses[i]
                delta = 2.0 * np.pi * n_layer * d / lam
                c = np.cos(delta)
                s = np.sin(delta)
                
                # Layer matrix
                M_i = np.array([[c, 1j*s/n_layer], [1j*n_layer*s, c]])
                
                # Multiply M = M * M_i
                M = M @ M_i
            
            # Calculate Admittance Y
            # Y = (M11*n_s + i*M12*n_0*n_s) / (M21*n_s + i*M22*n_0) ? 
            # Let's re-verify the formula.
            # E = M11 * Es + M12 * Hs
            # H = M21 * Es + M22 * Hs
            # At substrate: Hs = n_s * Es (assuming non-magnetic, H = n*E)
            # So:
            # E = (M11 + M12*n_s) * Es
            # H = (M21 + M22*n_s) * Es
            # Y = E/H = (M11 + M12*n_s) / (M21 + M22*n_s)
            
            Y = (M[0,0] + M[0,1]*n_s) / (M[1,0] + M[1,1]*n_s)
            
            r = (Y - n_0) / (Y + n_0)
            R = (abs(r)**2) * 100.0
            total_ref += R
            
        return total_ref / len(wavelengths)

    def is_valid(materials, thicknesses):
        if len(materials) > max_layers:
            return False
        if len(materials) != len(thicknesses):
            return False
        for t in thicknesses:
            if t < 0 or t > max_layer_nm:
                return False
        if sum(thicknesses) > max_total_nm:
            return False
        return True

    # Initial candidates
    # 1. Single L layer at 1/4 wave for center wavelength
    center_lam = 0.5 * (band[0] + band[1])
    t_L_1 = center_lam / (4.0 * n_L)
    candidates = [
        (['L'], [t_L_1]),
        (['L', 'H', 'L'], [t_L_1/2, t_L_1/2, t_L_1/2]), # Rough guess
        (['L', 'H', 'H', 'L'], [t_L_1/3, t_L_1/3, t_L_1/3, t_L_1/3]),
        (['L', 'L'], [t_L_1/2, t_L_1/2]) # Equivalent to single L, but allows perturbation
    ]
    
    best_score = float('inf')
    best_materials = []
    best_thicknesses = []
    
    for mats, ths in candidates:
        if not is_valid(mats, ths):
            continue
        score = compute_reflectance(mats, ths)
        if score < best_score:
            best_score = score
            best_materials = mats[:]
            best_thicknesses = ths[:]
            
    # Local Search
    start_time = time.time()
    time_limit = 50.0 # Leave some buffer
    
    current_materials = best_materials[:]
    current_thicknesses = best_thicknesses[:]
    current_score = best_score
    
    # Random seed for reproducibility if needed, but random is fine
    random.seed(42)
    
    while time.time() - start_time < time_limit:
        # Pick a random move
        move_type = random.choice(['thick', 'mat', 'add', 'rem'])
        
        new_materials = current_materials[:]
        new_thicknesses = current_thicknesses[:]
        
        valid_move = False
        
        if move_type == 'thick' and len(current_materials) > 0:
            idx = random.randint(0, len(current_materials)-1)
            # Change thickness by a random amount
            delta_t = random.uniform(-20, 20)
            new_thicknesses[idx] += delta_t
            # Clamp to [0, max_layer_nm]
            if new_thicknesses[idx] < 0:
                new_thicknesses[idx] = 0
            elif new_thicknesses[idx] > max_layer_nm:
                new_thicknesses[idx] = max_layer_nm
            valid_move = True
            
        elif move_type == 'mat' and len(current_materials) > 0:
            idx = random.randint(0, len(current_materials)-1)
            # Flip material
            if new_materials[idx] == 'L':
                new_materials[idx] = 'H'
            else:
                new_materials[idx] = 'L'
            valid_move = True
            
        elif move_type == 'add' and len(current_materials) < max_layers:
            # Add a layer at a random position
            pos = random.randint(0, len(new_materials))
            new_mat = random.choice(['L', 'H'])
            new_th = random.uniform(10, 100) # Start with small thickness
            new_materials.insert(pos, new_mat)
            new_thicknesses.insert(pos, new_th)
            valid_move = True
            
        elif move_type == 'rem' and len(current_materials) > 1:
            # Remove a random layer
            idx = random.randint(0, len(current_materials)-1)
            new_materials.pop(idx)
            new_thicknesses.pop(idx)
            valid_move = True
            
        if not valid_move:
            continue
            
        if is_valid(new_materials, new_thicknesses):
            new_score = compute_reflectance(new_materials, new_thicknesses)
            if new_score < current_score:
                current_score = new_score
                current_materials = new_materials
                current_thicknesses = new_thicknesses
                if current_score < best_score:
                    best_score = current_score
                    best_materials = current_materials[:]
                    best_thicknesses = current_thicknesses[:]
                    
    return best_materials, best_thicknesses
