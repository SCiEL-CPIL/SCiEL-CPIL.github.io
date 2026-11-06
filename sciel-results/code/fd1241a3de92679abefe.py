"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def design(problem):
    """Design a broadband antireflection coating using multi-start Powell optimization."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Problem parameters
    ambient_index = problem.get("ambient_index", 1.0)
    substrate_index = problem.get("substrate_index", 1.52)
    materials = problem.get("materials", {"L": 1.38, "H": 2.35})
    band_nm = problem.get("band_nm", [400, 800])
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    n_L = materials["L"]
    n_H = materials["H"]
    
    # Wavelengths for evaluation (401 points)
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    num_wl = len(wavelengths)
    
    def compute_reflectance(thicknesses_nm, layer_sequence):
        """
        Compute mean reflectance using characteristic matrix method.
        thicknesses_nm: array of physical thicknesses in nm
        layer_sequence: list of 'L' or 'H' strings
        Returns mean reflectance in percent.
        """
        n_layers = len(thicknesses_nm)
        if n_layers == 0:
            # Bare substrate
            R = ((ambient_index - substrate_index) / (ambient_index + substrate_index)) ** 2
            return R * 100.0
        
        # Build the characteristic matrix
        # For each wavelength, compute the total characteristic matrix
        total_reflectance = 0.0
        
        for wl in wavelengths:
            # Start with identity matrix
            M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
            
            for i in range(n_layers):
                n_i = n_L if layer_sequence[i] == 'L' else n_H
                d_i = thicknesses_nm[i]
                
                # Optical thickness in radians
                delta_i = 2.0 * np.pi * n_i * d_i / wl
                cos_delta = np.cos(delta_i)
                sin_delta = np.sin(delta_i)
                
                # Layer characteristic matrix
                B_i = np.array([[cos_delta, 1j * sin_delta / n_i],
                                [1j * n_i * sin_delta, cos_delta]], dtype=complex)
                
                M = M @ B_i
            
            # Substrate admittance
            Y_s = substrate_index  # Real, lossless
            
            # Total admittance at the front surface
            Y_total = M[1, 1] / M[0, 1] if abs(M[0, 1]) > 1e-15 else M[1, 0] / M[0, 0]
            
            # Reflectance
            Y_0 = ambient_index
            r = (Y_0 - Y_total) / (Y_0 + Y_total)
            R = abs(r) ** 2
            
            total_reflectance += R
        
        return (total_reflectance / num_wl) * 100.0
    
    def objective(params, layer_sequence):
        """
        Objective function for optimization.
        params: array of optical thicknesses (n*d) in nm, will be converted to physical
        layer_sequence: fixed sequence of 'L'/'H'
        """
        # Convert optical thicknesses to physical thicknesses
        physical_thicknesses = []
        for i, ot in enumerate(params):
            n_i = n_L if layer_sequence[i] == 'L' else n_H
            d_i = ot / n_i
            physical_thicknesses.append(d_i)
        
        # Check constraints
        for d in physical_thicknesses:
            if d < -1e-9 or d > max_layer_nm + 1e-9:
                return 1e10  # Large penalty
        
        total = sum(physical_thicknesses)
        if total > max_total_nm + 1e-9:
            return 1e10  # Large penalty
        
        # Clip small negatives to zero
        physical_thicknesses = np.maximum(physical_thicknesses, 0.0)
        
        return compute_reflectance(physical_thicknesses, layer_sequence)
    
    # Try different numbers of layers and sequences
    best_result = None
    best_score = 1e10
    best_materials = ["L"]
    best_thicknesses = [632.8 / (4.0 * n_L)]  # Fallback: single quarter-wave L layer
    
    # Candidate layer counts
    candidate_counts = [2, 3, 4, 5, 6, 8, 10]
    
    # For each layer count, try alternating patterns starting with L or H
    for n_layers in candidate_counts:
        if time.time() > deadline - 10.0:
            break
        
        # Two patterns: starting with L, starting with H
        for start_char in ['L', 'H']:
            if time.time() > deadline - 5.0:
                break
            
            # Build the sequence
            layer_sequence = []
            current = start_char
            for i in range(n_layers):
                layer_sequence.append(current)
                current = 'H' if current == 'L' else 'L'
            
            # Initial guess: quarter-wave at center wavelength
            center_wl = 0.5 * (band_nm[0] + band_nm[1])
            initial_params = []
            for i in range(n_layers):
                n_i = n_L if layer_sequence[i] == 'L' else n_H
                ot = n_i * center_wl / 4.0
                initial_params.append(ot)
            
            # Also try a few different initial points
            initial_points = [initial_params]
            
            # Add perturbed versions
            for _ in range(3):
                perturbed = [p * (0.5 + np.random.random()) for p in initial_params]
                initial_points.append(perturbed)
            
            for x0 in initial_points:
                if time.time() > deadline - 3.0:
                    break
                
                try:
                    result = minimize(
                        lambda x: objective(x, layer_sequence),
                        x0,
                        method='Powell',
                        options={
                            'maxiter': 200,
                            'maxfev': 500,
                            'xatol': 1e-6,
                            'fatol': 1e-8
                        }
                    )
                    
                    if result.fun < best_score:
                        # Convert back to physical thicknesses
                        physical_thicknesses = []
                        for i, ot in enumerate(result.x):
                            n_i = n_L if layer_sequence[i] == 'L' else n_H
                            d_i = max(0.0, ot / n_i)
                            physical_thicknesses.append(d_i)
                        
                        # Check constraints
                        if all(0 <= d <= max_layer_nm for d in physical_thicknesses) and \
                           sum(physical_thicknesses) <= max_total_nm:
                            best_score = result.fun
                            best_materials = layer_sequence.copy()
                            best_thicknesses = physical_thicknesses
                            
                except Exception:
                    pass
    
    # If no valid result found, return the fallback
    if best_result is None or best_score >= 1e9:
        center_wl = 0.5 * (band_nm[0] + band_nm[1])
        return ["L"], [center_wl / (4.0 * n_L)]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
