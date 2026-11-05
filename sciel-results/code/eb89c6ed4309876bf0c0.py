"""Broadband antireflection coating design using characteristic-matrix method."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

# EVOLVE-BLOCK-START

def _char_matrix(n, d, wl):
    """
    Characteristic matrix for a single lossless layer at normal incidence.
    n: refractive index (real)
    d: physical thickness (nm)
    wl: wavelength (nm)
    Returns 2x2 complex numpy array.
    """
    beta = 2.0 * np.pi * n * d / wl
    cos_b = np.cos(beta)
    sin_b = np.sin(beta)
    # M = [[cos(beta), -i*sin(beta)/n], [i*n*sin(beta), cos(beta)]]
    M = np.array([[cos_b, -1j * sin_b / n],
                  [1j * n * sin_b, cos_b]], dtype=complex)
    return M


def _compute_reflectance(materials, thicknesses, wl, n_air, n_sub):
    """
    Compute reflectance (power ratio) for a given stack at a single wavelength.
    materials: list of 'L'/'H'
    thicknesses: list of physical thicknesses in nm
    wl: wavelength in nm
    n_air: ambient index
    n_sub: substrate index
    Returns reflectance R (0 to 1).
    """
    if len(materials) == 0:
        r = (n_air - n_sub) / (n_air + n_sub)
        return abs(r) ** 2

    # Get the matrix for each layer
    M_total = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    
    for i, mat in enumerate(materials):
        if mat == 'L':
            n_layer = 1.38
        elif mat == 'H':
            n_layer = 2.35
        else:
            raise ValueError(f"Unknown material {mat}")
        
        d = thicknesses[i]
        if d <= 0:
            continue  # Skip zero-thickness layers
        
        M_layer = _char_matrix(n_layer, d, wl)
        M_total = M_total @ M_layer
    
    # Effective admittance Y
    # [B, C]^T = M_total @ [1, 0]^T (since E_s = 1, H_s = 0 at substrate)
    B = M_total[0, 0]
    C = M_total[1, 0]
    
    # Y = n_sub / (B + C * n_sub) ... wait, let's derive carefully.
    # The characteristic matrix relates (E, H) at air side to (E_s, H_s) at substrate.
    # [E]   = [B C] [E_s]
    # [H]     [D E] [H_s]
    # At substrate, the wave is incident from the coating side onto the substrate.
    # Actually, the standard convention is:
    # The total field at the air side is E = B*E_s + C*H_s, H = D*E_s + E*H_s.
    # But we normalize the substrate wave to have amplitude 1.
    # So E_s = 1, H_s = n_sub (since H = n E for forward wave in medium n).
    # Wait, H = n E / Z0? No, H = n E for normalized units where Z0=1.
    # Let's use the admittance method.
    # Y = (B + C * n_sub) / (D + E * n_sub)
    # Actually, the standard formula is:
    # Y = n_sub * (M22 + M21/n_sub) / (M12 + M11/n_sub) ? No.
    
    # Let's stick to the field matching.
    # At the substrate interface, the total field is the sum of the wave transmitted into the substrate
    # and the wave reflected from the substrate (which is 0 if it's semi-infinite and we only consider forward).
    # Wait, the characteristic matrix method usually defines the matrix M such that:
    # [E]   = [M11 M12] [1]
    # [H]     [M21 M22] [n_sub]
    # This assumes the substrate has index n_sub and we normalize the forward wave amplitude to 1.
    # So H_sub = n_sub * E_sub = n_sub.
    # Then E_air = M11 * 1 + M12 * n_sub
    # H_air = M21 * 1 + M22 * n_sub
    
    E_air = M_total[0, 0] + M_total[0, 1] * n_sub
    H_air = M_total[1, 0] + M_total[1, 1] * n_sub
    
    # Admittance Y = H_air / E_air
    Y = H_air / E_air
    
    # Reflection coefficient r = (n_air - Y) / (n_air + Y)
    r = (n_air - Y) / (n_air + Y)
    R = abs(r) ** 2
    
    # Clip to [0, 1] to avoid numerical issues
    R = max(0.0, min(1.0, float(np.real(R))))
    return R


def _mean_reflectance_percent(materials, thicknesses, problem):
    """
    Compute the mean reflectance in percent over the band.
    """
    band = problem["band_nm"]
    n_air = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    
    # 401 equally spaced wavelengths
    wavelengths = np.linspace(band[0], band[1], 401)
    
    total_R = 0.0
    for wl in wavelengths:
        R = _compute_reflectance(materials, thicknesses, wl, n_air, n_sub)
        total_R += R
    
    mean_R = total_R / 401.0
    return mean_R * 100.0


def _is_valid(materials, thicknesses, problem):
    """
    Check if the design is valid.
    """
    if len(materials) > problem["max_layers"]:
        return False
    if len(materials) != len(thicknesses):
        return False
    
    total_thickness = 0.0
    for i, t in enumerate(thicknesses):
        if t < 0 or t > problem["max_layer_nm"]:
            return False
        total_thickness += t
        
    if total_thickness > problem["max_total_nm"] + 1e-9:
        return False
        
    for mat in materials:
        if mat not in ['L', 'H']:
            return False
            
    return True


def _encode_stack(materials, thicknesses, max_layers):
    """
    Encode the stack into a fixed-length vector for optimization.
    We use a vector of length max_layers * 2.
    First max_layers elements are indices (0 for L, 1 for H).
    Next max_layers elements are thicknesses.
    """
    vec = np.zeros(max_layers * 2)
    for i in range(len(materials)):
        vec[i] = 1.0 if materials[i] == 'H' else 0.0
        vec[max_layers + i] = thicknesses[i]
    return vec


def _decode_stack(vec, max_layers):
    """
    Decode the vector back into materials and thicknesses.
    We only keep layers where thickness > small epsilon.
    """
    materials = []
    thicknesses = []
    for i in range(max_layers):
        idx = vec[i]
        th = vec[max_layers + i]
        if th > 1e-6:  # Threshold for "active" layer
            mat = 'H' if idx > 0.5 else 'L'
            materials.append(mat)
            thicknesses.append(th)
    return materials, thicknesses


def _objective_vec(vec, problem):
    """
    Objective function for the optimizer.
    vec: vector of length max_layers * 2
    """
    max_layers = problem["max_layers"]
    materials, thicknesses = _decode_stack(vec, max_layers)
    
    if not _is_valid(materials, thicknesses, problem):
        return 100.0  # Large penalty
    
    # Add a small penalty for using too many layers to encourage sparsity
    # (optional, but can help)
    penalty = 0.01 * len(materials)
    
    return _mean_reflectance_percent(materials, thicknesses, problem) + penalty


def design(problem):
    """
    Design a stack of at most twenty low- and high-index layers that minimizes
    the mean reflectance between 400 and 800 nm at normal incidence.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n_air = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    materials_dict = problem["materials"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    band = problem["band_nm"]
    
    # Pre-compute some constants
    # We will use differential evolution for the global search.
    
    # Initial best: the seed program (single L layer)
    centre = 0.5 * (band[0] + band[1])
    best_materials = ["L"]
    best_thicknesses = [centre / (4.0 * materials_dict["L"])]
    best_score = _mean_reflectance_percent(best_materials, best_thicknesses, problem)
    
    # Define bounds for the vector
    # First max_layers: [0, 1] for material index
    # Next max_layers: [0, max_layer_nm] for thickness
    bounds = [(0.0, 1.0)] * max_layers + [(0.0, max_layer_nm)] * max_layers
    
    # Run differential evolution
    # We need to be careful with the number of evaluations.
    # DE can be slow. Let's try to limit the population size and max iterations.
    
    # To speed up, we can reduce the number of wavelengths evaluated in the objective function
    # during the search, and then refine with the full set.
    
    def _objective_fast(vec):
        """
        Faster objective using fewer wavelengths.
        """
        max_layers = problem["max_layers"]
        materials, thicknesses = _decode_stack(vec, max_layers)
        
        if not _is_valid(materials, thicknesses, problem):
            return 100.0
        
        # Use 101 wavelengths for speed
        wavelengths = np.linspace(band[0], band[1], 101)
        total_R = 0.0
        for wl in wavelengths:
            R = _compute_reflectance(materials, thicknesses, wl, n_air, n_sub)
            total_R += R
        
        mean_R = total_R / 101.0
        penalty = 0.01 * len(materials)
        return mean_R * 100.0 + penalty
    
    # Try a few random restarts with differential evolution
    # Since DE is stochastic, we can run it for a few seconds.
    
    # We'll use scipy's differential_evolution
    # It has a 'maxiter' and 'popsize' parameter.
    # We can also set 'tol' to stop early.
    
    # Let's try to run DE for a limited time.
    # We'll wrap it in a loop and check the time.
    
    # Actually, DE in scipy doesn't have a time limit directly.
    # We can use a callback to check the time.
    
    def _callback(xk, convergence=True):
        nonlocal deadline
        if time.time() > deadline - 5.0:
            return True  # Stop
        return False
    
    # We need to be careful with the number of layers.
    # Using all 20 layers might be too complex for DE.
    # Let's try with a smaller number of layers first, e.g., 10.
    
    # We'll try a few different max_layers values.
    for ml in [5, 10, 15, 20]:
        if time.time() > deadline - 30.0:
            break
            
        bounds_ml = [(0.0, 1.0)] * ml + [(0.0, max_layer_nm)] * ml
        
        # Initial guess: random
        # DE doesn't need an initial guess, it generates a population.
        
        try:
            result = differential_evolution(
                _objective_fast,
                bounds_ml,
                maxiter=100,  # Limit iterations
                popsize=15,  # Population size
                tol=1e-4,
                mutation=(0.3, 1.2),
                recombination=0.9,
                seed=42,
                callback=_callback,
                polish=False  # Don't polish, it's slow
            )
            
            if result.success or result.fun < best_score:
                vec = result.x
                materials, thicknesses = _decode_stack(vec, ml)
                if _is_valid(materials, thicknesses, problem):
                    score = _mean_reflectance_percent(materials, thicknesses, problem)
                    if score < best_score:
                        best_score = score
                        best_materials = materials
                        best_thicknesses = thicknesses
                        
        except Exception as e:
            pass  # Ignore errors and continue
        
        if time.time() > deadline - 10.0:
            break
    
    # If we have time, refine the best solution with a local optimizer
    if time.time() < deadline - 5.0 and len(best_materials) > 0:
        # We can try to optimize the thicknesses of the best solution
        # using a local optimizer like L-BFGS-B or Nelder-Mead.
        
        # We'll only optimize the thicknesses, keeping the materials fixed.
        # This is faster and more likely to converge.
        
        def _objective_refine(th_vec):
            materials = best_materials
            thicknesses = th_vec
            if not _is_valid(materials, thicknesses, problem):
                return 100.0
            return _mean_reflectance_percent(materials, thicknesses, problem)
        
        bounds_refine = [(0.0, max_layer_nm)] * len(best_thicknesses)
        
        try:
            result = minimize(
                _objective_refine,
                best_thicknesses,
                method='L-BFGS-B',
                bounds=bounds_refine,
                options={'maxiter': 100}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_thicknesses = list(result.x)
                
        except Exception as e:
            pass
    
    # Ensure the final result is valid
    if not _is_valid(best_materials, best_thicknesses, problem):
        # Fallback to the seed
        best_materials = ["L"]
        best_thicknesses = [centre / (4.0 * materials_dict["L"])]
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
