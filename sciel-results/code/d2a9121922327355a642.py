"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def _compute_reflectance(wavelengths, n_layers, n_substrate, n_ambient, thicknesses):
    """
    Compute reflectance at given wavelengths using characteristic matrix method.
    
    Parameters:
    - wavelengths: array of wavelengths in nm
    - n_layers: array of refractive indices for each layer
    - n_substrate: substrate refractive index
    - n_ambient: ambient refractive index
    - thicknesses: array of physical thicknesses in nm
    
    Returns:
    - reflectances: array of reflectance values (0 to 1)
    """
    n = len(n_layers)
    # Start with identity matrix
    M = np.eye(2, dtype=complex)
    
    for i in range(n):
        n_i = n_layers[i]
        d_i = thicknesses[i]
        # Phase thickness
        delta = 2.0 * np.pi * n_i * d_i / wavelengths
        c = np.cos(delta)
        s = np.sin(delta)
        # Layer matrix
        M_i = np.array([[c, 1j * s / n_i],
                        [1j * n_i * s, c]], dtype=complex)
        M = M_i @ M
    
    # Total matrix M = [[M11, M12], [M21, M22]]
    M11, M12 = M[0, 0], M[0, 1]
    M21, M22 = M[1, 0], M[1, 1]
    
    # Admittance
    Y = M21 / M11
    
    # Reflectance
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r) ** 2
    
    return R

def design(problem):
    """
    Design a broadband antireflection coating using optimization.
    """
    n_ambient = problem["ambient_index"]  # 1.0
    n_substrate = problem["substrate_index"]  # 1.52
    materials = problem["materials"]  # {'L': 1.38, 'H': 2.35}
    band_nm = problem["band_nm"]  # [400, 800]
    max_layers = problem["max_layers"]  # 20
    max_total_nm = problem["max_total_nm"]  # 1000
    max_layer_nm = problem["max_layer_nm"]  # 400
    
    # Wavelengths for evaluation (401 equally spaced points)
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Try different numbers of layers and find the best
    best_result = None
    best_score = np.inf
    
    # Try various layer counts and patterns
    # Pattern: start with L, alternate L/H
    # We want to go from air (1.0) to glass (1.52), so we want to gradually increase index
    # L=1.38, H=2.35, substrate=1.52
    
    # Let's try different numbers of layers
    for n_layers in range(2, min(max_layers, 12) + 1, 2):  # Try even numbers of layers
        # Create layer index pattern: L, H, L, H, ...
        # Start with L (lower index, closer to air)
        n_layers_arr = np.zeros(n_layers)
        for i in range(n_layers):
            if i % 2 == 0:
                n_layers_arr[i] = materials["L"]
            else:
                n_layers_arr[i] = materials["H"]
        
        # Initial guess: quarter-wave at center wavelength
        center_wl = 0.5 * (band_nm[0] + band_nm[1])
        initial_thicknesses = np.array([center_wl / (4.0 * n) for n in n_layers_arr])
        
        # Define objective function
        def objective(thicknesses):
            # Check constraints
            if np.any(thicknesses < 0) or np.any(thicknesses > max_layer_nm):
                return 1e6
            if np.sum(thicknesses) > max_total_nm:
                return 1e6
            
            R = _compute_reflectance(wavelengths, n_layers_arr, n_substrate, n_ambient, thicknesses)
            return np.mean(R)
        
        # Bounds for each thickness
        bounds = [(0.0, max_layer_nm)] * n_layers
        
        # Use Nelder-Mead for small problems, or L-BFGS-B
        try:
            result = minimize(objective, initial_thicknesses, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 200, 'ftol': 1e-8})
            
            if result.success or result.fun < best_score:
                # Check if result is valid
                if (np.all(result.x >= 0) and np.all(result.x <= max_layer_nm) and 
                    np.sum(result.x) <= max_total_nm):
                    score = result.fun
                    if score < best_score:
                        best_score = score
                        # Convert to list of labels and thicknesses
                        labels = []
                        for i in range(n_layers):
                            if n_layers_arr[i] == materials["L"]:
                                labels.append("L")
                            else:
                                labels.append("H")
                        best_result = (labels, list(result.x))
        except Exception:
            continue
    
    # If no good result found, fall back to a simple quarter-wave stack
    if best_result is None:
        center_wl = 0.5 * (band_nm[0] + band_nm[1])
        # Use a 2-layer stack: L then H
        labels = ["L", "H"]
        thicknesses = [center_wl / (4.0 * materials["L"]), center_wl / (4.0 * materials["H"])]
        best_result = (labels, thicknesses)
    
    return best_result
# EVOLVE-BLOCK-END
