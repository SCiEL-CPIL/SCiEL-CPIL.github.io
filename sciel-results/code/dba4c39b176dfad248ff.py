"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def _reflectance_spectrum(thicknesses, labels, materials, ambient_index, substrate_index, wavelengths):
    """
    Compute normal-incidence reflectance spectrum using the characteristic matrix method.
    thicknesses: list of physical thicknesses in nm
    labels: list of 'L'/'H'
    materials: dict mapping label to refractive index
    wavelengths: array of wavelengths in nm
    Returns: array of reflectance in percent
    """
    n_layers = len(thicknesses)
    if n_layers == 0:
        # Bare substrate
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return (np.abs(r) ** 2) * 100.0
    
    # Precompute indices for each layer
    n_layers_arr = np.array([materials[lab] for lab in labels])
    t_layers_arr = np.array(thicknesses)
    
    # Wavelengths in vacuum (nm)
    lam = wavelengths
    
    # Phase delta for each layer: 2*pi*n*t/lam
    # Shape: (n_wavelengths, n_layers)
    delta = 2.0 * np.pi * n_layers_arr[None, :] * t_layers_arr[None, :] / lam[:, None]
    
    # Characteristic matrix for each layer:
    # M_j = [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
    # We need to multiply all matrices from air side to substrate side.
    
    n_wav = len(lam)
    
    # Initialize identity matrices
    # Each matrix is 2x2 complex
    # We'll accumulate the product
    
    # For efficiency, we can process wavelength by wavelength or use vectorization.
    # Let's do it wavelength by wavelength for clarity, but vectorize where possible.
    # Actually, let's use a loop over wavelengths since the matrix multiplication is small.
    
    reflectances = np.zeros(n_wav)
    
    for i, lam_i in enumerate(lam):
        # Accumulate the characteristic matrix
        # Start with identity
        M = np.array([[1.0 + 0.0j, 0.0 + 0.0j],
                      [0.0 + 0.0j, 1.0 + 0.0j]])
        
        for j in range(n_layers):
            n_j = n_layers_arr[j]
            t_j = t_layers_arr[j]
            delta_j = 2.0 * np.pi * n_j * t_j / lam_i
            
            cos_d = np.cos(delta_j)
            sin_d = np.sin(delta_j)
            
            # Layer matrix
            M_j = np.array([[cos_d, 1j * sin_d / n_j],
                            [1j * n_j * sin_d, cos_d]])
            
            M = M @ M_j
        
        # The substrate has index n_s, ambient has index n_0
        # The admittance Y = M[1,0] / M[0,0]
        # Reflectance r = (Y - n_0) / (Y + n_0)
        # R = |r|^2
        
        Y = M[1, 0] / M[0, 0]
        r = (Y - ambient_index) / (Y + ambient_index)
        R = np.abs(r) ** 2
        reflectances[i] = R * 100.0
    
    return reflectances

def design(problem):
    """
    Design a broadband antireflection coating using differential evolution.
    """
    band = problem["band_nm"]
    materials = problem["materials"]
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # We will try a few different stack sizes and pick the best one
    # Start with a reasonable number of layers, say 10, and maybe try 20
    
    # Wavelengths for evaluation (401 points)
    wavelengths = np.linspace(band[0], band[1], 401)
    
    best_mean_R = np.inf
    best_labels = None
    best_thick = None
    
    # Try different numbers of layers
    # The problem says "at most twenty" layers. We can choose any number up to 20.
    # Let's try a few candidates: 10, 15, 20
    # Actually, let's be smarter. Start with a good initial guess and optimize.
    
    # For differential evolution, we need to define the objective function.
    # The parameters are the thicknesses. The labels (L/H) are discrete.
    # We can fix a pattern of labels and optimize thicknesses.
    # A common effective pattern is alternating H/L or L/H.
    
    # Let's try a few patterns:
    # Pattern 1: Start with L, alternate
    # Pattern 2: Start with H, alternate
    
    # We'll try optimizing thicknesses for a fixed label pattern.
    
    def objective(thick_vec, labels):
        # Check total thickness
        total = np.sum(thick_vec)
        if total > max_total_nm:
            return 100.0  # Penalize heavily
        
        # Check individual thicknesses
        if np.any(thick_vec < 0) or np.any(thick_vec > max_layer_nm):
            return 100.0
        
        # Compute reflectance
        R = _reflectance_spectrum(thick_vec, labels, materials, ambient_index, substrate_index, wavelengths)
        return np.mean(R)
    
    # Try different numbers of layers and patterns
    for n_layers in [10, 15, 20]:
        # Pattern 1: L, H, L, H, ...
        labels1 = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        # Pattern 2: H, L, H, L, ...
        labels2 = ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
        
        for labels in [labels1, labels2]:
            # Define bounds for differential evolution
            # Each thickness in [0, max_layer_nm]
            bounds = [(0, max_layer_nm) for _ in range(n_layers)]
            
            # Initial population size
            # Use a moderate population size for speed
            pop_size = 50
            
            # Run differential evolution
            try:
                result = differential_evolution(
                    lambda x: objective(x, labels),
                    bounds,
                    seed=42,
                    maxiter=100,
                    popsize=pop_size,
                    tol=1e-4,
                    polish=True
                )
                
                mean_R = result.fun
                if mean_R < best_mean_R:
                    best_mean_R = mean_R
                    best_labels = labels
                    best_thick = result.x
            except Exception as e:
                pass
    
    # If no result was found (shouldn't happen), fall back to a simple design
    if best_labels is None:
        centre = 0.5 * (band[0] + band[1])
        return ["L"], [centre / (4.0 * materials["L"])]
    
    return best_labels, list(best_thick)
# EVOLVE-BLOCK-END
