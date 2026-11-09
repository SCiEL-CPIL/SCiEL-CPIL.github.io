"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a broadband antireflection coating using a periodic alternating L/H stack
    with a linear gradient factor, optimized via differential evolution.
    
    The stack is represented as N pairs of (L, H) layers, where each pair has thicknesses
    that follow a linear gradient from the air side to the substrate side.
    Specifically, for pair i (0-indexed), the L layer thickness is t0 * (1 - g*i/(N-1))
    and the H layer thickness is t0 * (1 + g*i/(N-1)), or similar parameterization.
    
    Actually, let's use a cleaner parameterization:
    - base_thickness: the nominal thickness for each layer
    - gradient: a factor that scales thickness linearly across the stack
    
    For a stack of 2*N layers (N L-layers and N H-layers alternating, starting with L),
    layer i (0-indexed) has thickness:
      if i is even (L layer): base_thickness * (1 + gradient * (i - (2*N-1)/2) / ((2*N-1)/2))
      if i is odd (H layer):  base_thickness * (1 + gradient * (i - (2*N-1)/2) / ((2*N-1)/2))
    
    Wait, let's simplify. Let's fix the number of layer pairs to be within the 20 layer limit.
    Let's try N_pairs = 5, so 10 layers total. Or maybe N_pairs = 6, 12 layers.
    
    Let's parameterize as:
    - N_pairs: number of LH pairs (fixed, say 5 or 6)
    - t0: base thickness
    - g: gradient factor in [-1, 1]
    
    For layer index j from 0 to 2*N_pairs-1:
      position_factor = (j - (2*N_pairs - 1)/2) / (N_pairs - 0.5)  # ranges from -1 to 1
      thickness_j = t0 * (1 + g * position_factor)
    
    This ensures a smooth linear gradient in thickness across the stack.
    
    We'll use differential evolution to optimize t0 and g.
    """
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    nL = materials["L"]
    nH = materials["H"]
    
    # Fix number of pairs to stay within max_layers=20
    # Let's use 5 pairs = 10 layers, leaving room for thickness variations
    N_pairs = 5
    n_layers = 2 * N_pairs
    
    # Wavelengths for evaluation
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    def compute_reflectance(t0, g):
        """Compute mean reflectance for given t0 and g."""
        # Generate thicknesses
        thicknesses = []
        labels = []
        for j in range(n_layers):
            position_factor = (j - (n_layers - 1) / 2.0) / (N_pairs - 0.5)
            t = t0 * (1.0 + g * position_factor)
            # Clamp to valid range
            t = np.clip(t, 0.0, max_layer_nm)
            thicknesses.append(t)
            # Alternate L and H, starting with L
            labels.append('L' if j % 2 == 0 else 'H')
        
        total_thickness = sum(thicknesses)
        if total_thickness > max_total_nm:
            return 100.0  # Penalize invalid designs
        
        # Compute reflectance using characteristic matrix method
        reflectances = []
        for lam in wavelengths:
            # Complex refractive indices (real since lossless)
            n_ambient = ambient_index + 0j
            n_substrate = substrate_index + 0j
            n_layers_arr = np.array([nL if l == 'L' else nH for l in labels], dtype=complex)
            t_layers_arr = np.array(thicknesses, dtype=float)
            
            # Optical admittances
            # For normal incidence, the characteristic matrix for each layer is:
            # M_j = [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
            # where delta_j = 2*pi*n_j*t_j/lam
            # The total matrix is the product of all layer matrices
            # Then reflectance R = |(Y_in - Y_0)/(Y_in + Y_0)|^2 where Y_in is the input admittance
            
            # Start with identity matrix
            M_total = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
            
            for n_j, t_j in zip(n_layers_arr, t_layers_arr):
                delta = 2.0 * np.pi * n_j * t_j / lam
                c = np.cos(delta)
                s = np.sin(delta)
                M_j = np.array([[c, 1j * s / n_j],
                                 [1j * n_j * s, c]], dtype=complex)
                M_total = M_total @ M_j
            
            # Input admittance
            B, C = M_total[0, 0], M_total[0, 1]
            D, E = M_total[1, 0], M_total[1, 1]
            
            # The input admittance Y_in satisfies:
            # E = B * Y_0 + D * Y_sub ... actually let me recall the formula
            # 
            # The characteristic matrix relates the field at the input to the field at the output:
            # [E]   = M_total @ [E']
            # [H]        [H']
            # where E' and H' are at the substrate side.
            # 
            # At the substrate, H' = Y_sub * E' where Y_sub = n_substrate (for normal incidence, TE)
            # 
            # So: E = B * E' + C * H' = (B + C * Y_sub) * E'
            #      H = D * E' + E * H' = (D + E * Y_sub) * E'
            # 
            # Input admittance Y_in = H / E = (D + E * Y_sub) / (B + C * Y_sub)
            # 
            # Wait, I'm using E for both the matrix element and the electric field. Let me rename.
            # Let matrix elements be M11, M12, M21, M22.
            # 
            # Y_in = (M21 + M22 * Y_sub) / (M11 + M12 * Y_sub)
            
            Y_sub = n_substrate
            Y_in = (M_total[1, 0] + M_total[1, 1] * Y_sub) / (M_total[0, 0] + M_total[0, 1] * Y_sub)
            
            # Reflectance
            r = (Y_in - n_ambient) / (Y_in + n_ambient)
            R = np.abs(r) ** 2 * 100.0  # in percent
            reflectances.append(R)
        
        return np.mean(reflectances)
    
    # Define bounds for differential evolution
    # t0: base thickness, reasonable range 50-400 nm
    # g: gradient factor, range [-1, 1]
    bounds = [(50.0, 400.0), (-1.0, 1.0)]
    
    # Objective function for differential evolution (minimize)
    def objective(params):
        t0, g = params
        return compute_reflectance(t0, g)
    
    # Run differential evolution
    result = differential_evolution(
        objective,
        bounds,
        maxiter=100,
        popsize=15,
        tol=1e-6,
        seed=42,
        polish=False
    )
    
    t0_opt, g_opt = result.x
    
    # Generate final design
    thicknesses = []
    labels = []
    for j in range(n_layers):
        position_factor = (j - (n_layers - 1) / 2.0) / (N_pairs - 0.5)
        t = t0_opt * (1.0 + g_opt * position_factor)
        t = np.clip(t, 0.0, max_layer_nm)
        thicknesses.append(float(t))
        labels.append('L' if j % 2 == 0 else 'H')
    
    # Verify constraints
    if len(labels) > max_layers:
        # Truncate if needed
        labels = labels[:max_layers]
        thicknesses = thicknesses[:max_layers]
    
    total_thickness = sum(thicknesses)
    if total_thickness > max_total_nm:
        # Scale down
        scale = max_total_nm / total_thickness
        thicknesses = [t * scale for t in thicknesses]
    
    return labels, thicknesses
# EVOLVE-BLOCK-END
