"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def _reflectance_spectrum(materials, thicknesses, ambient_index, substrate_index,
                          band_nm, n_points=401):
    """
    Compute normal-incidence reflectance (in percent) over a wavelength band
    using the characteristic-matrix method.
    """
    lam_min, lam_max = band_nm
    wavelengths = np.linspace(lam_min, lam_max, n_points)
    n_L = materials['L']
    n_H = materials['H']
    
    # Build the matrix product for each wavelength
    # Characteristic matrix for a layer:
    # M = [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
    # where delta = 2*pi*n*d / lambda
    
    reflectances = np.zeros(len(wavelengths))
    
    for i, lam in enumerate(wavelengths):
        # Start with identity matrix
        M_total = np.array([[1.0+0j, 0.0+0j], [0.0+0j, 1.0+0j]], dtype=complex)
        
        # Multiply by each layer's matrix from substrate side to air side? 
        # Actually, the standard convention: we go from air to substrate.
        # The characteristic matrix of the stack is the product of individual layer matrices.
        # For a stack of layers 1, 2, ..., N (from air to substrate):
        # M_stack = M_1 * M_2 * ... * M_N
        # Then the admittance Y = (C/D) where M_stack = [[A, B], [C, D]]
        # Reflectance R = |(n0 - Y)/(n0 + Y)|^2  (for normal incidence, n0 = ambient index)
        # Wait, let me be careful with the convention.
        
        # Standard characteristic matrix method:
        # For each layer j with refractive index n_j and thickness d_j:
        # delta_j = 2 * pi * n_j * d_j / lambda
        # M_j = [[cos(delta_j), 1j * sin(delta_j) / n_j],
        #        [1j * n_j * sin(delta_j), cos(delta_j)]]
        
        # The total matrix is M = M_1 @ M_2 @ ... @ M_N (layers ordered air to substrate)
        # Then Y = D / C ... no wait.
        
        # Let M = [[A, B], [C, D]]
        # The optical admittance at the front surface is Y = (C + D*n_s) / (A + B*n_s) ... no.
        
        # Actually, the standard formula:
        # If the substrate has index n_s, the admittance looking into the stack from the substrate side is n_s.
        # The admittance looking into the stack from the air side is:
        # Y = (n_s * A + C) / (n_s * B + D) ... hmm, let me derive or recall correctly.
        
        # From Heavens "Optical Properties of Thin Solid Films":
        # The characteristic matrix of the stack is M = M_1 M_2 ... M_N
        # where M_j is as defined above.
        # Then the reflectance is:
        # r = (n_0 * (A + B*n_s) - (C + D*n_s)) / (n_0 * (A + B*n_s) + (C + D*n_s))
        # R = |r|^2
        
        # Let me verify with a single quarter-wave layer.
        # n_0 = 1, n_s = 1.52, n_1 = 1.38, d_1 = lambda_0/(4*n_1)
        # At lambda_0: delta_1 = pi/2, cos=0, sin=1
        # M_1 = [[0, 1j/n_1], [1j*n_1, 0]]
        # A=0, B=1j/n_1, C=1j*n_1, D=0
        # n_0*(A+B*n_s) = 1*(0 + 1j*n_s/n_1) = 1j*n_s/n_1
        # C + D*n_s = 1j*n_1 + 0 = 1j*n_1
        # r = (1j*n_s/n_1 - 1j*n_1) / (1j*n_s/n_1 + 1j*n_1) = (n_s/n_1 - n_1)/(n_s/n_1 + n_1) = (n_s - n_1^2)/(n_s + n_1^2)
        # For n_1 = sqrt(n_0*n_s) = sqrt(1.52) ≈ 1.233, r=0. But n_1=1.38, so:
        # r = (1.52 - 1.38^2)/(1.52 + 1.38^2) = (1.52 - 1.9044)/(1.52 + 1.9044) = -0.3844/3.4244 ≈ -0.1122
        # R ≈ 0.0126 = 1.26%. Bare glass R = ((1.52-1)/(1.52+1))^2 = (0.52/2.52)^2 ≈ 0.0426 = 4.26%. Good.
        
        A = 1.0+0j
        B = 0.0+0j
        C = 0.0+0j
        D = 1.0+0j
        
        for j in range(len(thicknesses)):
            n_j = n_L if materials_labels[j] == 'L' else n_H
            d_j = thicknesses[j]
            delta = 2.0 * np.pi * n_j * d_j / lam
            c = np.cos(delta)
            s = np.sin(delta)
            M_j = np.array([[c, 1j*s/n_j], [1j*n_j*s, c]], dtype=complex)
            # M_total = M_total @ M_j
            A_new = A*c + B*1j*n_j*s
            B_new = A*1j*s/n_j + B*c
            C_new = C*c + D*1j*n_j*s
            D_new = C*1j*s/n_j + D*c
            A, B, C, D = A_new, B_new, C_new, D_new
        
        numerator = ambient_index * (A + B * substrate_index) - (C + D * substrate_index)
        denominator = ambient_index * (A + B * substrate_index) + (C + D * substrate_index)
        r = numerator / denominator
        reflectances[i] = 100.0 * np.abs(r)**2
    
    return np.mean(reflectances)


def design(problem):
    """
    Design a broadband antireflection coating using a periodic sequence of
    alternating L and H layers with a linear gradient factor, optimized via
    differential evolution.
    """
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # We'll use up to max_layers layers, alternating L and H starting with L.
    # The thickness of layer i is: d_i = base_thickness * (1.0 + gradient_factor * i / (n_layers - 1))
    # This creates a smooth gradient from low to high optical density.
    # Actually, to enforce a smooth transition from low to high optical density,
    # we can vary the thicknesses linearly. But the indices are fixed (L and H alternate).
    # So "optical density" gradient means the product n*d varies smoothly.
    # Let's parameterize: d_i = base * (1 + g * i/(N-1)) where i=0..N-1
    # N is the number of layers, which we can also optimize (as an integer).
    
    # For simplicity in differential evolution, let's fix N to a reasonable value
    # or include it as a parameter. Let's try N as a continuous parameter that we round.
    
    # Parameters to optimize:
    # x[0] = base_thickness (nm)
    # x[1] = gradient_factor (dimensionless)
    # x[2] = n_layers (continuous, will be rounded to integer)
    
    # Bounds:
    # base_thickness: [0, max_layer_nm]
    # gradient_factor: [-1, 1] (can be positive or negative gradient)
    # n_layers: [1, max_layers]
    
    bounds = [(1.0, max_layer_nm), (-1.0, 1.0), (1.0, float(max_layers))]
    
    def objective(x):
        base_thickness = x[0]
        gradient_factor = x[1]
        n_layers_float = x[2]
        n_layers = int(round(n_layers_float))
        n_layers = max(1, min(max_layers, n_layers))
        
        if n_layers == 1:
            thicknesses = [base_thickness]
        else:
            thicknesses = []
            for i in range(n_layers):
                frac = i / (n_layers - 1)
                d = base_thickness * (1.0 + gradient_factor * frac)
                # Clamp to valid range
                d = max(0.0, min(max_layer_nm, d))
                thicknesses.append(d)
        
        # Check total thickness constraint
        total = sum(thicknesses)
        if total > max_total_nm + 1e-6:
            return 100.0  # Penalize infeasible solutions
        
        # Build materials labels: alternating L, H, L, H, ...
        materials_labels = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        
        # Compute reflectance
        refl = _reflectance_spectrum(materials, thicknesses, ambient_index, substrate_index,
                                      band_nm, n_points=401)
        return refl
    
    # Run differential evolution
    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=42,
        maxiter=100,
        popsize=15,
        tol=1e-6,
        polish=True
    )
    
    base_thickness = result.x[0]
    gradient_factor = result.x[1]
    n_layers_float = result.x[2]
    n_layers = int(round(n_layers_float))
    n_layers = max(1, min(max_layers, n_layers))
    
    if n_layers == 1:
        thicknesses = [base_thickness]
    else:
        thicknesses = []
        for i in range(n_layers):
            frac = i / (n_layers - 1)
            d = base_thickness * (1.0 + gradient_factor * frac)
            d = max(0.0, min(max_layer_nm, d))
            thicknesses.append(d)
    
    # Check total thickness
    total = sum(thicknesses)
    if total > max_total_nm:
        # Scale down thicknesses to fit within max_total_nm
        if total > 0:
            scale = max_total_nm / total
            thicknesses = [d * scale for d in thicknesses]
        # Re-clamp
        thicknesses = [max(0.0, min(max_layer_nm, d)) for d in thicknesses]
    
    materials_labels = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
    
    return materials_labels, thicknesses
# EVOLVE-BLOCK-END
