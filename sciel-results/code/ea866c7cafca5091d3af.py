"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def _compute_mean_reflectance(labels, thicknesses, materials, band_nm, ambient_index, substrate_index, n_points=401):
    """Compute mean reflectance (%) over the band using characteristic-matrix method."""
    nL = materials['L']
    nH = materials['H']
    n_sub = substrate_index
    n_amb = ambient_index
    
    wavelengths = np.linspace(band_nm[0], band_nm[1], n_points)
    reflectances = np.zeros(n_points)
    
    n_layers = len(thicknesses)
    n_indices = np.zeros(n_layers)
    for i, label in enumerate(labels):
        n_indices[i] = nL if label == 'L' else nH
    
    for idx, lam in enumerate(wavelengths):
        M_total = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        
        for j in range(n_layers):
            n_j = n_indices[j]
            d_j = thicknesses[j]
            k_j = 2.0 * np.pi * n_j / lam
            phase = d_j * k_j
            
            c = np.cos(phase)
            s = np.sin(phase)
            
            M_j = np.array([[c, 1j * s / n_j],
                            [1j * n_j * s, c]], dtype=complex)
            
            M_total = M_total @ M_j
        
        B = M_total[0, 0] * 1.0 + M_total[0, 1] * n_sub
        C = M_total[1, 0] * 1.0 + M_total[1, 1] * n_sub
        
        Y = B / C
        r = (n_amb - Y) / (n_amb + Y)
        R = np.abs(r) ** 2
        reflectances[idx] = R * 100.0
    
    return np.mean(reflectances)

def design(problem):
    """Design a broadband antireflection coating using periodic L/H stack with DE optimization."""
    band_nm = problem["band_nm"]
    materials = problem["materials"]
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    nL = materials['L']
    nH = materials['H']
    
    best_score = np.inf
    best_labels = None
    best_thicknesses = None
    
    # Try different numbers of layers
    for n_layers in [8, 10, 12]:
        # Alternate L and H starting with L
        labels = []
        for i in range(n_layers):
            if i % 2 == 0:
                labels.append('L')
            else:
                labels.append('H')
        
        def make_objective(n_layers):
            def objective(params):
                t0, g = params
                if t0 < 0:
                    return 1e10
                thicknesses = []
                for i in range(n_layers):
                    if n_layers > 1:
                        t_i = t0 * (1.0 + g * i / (n_layers - 1))
                    else:
                        t_i = t0
                    if t_i < 0 or t_i > max_layer_nm:
                        return 1e10
                    thicknesses.append(t_i)
                if sum(thicknesses) > max_total_nm:
                    return 1e10
                return _compute_mean_reflectance(labels, thicknesses, materials, band_nm, ambient_index, substrate_index)
            return objective
        
        objective = make_objective(n_layers)
        
        # Bounds: t0 in [10, 200], g in [-1.0, 1.0]
        bounds = [(10.0, 200.0), (-1.0, 1.0)]
        
        try:
            result = differential_evolution(objective, bounds, maxiter=80, popsize=15, tol=1e-5, seed=42, polish=False)
            if result.fun < best_score:
                best_score = result.fun
                t0, g = result.x
                thicknesses = []
                for i in range(n_layers):
                    if n_layers > 1:
                        t_i = t0 * (1.0 + g * i / (n_layers - 1))
                    else:
                        t_i = t0
                    thicknesses.append(t_i)
                best_labels = labels[:]
                best_thicknesses = thicknesses[:]
        except Exception as e:
            continue
    
    if best_labels is None:
        centre = 0.5 * (band_nm[0] + band_nm[1])
        return ["L"], [centre / (4.0 * nL)]
    
    return best_labels, best_thicknesses
# EVOLVE-BLOCK-END
