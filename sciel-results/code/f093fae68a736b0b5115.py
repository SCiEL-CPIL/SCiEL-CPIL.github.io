"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

# EVOLVE-BLOCK-START

def _reflectance_spectrum(materials, thicknesses, band_nm, ambient_index, substrate_index):
    """Compute mean reflectance (%) over the band using characteristic-matrix method."""
    wl0, wl1 = band_nm
    wls = np.linspace(wl0, wl1, 401)
    n_air = ambient_index
    n_sub = substrate_index
    n_layers = len(materials)
    
    # Precompute refractive indices
    n_stack = np.array([materials[m] for m in materials])
    d_stack = np.array(thicknesses, dtype=float)
    
    # 401 wavelengths
    k = 2.0 * np.pi / wls  # 1/nm
    # Phase thicknesses: k * n * d  (shape: (401, n_layers))
    phase = k[:, None] * n_stack[None, :] * d_stack[None, :]
    
    # Build characteristic matrices for each layer
    # M_j = [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
    # We'll compute the product M_total = M_1 * M_2 * ... * M_n
    # Using complex arrays: shape (401, 2, 2)
    
    # Start with identity
    M = np.ones((len(wls), 2, 2), dtype=complex)
    
    for j in range(n_layers):
        cos_d = np.cos(phase[:, j])
        sin_d = np.sin(phase[:, j])
        n_j = n_stack[j]
        
        # M_j
        Mj = np.ones((len(wls), 2, 2), dtype=complex)
        Mj[:, 0, 0] = cos_d
        Mj[:, 0, 1] = 1j * sin_d / n_j
        Mj[:, 1, 0] = 1j * n_j * sin_d
        Mj[:, 1, 1] = cos_d
        
        # M = M @ Mj  (matrix multiply over last two dims)
        M = np.matmul(M, Mj)
    
    # M_total = [[m11, m12], [m21, m22]]
    m11 = M[:, 0, 0]
    m12 = M[:, 0, 1]
    m21 = M[:, 1, 0]
    m22 = M[:, 1, 1]
    
    # Reflectance: R = |(n_air*m11 + m21 - n_sub*m12 - m22) / (n_air*m11 + m21 + n_sub*m12 + m22)|^2
    # Wait, the standard formula for normal incidence:
    # The admittance Y = (n_air * m11 + m21) / (n_air * m12 + m22) ... 
    # Actually let me use the standard formula:
    # R = | (Y_air - Y_stack) / (Y_air + Y_stack) |^2
    # where Y_stack = (m21 + n_sub * m22) / (m11 + n_sub * m12)  ... no.
    
    # Let me use the standard characteristic matrix method:
    # For a stack with ambient n_0 and substrate n_s:
    # The effective admittance seen from the air side is:
    # Y = (n_0 * m21 + m22) / (n_0 * m11 + m12) ... 
    # 
    # Actually, the standard formula is:
    # R = | (m11 + m12/n_s - n_0*m21 - n_0*m22/n_s) / (m11 + m12/n_s + n_0*m21 + n_0*m22/n_s) |^2
    # 
    # Let me use a cleaner formulation. The characteristic matrix for the stack is M = M_1 M_2 ... M_n
    # where each M_j = [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
    # 
    # The reflectance is:
    # r = (n_0 * (m11 + m12/n_s) - (m21 + m22/n_s)) / (n_0 * (m11 + m12/n_s) + (m21 + m22/n_s))
    # R = |r|^2
    
    # Let me verify with bare glass (no layers):
    # M = I, so m11=1, m12=0, m21=0, m22=1
    # r = (n_0 * (1 + 0) - (0 + 1/n_s)) / (n_0 * (1 + 0) + (0 + 1/n_s))
    # r = (n_0 - 1/n_s) / (n_0 + 1/n_s)
    # For n_0=1, n_s=1.52: r = (1 - 0.658) / (1 + 0.658) = 0.342/1.658 = 0.206
    # R = 0.206^2 = 0.0425 = 4.25% ✓
    
    numerator = n_air * (m11 + m12 / n_sub) - (m21 + m22 / n_sub)
    denominator = n_air * (m11 + m12 / n_sub) + (m21 + m22 / n_sub)
    r = numerator / denominator
    R = np.abs(r) ** 2 * 100.0  # in percent
    
    return np.mean(R)


def _objective(params, problem, max_layers, max_total_nm, max_layer_nm):
    """Objective function for optimization. Returns mean reflectance or penalty."""
    n_layers = len(params) // 2
    # params: [t0, t1, t2, ..., t_{n-1}]  (thicknesses)
    # The layer sequence is fixed: alternating L, H, L, H, ... starting with L
    thicknesses = params
    materials = []
    for i in range(n_layers):
        materials.append('L' if i % 2 == 0 else 'H')
    
    # Check constraints
    if any(t < 0 or t > max_layer_nm for t in thicknesses):
        return 100.0  # penalty
    if sum(thicknesses) > max_total_nm:
        return 100.0  # penalty
    
    return _reflectance_spectrum(materials, thicknesses, problem["band_nm"], 
                                  problem["ambient_index"], problem["substrate_index"])


def _optimize_for_n_layers(n_layers, problem, deadline, best_result, best_score):
    """Optimize thicknesses for a given number of layers with alternating L/H pattern."""
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    bounds = [(0, max_layer_nm)] * n_layers
    
    def obj(params):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 100.0
        return _objective(params, problem, max_layers_global, max_total_nm, max_layer_nm)
    
    # Use differential evolution
    try:
        result = differential_evolution(
            obj, bounds, maxiter=50, popsize=10, tol=1e-6, 
            mutation=(0.3, 1.0), recombination=0.9, seed=42,
            polish=False, init='latinhypercube'
        )
        
        if result.success and result.fun < best_score[0]:
            thicknesses = list(result.x)
            materials = []
            for i in range(n_layers):
                materials.append('L' if i % 2 == 0 else 'H')
            best_score[0] = result.fun
            best_result[0] = (materials, thicknesses)
    except Exception:
        pass
    
    # Polish with Nelder-Mead if we have time
    elapsed = time.time() - start_time
    if elapsed < deadline - 5 and best_result[0] is not None and len(best_result[0][0]) == n_layers:
        try:
            res2 = minimize(
                obj, result.x, method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if res2.success and res2.fun < best_score[0]:
                thicknesses = list(res2.x)
                materials = []
                for i in range(n_layers):
                    materials.append('L' if i % 2 == 0 else 'H')
                best_score[0] = res2.fun
                best_result[0] = (materials, thicknesses)
        except Exception:
            pass


def design(problem):
    """Design antireflection coating using global optimization over alternating stacks."""
    global start_time, max_layers_global
    
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s kill
    
    max_layers_global = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    best_result = [None]
    best_score = [float('inf')]
    
    # Try different numbers of layers
    # Start with a reasonable number and work up
    for n_layers in range(2, max_layers_global + 1, 2):  # Even numbers for symmetric stacks
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        _optimize_for_n_layers(n_layers, problem, deadline, best_result, best_score)
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break
    
    # If no valid result found, return a simple quarter-wave stack
    if best_result[0] is None:
        centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
        return ["L"], [centre / (4.0 * problem["materials"]["L"])]
    
    return best_result[0]

# EVOLVE-BLOCK-END
